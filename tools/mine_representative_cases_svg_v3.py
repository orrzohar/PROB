import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main_open_world import get_args_parser, build_datasets
from models import build_model
from util import box_ops
from util.misc import nested_tensor_from_tensor_list
from tools.figure_svg_utils import write_gallery_svg

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CATEGORY_COLORS = {
    'known': (0, 166, 90),
    'unknown': (216, 27, 96),
    'odqe_salient': (30, 136, 229),
}


def to_numpy_image(image_tensor, target_hw=None):
    image = image_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = np.clip(image, 0.0, 1.0)
    if target_hw is not None:
        height, width = int(target_hw[0]), int(target_hw[1])
        image = image[:height, :width]
    return (image * 255).astype(np.uint8)


def _energy_to_prob(energy, temperature):
    return torch.exp(-temperature * energy.detach()).clamp(min=1e-6, max=1.0)


def compute_final_scores(outputs, args, invalid_cls_logits):
    hidden_dim = float(getattr(args, 'hidden_dim', 256))
    obj_temp = float(getattr(args, 'obj_temp', 1.0)) / hidden_dim
    known_temp = float(getattr(args, 'uod_known_temp', getattr(args, 'obj_temp', 1.0))) / hidden_dim

    pred_logits = outputs['pred_logits'].detach()
    pred_boxes = outputs['pred_boxes'].detach()
    pred_obj = outputs.get('pred_obj')
    pred_known = outputs.get('pred_known')

    class_prob = pred_logits.sigmoid().clone()
    if invalid_cls_logits:
        class_prob[:, :, invalid_cls_logits] = 0.0
    if class_prob.shape[-1] > 0:
        class_prob[:, :, -1] = 0.0
    max_known = class_prob.max(-1).values
    argmax_known = class_prob.argmax(-1)

    if pred_obj is not None:
        obj_prob = _energy_to_prob(pred_obj, obj_temp)
    else:
        obj_prob = torch.ones_like(max_known)

    if pred_known is not None:
        known_prob = _energy_to_prob(pred_known, known_temp)
        unknown_prob = (1.0 - known_prob).clamp(min=0.0, max=1.0)
    else:
        known_prob = torch.ones_like(obj_prob)
        unknown_prob = torch.zeros_like(obj_prob)

    known_score = obj_prob * known_prob * max_known
    unknown_score = obj_prob * unknown_prob * float(getattr(args, 'uod_postprocess_unknown_scale', 20.0))
    return {
        'boxes': pred_boxes[0],
        'obj_prob': obj_prob[0],
        'known_prob': known_prob[0],
        'unknown_prob': unknown_prob[0],
        'max_known': max_known[0],
        'argmax_known': argmax_known[0],
        'known_score': known_score[0],
        'unknown_score': unknown_score[0],
    }


class ODQEGateRecorder:
    def __init__(self, model):
        self.records = []
        self.patches = []
        self.decay = None
        if hasattr(model, 'odqe_layer_decay'):
            try:
                self.decay = model.odqe_layer_decay.detach().cpu().numpy().astype(np.float32)
            except Exception:
                self.decay = None
        for name, module in model.named_modules():
            if name.startswith('gate_mlp.') and hasattr(module, 'layers') and callable(getattr(module, 'forward', None)):
                self._patch_module(name, module)

    @staticmethod
    def _parse_layer_id(name):
        try:
            return int(name.split('gate_mlp.')[1].split('.')[0])
        except Exception:
            return -1

    def _patch_module(self, name, module):
        original_forward = module.forward
        recorder = self
        layer_id = self._parse_layer_id(name)

        def wrapped_forward(x):
            out = original_forward(x)
            gate = torch.sigmoid(out)
            decay_value = 1.0
            if recorder.decay is not None and 0 <= layer_id < len(recorder.decay):
                decay_value = float(recorder.decay[layer_id])
            recorder.records.append({'layer_id': layer_id, 'effective_gate': (gate * decay_value).detach().cpu()})
            return out

        module.forward = wrapped_forward
        self.patches.append((module, original_forward))

    def clear(self):
        self.records.clear()

    def restore(self):
        for module, original_forward in self.patches:
            module.forward = original_forward


def ordered_gate_records(records):
    ordered = sorted(records, key=lambda item: item['layer_id'])
    return [item for item in ordered if item['layer_id'] >= 0]


def _to_cpu_target(target):
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in target.items()}


def _to_device_target(target, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in target.items()}


def _effective_pseudo_epoch(args, checkpoint, criterion):
    override = getattr(args, 'pseudo_epoch', -1)
    if override is not None and int(override) >= 0:
        return int(override)
    ckpt_epoch = int(checkpoint.get('epoch', -1)) if isinstance(checkpoint, dict) else -1
    min_epoch = int(getattr(criterion, 'uod_start_epoch', 0)) + int(getattr(criterion, 'uod_neg_warmup_epochs', 0)) + 1
    return max(ckpt_epoch, min_epoch)


def _future_unknown_mask(target, args):
    current_known_upper = int(getattr(args, 'PREV_INTRODUCED_CLS', 0)) + int(getattr(args, 'CUR_INTRODUCED_CLS', 0))
    unknown_label = int(getattr(args, 'num_classes', 0) - 1)
    labels = target['labels']
    return (labels >= current_known_upper) & (labels < unknown_label)


def _current_known_mask(target, args):
    current_known_upper = int(getattr(args, 'PREV_INTRODUCED_CLS', 0)) + int(getattr(args, 'CUR_INTRODUCED_CLS', 0))
    labels = target['labels']
    return labels < current_known_upper


def _compute_overlap_to_gt(box_cxcywh, gt_boxes_cxcywh):
    if gt_boxes_cxcywh.numel() == 0:
        return 0.0
    box_xyxy = box_ops.box_cxcywh_to_xyxy(box_cxcywh.unsqueeze(0))
    gt_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes_cxcywh)
    return float(box_ops.box_iou(box_xyxy, gt_xyxy)[0].max().item())


def _select_best_known_query(scores, device_target, criterion, args):
    known_gt_mask = _current_known_mask(device_target, args)
    known_gt_boxes = device_target['boxes'][known_gt_mask]
    order = torch.argsort(scores['known_score'], descending=True).tolist()
    best = None
    for q in order:
        if not criterion._is_valid_geometry(scores['boxes'][q].detach().cpu()):
            continue
        overlap = _compute_overlap_to_gt(scores['boxes'][q], known_gt_boxes)
        rank_score = float(scores['known_score'][q].item()) * max(overlap, 1e-4)
        if best is None or rank_score > best[1]:
            best = (int(q), rank_score, overlap)
    return best


def _select_unknown_candidates(outputs, device_target, scores, criterion, epoch):
    matcher_inputs = {'pred_logits': outputs['pred_logits'].detach(), 'pred_boxes': outputs['pred_boxes'].detach()}
    indices = criterion.matcher(matcher_inputs, [device_target])
    dummy_pos_indices, _, dummy_pos_weights, _, _, _ = criterion._mine_uod_pseudo(outputs, [device_target], indices, epoch)
    selected_q = dummy_pos_indices[0] if len(dummy_pos_indices) > 0 else []
    selected_w = dummy_pos_weights[0] if len(dummy_pos_weights) > 0 else []
    candidates = []
    for local_idx, q in enumerate(selected_q):
        weight = float(selected_w[local_idx]) if local_idx < len(selected_w) else 1.0
        score = float(scores['unknown_score'][q].item() * weight)
        candidates.append((int(q), score, weight))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def _filter_candidates_by_future_unknown_gt(candidates, scores, device_target, args, min_iou):
    future_mask = _future_unknown_mask(device_target, args)
    future_boxes = device_target['boxes'][future_mask]
    if future_boxes.numel() == 0:
        return []
    filtered = []
    for q, base_score, weight in candidates:
        iou = _compute_overlap_to_gt(scores['boxes'][q], future_boxes)
        if iou < float(min_iou):
            continue
        filtered.append((q, base_score * iou, weight, iou))
    filtered.sort(key=lambda item: item[1], reverse=True)
    return filtered


def _select_best_odqe_query(candidates, gate_records, scores):
    if not candidates or not gate_records:
        return None
    best = None
    for q, _, weight, iou in candidates:
        eff_gate_mean = float(np.mean([record['effective_gate'][0, q].mean().item() for record in gate_records]))
        signal = eff_gate_mean * float(scores['unknown_prob'][q].item()) * float(scores['obj_prob'][q].item()) * float(iou)
        if best is None or signal > best[1]:
            best = (int(q), float(signal), eff_gate_mean, weight, iou)
    return best


def make_case_entry(category, sample_index, target, scores, query_index, category_score, gate_records, selection_source='unknown_pseudo', pseudo_weight=None, gt_overlap=None):
    image_id = int(target['image_id'].item()) if 'image_id' in target else int(sample_index)
    gate_mean = None
    gate_peak = None
    gate_depth_delta = None
    if gate_records:
        per_layer_gate = [float(record['effective_gate'][0, query_index].mean().item()) for record in gate_records]
        gate_mean = float(np.mean(per_layer_gate))
        gate_peak = float(np.max(per_layer_gate))
        gate_depth_delta = float(per_layer_gate[-1] - per_layer_gate[0]) if len(per_layer_gate) >= 2 else 0.0
    return {
        'category': category,
        'sample_index': int(sample_index),
        'image_id': image_id,
        'query_index': int(query_index),
        'category_score': float(category_score),
        'selection_source': selection_source,
        'pseudo_weight': None if pseudo_weight is None else float(pseudo_weight),
        'gt_overlap': None if gt_overlap is None else float(gt_overlap),
        'obj_prob': float(scores['obj_prob'][query_index].item()),
        'known_prob': float(scores['known_prob'][query_index].item()),
        'unknown_prob': float(scores['unknown_prob'][query_index].item()),
        'max_known': float(scores['max_known'][query_index].item()),
        'known_score': float(scores['known_score'][query_index].item()),
        'unknown_score': float(scores['unknown_score'][query_index].item()),
        'argmax_known': int(scores['argmax_known'][query_index].item()),
        'box_cxcywh': [float(v) for v in scores['boxes'][query_index].detach().cpu().tolist()],
        'gate_mean': gate_mean,
        'gate_peak': gate_peak,
        'gate_depth_delta': gate_depth_delta,
    }


def update_top_cases(top_cases, entry, top_k):
    category = entry['category']
    top_cases[category].append(entry)
    top_cases[category] = sorted(top_cases[category], key=lambda item: item['category_score'], reverse=True)[:top_k]


def draw_case_tile(dataset, entry, category, tile_size=420):
    image, target = dataset[entry['sample_index']]
    image_hw = target['size'].tolist() if 'size' in target else image.shape[-2:]
    image_np = to_numpy_image(image, image_hw)
    h, w = image_np.shape[:2]
    box = np.array(entry['box_cxcywh'], dtype=np.float32)[None, :]
    box_xyxy = box_ops.box_cxcywh_to_xyxy(torch.from_numpy(box)).numpy()[0]
    box_xyxy[[0, 2]] *= w
    box_xyxy[[1, 3]] *= h
    pil = Image.fromarray(image_np).convert('RGB').resize((tile_size, tile_size))
    draw = ImageDraw.Draw(pil)
    sx = tile_size / max(float(w), 1.0)
    sy = tile_size / max(float(h), 1.0)
    x1, y1, x2, y2 = box_xyxy
    draw.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=CATEGORY_COLORS[category], width=4)
    return pil


def save_contact_sheet_svg(dataset, entries, category, output_path):
    if not entries:
        return
    items = []
    for entry in entries:
        tile = draw_case_tile(dataset, entry, category)
        items.append({
            'pil_image': tile,
            'label_lines': [
                f'{category} | sample {entry["sample_index"]}',
                f'img {entry["image_id"]} q{entry["query_index"]} s={entry["category_score"]:.3f}',
                f'obj={entry["obj_prob"]:.3f} unk={entry["unknown_prob"]:.3f} ov={entry.get("gt_overlap", "n/a")}',
            ],
        })
    write_gallery_svg(items, output_path, title=f'{category} representative cases', mode='sampling', cols=3, tile_width=420)


def save_category_csv(entries, output_path):
    if not entries:
        return
    fieldnames = [
        'category', 'sample_index', 'image_id', 'query_index', 'category_score', 'selection_source', 'pseudo_weight', 'gt_overlap',
        'obj_prob', 'known_prob', 'unknown_prob', 'max_known', 'known_score', 'unknown_score', 'argmax_known',
        'box_cxcywh', 'gate_mean', 'gate_peak', 'gate_depth_delta'
    ]
    with open(output_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            row = entry.copy()
            row['box_cxcywh'] = json.dumps(row['box_cxcywh'])
            writer.writerow(row)


def build_parser():
    parser = argparse.ArgumentParser('Representative case mining v3 using GT-aware future-unknown filtering and GPU targets', parents=[get_args_parser()])
    parser.add_argument('--checkpoint', required=True, type=str, help='checkpoint path to load')
    parser.add_argument('--split', default='eval', choices=['train', 'eval'])
    parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--max_samples', default=300, type=int)
    parser.add_argument('--top_k', default=18, type=int)
    parser.add_argument('--pseudo_epoch', default=-1, type=int, help='override pseudo mining epoch; negative uses checkpoint epoch or a safe fallback')
    parser.add_argument('--unknown_gt_min_iou', default=0.1, type=float)
    parser.add_argument('--output_subdir', default='infer/representative_cases', type=str)
    return parser


def main(parsed_args):
    device = torch.device(parsed_args.device)
    checkpoint = torch.load(parsed_args.checkpoint, map_location='cpu')
    model, criterion, _, _ = build_model(parsed_args, mode=parsed_args.model_type)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    criterion.to(device)
    criterion.eval()

    train_dataset, eval_dataset = build_datasets(parsed_args)
    dataset = eval_dataset if parsed_args.split == 'eval' else train_dataset
    total = len(dataset)
    end_index = min(total, parsed_args.start_index + parsed_args.max_samples)

    gate_recorder = ODQEGateRecorder(model)
    top_cases = {'known': [], 'unknown': [], 'odqe_salient': []}
    invalid_cls_logits = list(range(parsed_args.PREV_INTRODUCED_CLS + parsed_args.CUR_INTRODUCED_CLS, parsed_args.num_classes - 1))
    effective_epoch = _effective_pseudo_epoch(parsed_args, checkpoint, criterion)

    for sample_index in range(parsed_args.start_index, end_index):
        image, target = dataset[sample_index]
        samples = nested_tensor_from_tensor_list([image]).to(device)
        cpu_target = _to_cpu_target(target)
        device_target = _to_device_target(target, device)
        gate_recorder.clear()
        with torch.no_grad():
            outputs = model(samples)
        gate_records = ordered_gate_records(gate_recorder.records)
        scores = compute_final_scores(outputs, parsed_args, invalid_cls_logits)

        known_best = _select_best_known_query(scores, device_target, criterion, parsed_args)
        if known_best is not None:
            q, rank_score, overlap = known_best
            update_top_cases(
                top_cases,
                make_case_entry('known', sample_index, cpu_target, scores, q, rank_score, gate_records, selection_source='known_gt_overlap', gt_overlap=overlap),
                parsed_args.top_k,
            )

        unknown_candidates = _select_unknown_candidates(outputs, device_target, scores, criterion, effective_epoch)
        filtered_unknown = _filter_candidates_by_future_unknown_gt(unknown_candidates, scores, device_target, parsed_args, parsed_args.unknown_gt_min_iou)
        if filtered_unknown:
            q, score, weight, iou = filtered_unknown[0]
            update_top_cases(
                top_cases,
                make_case_entry('unknown', sample_index, cpu_target, scores, q, score, gate_records, selection_source='future_unknown_gt_overlap', pseudo_weight=weight, gt_overlap=iou),
                parsed_args.top_k,
            )
            odqe_best = _select_best_odqe_query(filtered_unknown, gate_records, scores)
            if odqe_best is not None:
                q_odqe, signal, _, weight_odqe, iou_odqe = odqe_best
                update_top_cases(
                    top_cases,
                    make_case_entry('odqe_salient', sample_index, cpu_target, scores, q_odqe, signal, gate_records, selection_source='future_unknown_gt_overlap_odqe', pseudo_weight=weight_odqe, gt_overlap=iou_odqe),
                    parsed_args.top_k,
                )

    gate_recorder.restore()

    out_dir = Path(parsed_args.output_dir) / parsed_args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'split': parsed_args.split,
        'start_index': int(parsed_args.start_index),
        'end_index': int(end_index),
        'top_k': int(parsed_args.top_k),
        'pseudo_epoch': int(effective_epoch),
        'unknown_gt_min_iou': float(parsed_args.unknown_gt_min_iou),
        'categories': top_cases,
    }
    with open(out_dir / 'representative_case_manifest.json', 'w', encoding='utf-8') as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    for category, entries in top_cases.items():
        save_category_csv(entries, out_dir / f'{category}_top_cases.csv')
        save_contact_sheet_svg(dataset, entries, category, out_dir / f'{category}_contact_sheet.svg')

    print(f'Saved representative case mining v3 results to: {out_dir}')


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(args)
