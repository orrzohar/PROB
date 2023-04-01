# partly taken from  https://github.com/pytorch/vision/blob/master/torchvision/datasets/voc.py
import functools
import torch

import os
import tarfile
import collections
import logging
import copy
from torchvision.datasets import VisionDataset
import itertools

import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg

#OWOD splits
VOC_CLASS_NAMES_COCOFIED = [
    "airplane",  "dining table", "motorcycle",
    "potted plant", "couch", "tv"
]

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant",  "sofa", "tvmonitor"
]
UNK_CLASS = ["unknown"]

VOC_COCO_CLASS_NAMES={}


T1_CLASS_NAMES = [
    "aeroplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorbike","sheep","train",
    "elephant","bear","zebra","giraffe","truck","person"
]

T2_CLASS_NAMES = [
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","diningtable",
    "pottedplant","backpack","umbrella","handbag",
    "tie","suitcase","microwave","oven","toaster","sink",
    "refrigerator","bed","toilet","sofa"
]

T3_CLASS_NAMES = [
    "frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake"
]

T4_CLASS_NAMES = [
    "laptop","mouse","remote","keyboard","cell phone","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush",
    "wine glass","cup","fork","knife","spoon","bowl","tvmonitor","bottle"
]

VOC_COCO_CLASS_NAMES["OWDETR"] = tuple(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))


VOC_CLASS_NAMES = [
"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
"pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

VOC_CLASS_NAMES_COCOFIED = [
    "airplane",  "dining table", "motorcycle",
    "potted plant", "couch", "tv"
]

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant",  "sofa", "tvmonitor"
]

T2_CLASS_NAMES = [
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator"
]

T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]

T4_CLASS_NAMES = [
    "bed", "toilet", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl"
]
VOC_COCO_CLASS_NAMES["TOWOD"] = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))
VOC_COCO_CLASS_NAMES["VOC2007"] = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))


print(VOC_COCO_CLASS_NAMES)

class OWDetection(VisionDataset):
    """`OWOD in Pascal VOC format <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            (default: alphabetic indexing of VOC's 20 classes).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 args,
                 root,
                 image_set='train',
                 transforms=None,
                 filter_pct=-1,
                 dataset='OWDETR'):
        super(OWDetection, self).__init__(transforms)
        self.images = []
        self.annotations = []
        self.imgids = []
        self.imgid2annotations = {}
        self.image_set = []
        self.transforms=transforms
        self.CLASS_NAMES = VOC_COCO_CLASS_NAMES[dataset]
        self.MAX_NUM_OBJECTS = 64
        self.args = args
        self.dataset=dataset

        self.root=str(root)
        annotation_dir = os.path.join(self.root, 'Annotations')
        image_dir = os.path.join(self.root, 'JPEGImages')

        file_names = self.extract_fns(image_set, self.root)
        if image_set == 'voc2007_trainval':
            print('PASCAL-VOC2007 dataset used; clearing images with missing object classes')
            prev_intro_cls = self.args.PREV_INTRODUCED_CLS
            curr_intro_cls = self.args.CUR_INTRODUCED_CLS
            valid_classes = range(prev_intro_cls, prev_intro_cls + curr_intro_cls)
            current_file_names=[]
            for file in file_names:
                annot = os.path.join(annotation_dir, file + ".xml")
                tree = ET.parse(annot)
                target = self.parse_voc_xml(tree.getroot())
                instances = []
                for obj in target['annotation']['object']:
                    cls = obj["name"]
                    if cls in VOC_CLASS_NAMES_COCOFIED:
                        cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
                    
                    if self.CLASS_NAMES.index(cls) in valid_classes:
                        instance = dict(
                            category_id=self.CLASS_NAMES.index(cls),
                        )
                        instances.append(instance)
                if len(instances)>0:
                    current_file_names.append(file)

            self.image_set.extend(current_file_names)
            self.images.extend([os.path.join(image_dir, x + ".jpg") for x in current_file_names])
            self.annotations.extend([os.path.join(annotation_dir, x + ".xml") for x in current_file_names])
            self.imgids.extend(self.convert_image_id(x, to_integer=True) for x in current_file_names)
        else: 
            self.image_set.extend(file_names)
            self.images.extend([os.path.join(image_dir, x + ".jpg") for x in file_names])
            self.annotations.extend([os.path.join(annotation_dir, x + ".xml") for x in file_names])
            self.imgids.extend(self.convert_image_id(x, to_integer=True) for x in file_names)
            
        self.imgid2annotations.update(dict(zip(self.imgids, self.annotations)))

        if filter_pct > 0:
            num_keep = float(len(self.imgids)) * filter_pct
            keep = np.random.choice(np.arange(len(self.imgids)), size=round(num_keep), replace=False).tolist()
            flt = lambda l: [l[i] for i in keep]
            self.image_set, self.images, self.annotations, self.imgids = map(flt, [self.image_set, self.images,
                                                                                   self.annotations, self.imgids])
        assert (len(self.images) == len(self.annotations) == len(self.imgids))

    @staticmethod
    def convert_image_id(img_id, to_integer=False, to_string=False, prefix='2021'):
        if to_integer:
            return int(prefix + img_id.replace('_', ''))
        if to_string:
            x = str(img_id)
            assert x.startswith(prefix)
            x = x[len(prefix):]
            if len(x) == 12 or len(x) == 6:
                return x
            return x[:4] + '_' + x[4:]

    @functools.lru_cache(maxsize=None)
    def load_instances(self, img_id):
        tree = ET.parse(self.imgid2annotations[img_id])
        target = self.parse_voc_xml(tree.getroot())
        image_id = target['annotation']['filename']
        instances = []
        for obj in target['annotation']['object']:
            cls = obj["name"]

            if cls in VOC_CLASS_NAMES_COCOFIED:
                cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
            bbox = obj["bndbox"]
            bbox = [float(bbox[x]) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instance = dict(
                category_id=self.CLASS_NAMES.index(cls),
                bbox=bbox,
                area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                image_id=img_id
            )
            instances.append(instance)
        return target, instances

    def extract_fns(self, image_set, voc_root):
        splits_dir = os.path.join(voc_root, 'ImageSets')
        splits_dir = os.path.join(splits_dir, self.dataset)
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        return file_names

    ### OWOD
    def remove_prev_class_and_unk_instances(self, target):
        # For training data. Removing earlier seen class objects and the unknown objects..
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        valid_classes = range(prev_intro_cls, prev_intro_cls + curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] not in valid_classes:
                entry.remove(annotation)
        return entry

    def remove_unknown_instances(self, target):
        # For finetune data. Removing the unknown objects...
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        valid_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] not in valid_classes:
                entry.remove(annotation)
        return entry

    def label_known_class_and_unknown(self, target):
        # For test and validation data.
        # Label known instances the corresponding label and unknown instances as unknown.
        prev_intro_cls = self.args.PREV_INTRODUCED_CLS
        curr_intro_cls = self.args.CUR_INTRODUCED_CLS
        total_num_class = self.args.num_classes #81
        known_classes = range(0, prev_intro_cls+curr_intro_cls)
        entry = copy.copy(target)
        for annotation in  copy.copy(entry):
        # for annotation in entry:
            if annotation["category_id"] not in known_classes:
                annotation["category_id"] = total_num_class - 1
        return entry

    def __getitem__(self, index):
        """
        Args:
            index (int): Indexin

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """

        image_set = self.transforms[0]
        img = Image.open(self.images[index]).convert('RGB')
        target, instances = self.load_instances(self.imgids[index])
        if 'train' in image_set:
            instances = self.remove_prev_class_and_unk_instances(instances)
        elif 'test' in image_set:
            instances = self.label_known_class_and_unknown(instances)
        elif 'ft' in image_set:
            instances = self.remove_unknown_instances(instances)

        w, h = map(target['annotation']['size'].get, ['width', 'height'])
        target = dict(
            image_id=torch.tensor([self.imgids[index]], dtype=torch.int64),
            org_image_id=torch.Tensor([ord(c) for c in self.annotations[index].split('/')[-1].split('.xml')[0]]),
            labels=torch.tensor([i['category_id'] for i in instances], dtype=torch.int64),
            area=torch.tensor([i['area'] for i in instances], dtype=torch.float32),
            boxes=torch.as_tensor([i['bbox'] for i in instances], dtype=torch.float32),
            orig_size=torch.as_tensor([int(h), int(w)]),
            size=torch.as_tensor([int(h), int(w)]),
            iscrowd=torch.zeros(len(instances), dtype=torch.uint8)
        )
        #import ipdb;ipdb.set_trace()

        if self.transforms[-1] is not None:
            img, target = self.transforms[-1](img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)