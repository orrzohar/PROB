#!/usr/bin/env bash

echo running training of prob-detr, S-OWODB dataset

set -x

EXP_DIR=exps/SOWODB/PROB_V1
PY_ARGS=${@:1}
WANDB_NAME=PROB_OWDETR_V1

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t1" --dataset OWDETR --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19\
    --train_set 'owdetr_t1_train' --test_set 'owdetr_test' --epochs 41 --lr_drop 31\
    --model_type 'prob' --obj_loss_coef 1e-3 --obj_temp 1.3\
    --wandb_name "${WANDB_NAME}_t1" --exemplar_replay_selection --exemplar_replay_max_length 850\
    --exemplar_replay_dir ${WANDB_NAME} --exemplar_replay_cur_file "learned_owdetr_t1_ft.txt"\
    ${PY_ARGS}
    

PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t2" --dataset OWDETR --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21\
    --train_set 'owdetr_t2_train' --test_set 'owdetr_test' --epochs 51\
    --model_type 'prob' --obj_loss_coef 1e-3 --freeze_prob_model --obj_temp 1.3\
    --wandb_name "${WANDB_NAME}_t2"\
    --exemplar_replay_selection --exemplar_replay_max_length 1679 --exemplar_replay_dir ${WANDB_NAME}\
    --exemplar_replay_prev_file "learned_owdetr_t1_ft.txt" --exemplar_replay_cur_file "learned_owdetr_t2_ft.txt"\
    --pretrain "${EXP_DIR}/t1.pth" --lr 2e-5\
    ${PY_ARGS}
    

PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t2_ft" --dataset OWDETR --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 \
    --train_set "${WANDB_NAME}/learned_owdetr_t2_ft" --test_set 'owdetr_test' --epochs 121 --lr_drop 50\
    --model_type 'prob' --obj_loss_coef 1e-3 --obj_temp 1.3\
    --wandb_name "${WANDB_NAME}_t2_ft"\
    --pretrain "${EXP_DIR}/t2/checkpoint0050.pth"\
    ${PY_ARGS}
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t3" --dataset OWDETR --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20\
    --train_set 'owdetr_t3_train' --test_set 'owdetr_test' --epochs 131\
    --model_type 'prob' --obj_loss_coef 1e-3 --freeze_prob_model --obj_temp 1.3\
    --wandb_name "${WANDB_NAME}_t3"\
    --exemplar_replay_selection --exemplar_replay_max_length 2345 --exemplar_replay_dir ${WANDB_NAME}\
    --exemplar_replay_prev_file "learned_owdetr_t2_ft.txt" --exemplar_replay_cur_file "learned_owdetr_t3_ft.txt"\
    --pretrain "${EXP_DIR}/t2_ft/checkpoint0120.pth" --lr 2e-5 \
    ${PY_ARGS}
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t3_ft" --dataset OWDETR --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
    --train_set "${WANDB_NAME}/learned_owdetr_t3_ft" --test_set 'owdetr_test' --epochs 201 --lr_drop 50\
    --model_type 'prob' --obj_loss_coef 1e-3 --obj_temp 1.3\
    --wandb_name "${WANDB_NAME}_t3_ft"\
    --pretrain "${EXP_DIR}/t3/checkpoint0130.pth"\
    ${PY_ARGS}
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t4" --dataset OWDETR --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20\
    --train_set 'owdetr_t4_train' --test_set 'owdetr_test' --epochs 211 \
    --model_type 'prob' --obj_loss_coef 1e-3 --freeze_prob_model --obj_temp 1.3\
    --wandb_name "${WANDB_NAME}_t4"\
    --exemplar_replay_selection --exemplar_replay_max_length 2664 --exemplar_replay_dir ${WANDB_NAME}\
    --exemplar_replay_prev_file "learned_owdetr_t3_ft.txt" --exemplar_replay_cur_file "learned_owdetr_t4_ft.txt"\
    --num_inst_per_class 40\
    --pretrain "${EXP_DIR}/t3_ft/checkpoint0200.pth" --lr 2e-5\
    ${PY_ARGS}
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t4_ft" --dataset OWDETR --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20\
    --train_set "${WANDB_NAME}/learned_owdetr_t4_ft" --test_set 'owdetr_test' --epochs 301 --lr_drop 50\
    --model_type 'prob' --obj_loss_coef 1e-3 --obj_temp 1.3\
    --wandb_name "${WANDB_NAME}_t4_ft"\
    --pretrain "${EXP_DIR}/t4/checkpoint0210.pth" \
    ${PY_ARGS}