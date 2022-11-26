#!/usr/bin/env bash

echo running eval of OW-DETR on M-OWODB

set -x

EXP_DIR=exps/SOWODB/OWDETR
PY_ARGS=${@:1}
WANDB_NAME=PROB_V1
 
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/eval" --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
    --train_set "owod_t1_train" --test_set 'owod_all_task_test' --epochs 191 --lr_drop 35\
    --model_type 'owdetr' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --pretrain "${EXP_DIR}/t1.pth" --eval --wandb_project "" --unk_conf_w 4.1 --eval\
    ${PY_ARGS}
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/eval" --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
    --train_set "owod_t1_train" --test_set 'owod_all_task_test' --epochs 191 --lr_drop 35\
    --model_type 'owdetr' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --pretrain "${EXP_DIR}/t1.pth" --eval --wandb_project "" --unk_conf_w 4.2 --eval\
    ${PY_ARGS}
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/eval" --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
    --train_set "owod_t1_train" --test_set 'owod_all_task_test' --epochs 191 --lr_drop 35\
    --model_type 'owdetr' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --pretrain "${EXP_DIR}/t1.pth" --eval --wandb_project "" --unk_conf_w 4.3 --eval\
    ${PY_ARGS}
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/eval" --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
    --train_set "owod_t1_train" --test_set 'owod_all_task_test' --epochs 191 --lr_drop 35\
    --model_type 'owdetr' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --pretrain "${EXP_DIR}/t1.pth" --eval --wandb_project "" --unk_conf_w 4.4 --eval\
    ${PY_ARGS}
    
    