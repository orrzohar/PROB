#!/usr/bin/env bash

echo running eval ofnano prob-detr, M-OWODB dataset

set -x

EXP_DIR=exps/MOWODB/PROB
PY_ARGS=${@:1}
WANDB_NAME=PROB_V1
 
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/eval" --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
    --train_set "owod_t1_train" --test_set 'owod_all_task_test' --epochs 191 --lr_drop 35\
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --pretrain "${EXP_DIR}/t1.pth" --eval --wandb_project ""\
    ${PY_ARGS}
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/eval" --dataset TOWOD --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 \
    --train_set "owod_t1_train" --test_set 'owod_all_task_test' --epochs 191 --lr_drop 35\
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --pretrain "${EXP_DIR}/t2.pth" --eval --wandb_project ""\
    ${PY_ARGS}
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/eval" --dataset TOWOD --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
    --train_set "owod_t1_train" --test_set 'owod_all_task_test' --epochs 191 --lr_drop 35\
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --pretrain "${EXP_DIR}/t3.pth" --eval --wandb_project ""\
    ${PY_ARGS}
    
    
PY_ARGS=${@:1}
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/eval" --dataset TOWOD --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
    --train_set "owod_t1_train" --test_set 'owod_all_task_test' --epochs 191 --lr_drop 35\
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --pretrain "${EXP_DIR}/t4.pth" --eval --wandb_project ""\
    ${PY_ARGS}
    
    