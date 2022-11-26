#!/bin/bash

cd PROB

conda create --name prob python=3.10.4

conda activate prob

pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install wandb

pip install einops

pip install pycocotools

conda install -c anaconda scikit-image

pip install joblib

pip install tqdm

pip install notebook

pip install ipdb

pip install pandas

pip install seaborn

pip install numpy==1.21.5

cd models

wget https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth

cd ops

sh ./make.sh

python test.py