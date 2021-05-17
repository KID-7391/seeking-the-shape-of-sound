#!/bin/sh
export CUDA_VISIBLE_DEVICES=5
python3 train.py config/train_reweight.yaml
python3 extract_id_weight.py config/train_reweight.yaml
python3 train.py config/train_main.yaml
python3 eval.py config/train_main.yaml
