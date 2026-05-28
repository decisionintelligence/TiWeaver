#!/bin/bash

python ./Run/train.py \
    --model_name TiWeaver \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --random_seed 2024 \
    --data-data_name event \
    --data-data_path HumanActivity \
    --model-seq_len 3000 \
    --model-pred_len 1000 \
    --model-e_layers 3 \
    --model-threshold 0.8 \
    --model-enc_in 12 \
    --model-main_dim 12 \
    --model-d_model 64 \
    --train-lr 0.0003 \
    --data-batch_size 4 \
    --GPU-gpu 0
