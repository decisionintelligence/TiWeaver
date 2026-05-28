#!/bin/bash

# ============================================================
# exchange
# ============================================================
python ./Run/train.py \
    --data-data_name "exchange" \
    --model_name TiWeaver \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --random_seed 2021 \
    --data-root_path dataset/exchange \
    --data-data_path exchange.csv \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --model-main_dim 8 \
    --model-d_model 64 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-d_ff 128 \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 128 \
    --GPU-gpu 0

# ============================================================
# exchange_freq
# ============================================================
python ./Run/train.py \
    --data-data_name "exchange" \
    --model_name TiWeaver \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --random_seed 2021 \
    --data-root_path dataset/exchange \
    --data-data_path exchange_freq.csv \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --model-main_dim 8 \
    --model-d_model 64 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-d_ff 128 \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 16 \
    --GPU-gpu 0

# ============================================================
# exchange_masked_20pct
# ============================================================
python ./Run/train.py \
    --data-data_name "exchange" \
    --model_name TiWeaver \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --random_seed 2021 \
    --data-root_path dataset/exchange \
    --data-data_path exchange_masked_20pct.csv \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --model-d_model 64 \
    --train-lr 0.001 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-d_ff 128 \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 128 \
    --GPU-gpu 0