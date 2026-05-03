#!/bin/bash
# Baseline预测模型实验脚本
# 模型: APN_adapter, DLinear_adapter
# 数据集: human_activity, exchange, exchange_freq, exchange_masked_20pct

CHECKPOINTS="./Run/baseline/apn/checkpoints"
LOGS="./Run/baseline/apn/logs"

# ============================================================
# APN_adapter
# ============================================================

for pred_len in 96; do

# # APN - HumanActivity
# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "irregular" \
#     --config_filename ./Model_Config/apn.yaml \
#     --data-data_path HumanActivity \
#     --random_seed 2024 \
#     --train-lradj ExponentialDecayLR \
#     --des 1 \
#     --model_name APN_adapter \
#     --model-seq_len 3000 \
#     --model-pred_len 1000 \
#     --model-e_layers 3 \
#     --model-threshold 0.8 \
#     --model-enc_in 12 \
#     --model-main_dim 12 \
#     --data-target_num 12 \
#     --model-d_model 64 \
#     --model-apn_te_dim 16 \
#     --model-apn_npatch 8 \
#     --train-lr 0.0003 \
#     --data-batch_size 4 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS \
#     --model-loss-criterion mse \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --GPU-gpu 3

# # APN - exchange
# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "exchange_ir" \
#     --config_filename ./Model_Config/apn.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --train-lradj ExponentialDecayLR \
#     --data-root_path dataset/exchange \
#     --data-data_path exchange.csv \
#     --model_name APN_adapter \
#     --model-threshold 0.5 \
#     --model-enc_in 8 \
#     --model-seq_len 96 \
#     --model-pred_len 96 \
#     --GPU-gpu 3 \
#     --model-main_dim 8 \
#     --data-target_num 8 \
#     --model-d_model 64 \
#     --model-apn_te_dim 16 \
#     --model-apn_npatch 8 \
#     --train-lr 0.0005 \
#     --model-e_layers 3 \
#     --model-min_patch_size 4 \
#     --model-d_ff 128 \
#     --model-loss-criterion mse \
#     --data-split_dataset "[0.7, 0.1, 0.2]" \
#     --data-batch_size 128 \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS

# # APN - exchange_freq
# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "exchange_ir" \
#     --config_filename ./Model_Config/apn.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --train-lradj ExponentialDecayLR \
#     --data-root_path dataset/exchange \
#     --data-data_path exchange_freq.csv \
#     --model_name APN_adapter \
#     --model-threshold 0.5 \
#     --model-enc_in 8 \
#     --model-seq_len 96 \
#     --model-pred_len $pred_len \
#     --GPU-gpu 3 \
#     --model-main_dim 8 \
#     --data-target_num 8 \
#     --model-d_model 64 \
#     --model-apn_te_dim 16 \
#     --model-apn_npatch 8 \
#     --train-lr 0.0005 \
#     --model-e_layers 3 \
#     --model-min_patch_size 4 \
#     --model-d_ff 128 \
#     --model-loss-criterion mse \
#     --data-split_dataset "[0.7, 0.1, 0.2]" \
#     --data-batch_size 16 \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS

# # APN - exchange_masked_20pct
# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "exchange_ir" \
#     --config_filename ./Model_Config/apn.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --train-lradj ExponentialDecayLR \
#     --data-root_path dataset/exchange \
#     --data-data_path exchange_masked_20pct.csv \
#     --model_name APN_adapter \
#     --model-threshold 0.5 \
#     --model-enc_in 8 \
#     --model-seq_len 96 \
#     --model-pred_len $pred_len \
#     --GPU-gpu 3 \
#     --model-main_dim 8 \
#     --data-target_num 8 \
#     --model-d_model 64 \
#     --model-apn_te_dim 16 \
#     --model-apn_npatch 8 \
#     --train-lr 0.001 \
#     --model-e_layers 3 \
#     --model-min_patch_size 4 \
#     --model-d_ff 128 \
#     --model-loss-criterion mse \
#     --data-split_dataset "[0.7, 0.1, 0.2]" \
#     --data-batch_size 128 \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS

# ============================================================
# DLinear_adapter
# ============================================================

CHECKPOINTS="./Run/baseline/dl/checkpoints"
LOGS="./Run/baseline/dl/logs"

# # DLinear - HumanActivity
# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "irregular" \
#     --config_filename ./Model_Config/dlinear.yaml \
#     --data-data_path HumanActivity \
#     --random_seed 2024 \
#     --train-lradj ExponentialDecayLR \
#     --des 1 \
#     --model_name DLinear_adapter \
#     --model-seq_len 3000 \
#     --model-pred_len 1000 \
#     --model-e_layers 3 \
#     --model-threshold 0.8 \
#     --model-enc_in 12 \
#     --model-main_dim 12 \
#     --data-target_num 12 \
#     --model-d_model 64 \
#     --train-lr 0.0003 \
#     --data-batch_size 4 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS \
#     --model-loss-criterion mse \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --GPU-gpu 3

# # DLinear - exchange
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/dlinear.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange.csv \
    --model_name DLinear_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 3 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-d_ff 128 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 128 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# DLinear - exchange_freq
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/dlinear.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_freq.csv \
    --model_name DLinear_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len $pred_len \
    --GPU-gpu 3 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-d_ff 128 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 16 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# DLinear - exchange_masked_20pct
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/dlinear.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_masked_20pct.csv \
    --model_name DLinear_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len $pred_len \
    --GPU-gpu 3 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --train-lr 0.001 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-d_ff 128 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 128 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS
done
