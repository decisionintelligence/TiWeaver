#!/bin/bash
# ETTh1/ETTm1/electricity 及其变体 pred_len=96 实验脚本
# 数据集: ETTh1, ETTm1, electricity (原始/freq/masked_20pct)
# 共 3 x 3 = 9 个实验

model_name="dynamic_patching_model_GAN"
CHECKPOINTS="./Run/TiWeaver/ett/checkpoints"
LOGS="./Run/TiWeaver/ett/logs"

# ============================================================
# ETTm1 (7 variables, 69680 rows, 15-min)
# ============================================================

# # ETTm1 原始
# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "ETTm1" \
#     --config_filename ./Model_Config/TiWeaver.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --train-lradj ExponentialDecayLR \
#     --data-root_path dataset/ETT-small \
#     --data-data_path ETTm1.csv \
#     --model_name $model_name \
#     --model-threshold 0.5 \
#     --model-enc_in 7 \
#     --model-seq_len 96 \
#     --model-pred_len 96 \
#     --GPU-gpu 2 \
#     --model-main_dim 7 \
#     --data-target_num 7 \
#     --train-lr 0.0005 \
#     --model-e_layers 3 \
#     --model-min_patch_size 4 \
#     --model-d_model 64 \
#     --model-d_ff 128 \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --model-loss-criterion mse \
#     --data-split_dataset "[0.6, 0.2, 0.2]" \
#     --data-batch_size 32 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS

# # ETTm1 freq
# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "ETTm1" \
#     --config_filename ./Model_Config/TiWeaver.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --train-lradj ExponentialDecayLR \
#     --data-root_path dataset/ETT-small \
#     --data-data_path ETTm1_freq.csv \
#     --model_name $model_name \
#     --model-threshold 0.5 \
#     --model-enc_in 7 \
#     --model-seq_len 96 \
#     --model-pred_len 96 \
#     --GPU-gpu 2 \
#     --model-main_dim 7 \
#     --data-target_num 7 \
#     --train-lr 0.0005 \
#     --model-e_layers 3 \
#     --model-min_patch_size 4 \
#     --model-d_model 64 \
#     --model-d_ff 128 \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --model-loss-criterion mse \
#     --data-split_dataset "[0.6, 0.2, 0.2]" \
#     --data-batch_size 32 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS

# # ETTm1 masked_20pct
# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "ETTm1" \
#     --config_filename ./Model_Config/TiWeaver.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --train-lradj ExponentialDecayLR \
#     --data-root_path dataset/ETT-small \
#     --data-data_path ETTm1_masked_20pct.csv \
#     --model_name $model_name \
#     --model-threshold 0.5 \
#     --model-enc_in 7 \
#     --model-seq_len 96 \
#     --model-pred_len 96 \
#     --GPU-gpu 2 \
#     --model-main_dim 7 \
#     --data-target_num 7 \
#     --train-lr 0.001 \
#     --model-e_layers 3 \
#     --model-min_patch_size 4 \
#     --model-d_model 64 \
#     --model-d_ff 128 \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --model-loss-criterion mse \
#     --data-split_dataset "[0.6, 0.2, 0.2]" \
#     --data-batch_size 32 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS

# ============================================================
# electricity (321 variables, 26304 rows, hourly)
# ============================================================

# electricity 原始
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "electricity" \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/electricity \
    --data-data_path electricity.csv \
    --model_name $model_name \
    --model-threshold 0.5 \
    --model-enc_in 321 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 2 \
    --model-main_dim 321 \
    --data-target_num 321 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 16 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# # electricity freq
# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "electricity" \
#     --config_filename ./Model_Config/TiWeaver.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --train-lradj ExponentialDecayLR \
#     --data-root_path dataset/electricity \
#     --data-data_path electricity_freq.csv \
#     --model_name $model_name \
#     --model-threshold 0.5 \
#     --model-enc_in 321 \
#     --model-seq_len 96 \
#     --model-pred_len 96 \
#     --GPU-gpu 2 \
#     --model-main_dim 321 \
#     --data-target_num 321 \
#     --train-lr 0.0005 \
#     --model-e_layers 3 \
#     --model-min_patch_size 4 \
#     --model-d_model 64 \
#     --model-d_ff 128 \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --model-loss-criterion mse \
#     --data-split_dataset "[0.7, 0.1, 0.2]" \
#     --data-batch_size 4 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS

# # electricity masked_20pct
# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "electricity" \
#     --config_filename ./Model_Config/TiWeaver.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --train-lradj ExponentialDecayLR \
#     --data-root_path dataset/electricity \
#     --data-data_path electricity_masked_20pct.csv \
#     --model_name $model_name \
#     --model-threshold 0.5 \
#     --model-enc_in 321 \
#     --model-seq_len 96 \
#     --model-pred_len 96 \
#     --GPU-gpu 2 \
#     --model-main_dim 321 \
#     --data-target_num 321 \
#     --train-lr 0.001 \
#     --model-e_layers 3 \
#     --model-min_patch_size 4 \
#     --model-d_model 64 \
#     --model-d_ff 128 \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --model-loss-criterion mse \
#     --data-split_dataset "[0.7, 0.1, 0.2]" \
#     --data-batch_size 4 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS
