#!/bin/bash
# TimeMosaic Baseline 预测实验脚本
# 模型: TimeMosaic_adapter
# 数据集:
#   - human_activity (12变量, seq=3000, pred=1000)
#   - exchange / exchange_freq / exchange_masked_20pct (8变量, seq=96, pred=96/192/336/720)
#   - ETTh1 / ETTm1 及其 freq/masked_20pct 变体 (7变量, seq=96, pred=96)

CHECKPOINTS="./Run/baseline/tim/checkpoints"
LOGS="./Run/baseline/tim/logs"

# ============================================================
# TimeMosaic - HumanActivity
# ============================================================
# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "irregular" \
#     --config_filename ./Model_Config/tim.yaml \
#     --data-data_path HumanActivity \
#     --random_seed 2024 \
#     --train-lradj ExponentialDecayLR \
#     --des 1 \
#     --model_name TimeMosaic_adapter \
#     --model-seq_len 3000 \
#     --model-pred_len 1000 \
#     --model-e_layers 3 \
#     --model-threshold 0.8 \
#     --model-enc_in 12 \
#     --model-main_dim 12 \
#     --data-target_num 12 \
#     --model-d_model 64 \
#     --model-d_ff 128 \
#     --model-n_heads 2 \
#     --model-patch_len_list "[7,14,49]" \
#     --model-num_latent_token 4 \
#     --train-lr 0.0003 \
#     --data-batch_size 128 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS \
#     --model-loss-criterion mse \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --GPU-gpu 2

# ============================================================
# TimeMosaic - exchange 多步长
# ============================================================

# exchange pred_len=96
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/tim.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange.csv \
    --model_name TimeMosaic_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-n_heads 2 \
    --model-patch_len_list "[8,16,32]" \
    --model-num_latent_token 4 \
    --model-pre96 32 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 128 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# exchange pred_len=192
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/tim.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange.csv \
    --model_name TimeMosaic_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 192 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-n_heads 2 \
    --model-patch_len_list "[8,16,32]" \
    --model-num_latent_token 4 \
    --model-pre192 64 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 128 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# exchange pred_len=336
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/tim.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange.csv \
    --model_name TimeMosaic_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 336 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-n_heads 2 \
    --model-patch_len_list "[8,16,32]" \
    --model-num_latent_token 4 \
    --model-pre336 168 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 128 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# exchange pred_len=720
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/tim.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange.csv \
    --model_name TimeMosaic_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 720 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-n_heads 2 \
    --model-patch_len_list "[8,16,32]" \
    --model-num_latent_token 4 \
    --model-pre720 240 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 128 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# ============================================================
# TimeMosaic - exchange_freq 多步长
# ============================================================

# exchange_freq pred_len=96
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/tim.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_freq.csv \
    --model_name TimeMosaic_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-n_heads 2 \
    --model-patch_len_list "[8,16,32]" \
    --model-num_latent_token 4 \
    --model-pre96 32 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 16 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# exchange_freq pred_len=192
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/tim.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_freq.csv \
    --model_name TimeMosaic_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 192 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-n_heads 2 \
    --model-patch_len_list "[8,16,32]" \
    --model-num_latent_token 4 \
    --model-pre192 64 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 16 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# exchange_freq pred_len=336
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/tim.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_freq.csv \
    --model_name TimeMosaic_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 336 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-n_heads 2 \
    --model-patch_len_list "[8,16,32]" \
    --model-num_latent_token 4 \
    --model-pre336 168 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 16 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# exchange_freq pred_len=720
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/tim.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_freq.csv \
    --model_name TimeMosaic_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 720 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-n_heads 2 \
    --model-patch_len_list "[8,16,32]" \
    --model-num_latent_token 4 \
    --model-pre720 240 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 16 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# ============================================================
# TimeMosaic - exchange_masked_20pct 多步长
# ============================================================

# exchange_masked_20pct pred_len=96
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/tim.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_masked_20pct.csv \
    --model_name TimeMosaic_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-n_heads 2 \
    --model-patch_len_list "[8,16,32]" \
    --model-num_latent_token 4 \
    --model-pre96 32 \
    --train-lr 0.001 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 128 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# exchange_masked_20pct pred_len=192
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/tim.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_masked_20pct.csv \
    --model_name TimeMosaic_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 192 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-n_heads 2 \
    --model-patch_len_list "[8,16,32]" \
    --model-num_latent_token 4 \
    --model-pre192 64 \
    --train-lr 0.001 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 128 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# exchange_masked_20pct pred_len=336
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/tim.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_masked_20pct.csv \
    --model_name TimeMosaic_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 336 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-n_heads 2 \
    --model-patch_len_list "[8,16,32]" \
    --model-num_latent_token 4 \
    --model-pre336 168 \
    --train-lr 0.001 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 128 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# exchange_masked_20pct pred_len=720
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/tim.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_masked_20pct.csv \
    --model_name TimeMosaic_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 720 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-n_heads 2 \
    --model-patch_len_list "[8,16,32]" \
    --model-num_latent_token 4 \
    --model-pre720 240 \
    --train-lr 0.001 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 128 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# ============================================================
# TimeMosaic - ETTh1 及变体 (pred_len=96)
# ============================================================

# ETTh1 原始
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "ETTh1s" \
    --config_filename ./Model_Config/tim.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/ETT-small \
    --data-data_path ETTh1.csv \
    --model_name TimeMosaic_adapter \
    --model-threshold 0.5 \
    --model-enc_in 7 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 2 \
    --model-main_dim 7 \
    --data-target_num 7 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-n_heads 2 \
    --model-patch_len_list "[8,16,32]" \
    --model-num_latent_token 4 \
    --model-pre96 32 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.6, 0.2, 0.2]" \
    --data-batch_size 32 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# ETTm1 原始
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "ETTm1" \
    --config_filename ./Model_Config/tim.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/ETT-small \
    --data-data_path ETTm1.csv \
    --model_name TimeMosaic_adapter \
    --model-threshold 0.5 \
    --model-enc_in 7 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 2 \
    --model-main_dim 7 \
    --data-target_num 7 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-n_heads 2 \
    --model-patch_len_list "[8,16,32]" \
    --model-num_latent_token 4 \
    --model-pre96 32 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.6, 0.2, 0.2]" \
    --data-batch_size 32 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS


# electricity 原始
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "electricity" \
    --config_filename ./Model_Config/tim.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/electricity \
    --data-data_path electricity.csv \
    --model_name TimeMosaic_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-n_heads 2 \
    --model-patch_len_list "[8,16,32]" \
    --model-num_latent_token 4 \
    --model-pre96 32 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.6, 0.2, 0.2]" \
    --data-batch_size 32 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS


CHECKPOINTS="./Run/baseline/apn/checkpoints"
LOGS="./Run/baseline/apn/logs"


# APN - exchange 192
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/apn.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange.csv \
    --model_name APN_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 192 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --model-apn_te_dim 16 \
    --model-apn_npatch 8 \
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


# APN - exchange 336
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/apn.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange.csv \
    --model_name APN_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 336 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --model-apn_te_dim 16 \
    --model-apn_npatch 8 \
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

# APN - exchange 720
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/apn.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange.csv \
    --model_name APN_adapter \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 720 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-d_model 64 \
    --model-apn_te_dim 16 \
    --model-apn_npatch 8 \
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

CHECKPOINTS="./Run/baseline/dl/checkpoints"
LOGS="./Run/baseline/dl/logs"

# DLinear - exchange 192
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
    --model-pred_len 192 \
    --GPU-gpu 2 \
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

# DLinear - exchange 336
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
    --model-pred_len 336 \
    --GPU-gpu 2 \
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

# DLinear - exchange 720
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
    --model-pred_len 720 \
    --GPU-gpu 2 \
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