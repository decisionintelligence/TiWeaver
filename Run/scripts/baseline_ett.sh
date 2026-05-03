#!/bin/bash
# Baseline预测模型实验脚本
# 模型: APN_adapter, DLinear_adapter
# 数据集: human_activity, exchange, exchange_freq, exchange_masked_20pct

CHECKPOINTS="./Run/baseline/ett/checkpoints"
LOGS="./Run/baseline/ett/logs"

# ============================================================
# APN_adapter
# ============================================================

# for dataset_name in "ETTh1_freq.csv" "ETTh1_masked_20pct.csv"; do

# # APN - exchange
# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "ETTh1" \
#     --config_filename ./Model_Config/apn.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --train-lradj ExponentialDecayLR \
#     --data-root_path dataset/ETT-small \
#     --data-data_path $dataset_name \
#     --model_name APN_adapter \
#     --model-threshold 0.5 \
#     --model-enc_in 7 \
#     --model-seq_len 96 \
#     --model-pred_len 96 \
#     --GPU-gpu 1 \
#     --model-main_dim 7 \
#     --data-target_num 7 \
#     --model-d_model 64 \
#     --model-apn_te_dim 16 \
#     --model-apn_npatch 8 \
#     --train-lr 0.0005 \
#     --model-e_layers 3 \
#     --model-min_patch_size 4 \
#     --model-d_ff 128 \
#     --model-loss-criterion mse \
#     --data-split_dataset "[0.6, 0.2, 0.2]" \
#     --data-batch_size 128 \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS

# # ============================================================
# # DLinear_adapter
# # ============================================================

# # DLinear - exchange
# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "ETTh1" \
#     --config_filename ./Model_Config/dlinear.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --train-lradj ExponentialDecayLR \
#     --data-root_path dataset/ETT-small \
#     --data-data_path $dataset_name \
#     --model_name DLinear_adapter \
#     --model-threshold 0.5 \
#     --model-enc_in 7 \
#     --model-seq_len 96 \
#     --model-pred_len 96 \
#     --GPU-gpu 1 \
#     --model-main_dim 7 \
#     --data-target_num 7 \
#     --model-d_model 64 \
#     --train-lr 0.0005 \
#     --model-e_layers 3 \
#     --model-min_patch_size 4 \
#     --model-d_ff 128 \
#     --model-loss-criterion mse \
#     --data-split_dataset "[0.6, 0.2, 0.2]" \
#     --data-batch_size 128 \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS


# # ============================================================
# # tim_adapter
# # ============================================================

# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "ETTh1" \
#     --config_filename ./Model_Config/tim.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --train-lradj ExponentialDecayLR \
#     --data-root_path dataset/ETT-small \
#     --data-data_path $dataset_name \
#     --model_name TimeMosaic_adapter \
#     --model-threshold 0.5 \
#     --model-enc_in 7 \
#     --model-seq_len 96 \
#     --model-pred_len 192 \
#     --GPU-gpu 1 \
#     --model-main_dim 7 \
#     --data-target_num 7 \
#     --model-d_model 64 \
#     --model-d_ff 128 \
#     --model-n_heads 2 \
#     --model-patch_len_list "[8,16,32]" \
#     --model-num_latent_token 4 \
#     --model-pre96 32 \
#     --train-lr 0.0005 \
#     --model-e_layers 3 \
#     --model-min_patch_size 4 \
#     --model-loss-criterion mse \
#     --data-split_dataset "[0.6, 0.2, 0.2]" \
#     --data-batch_size 128 \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS

# done

# for data_path in "ETTm1_freq.csv"; do

# python ./Run/train.py \
#     --config_filename ./Model_Config/BAOWU/PatchTST.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --data-data_path $data_path \
#     --data-root_path "dataset/ETT-small" \
#     --data-data_name "ETTm1" \
#     --exp_name irregular \
#     --model-main_dim 7 \
#     --data-split_dataset "[0.6, 0.2, 0.2]" \
#     --model-enc_in 7 \
#     --data-target_num 7 \
#     --model-seq_len 96 \
#     --model-pred_len 96 \
#     --GPU-gpu 1 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS

# python ./Run/train.py \
#     --config_filename ./Model_Config/BAOWU/PDF.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --data-data_path $data_path \
#     --data-root_path "dataset/ETT-small" \
#     --data-data_name "ETTm1" \
#     --exp_name irregular \
#     --model-main_dim 7 \
#     --model-d_ff 48 \
#     --model-d_model 48 \
#     --model-dropout 0.5 \
#     --model-e_layers 3 \
#     --model-fc_dropout 0.25 \
#     --model-kernel_list "[5, 7, 11, 15]" \
#     --model-n_heads 16 \
#     --model-patch_len "[3, 6, 16, 32, 48]" \
#     --train-pct_start 0.2 \
#     --train-lradj "TST" \
#     --model-period "[48, 90, 102, 360, 720]" \
#     --model-stride "[3, 6, 16, 32, 48]" \
#     --data-split_dataset "[0.6, 0.2, 0.2]" \
#     --data-target_num 7 \
#     --model-seq_len 96 \
#     --model-pred_len 96 \
#     --GPU-gpu 1 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS

# # APN - exchange
# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "ETTh1" \
#     --config_filename ./Model_Config/apn.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --train-lradj ExponentialDecayLR \
#     --data-root_path dataset/ETT-small \
#     --data-data_path $dataset_name \
#     --model_name APN_adapter \
#     --model-threshold 0.5 \
#     --model-enc_in 7 \
#     --model-seq_len 96 \
#     --model-pred_len 96 \
#     --GPU-gpu 1 \
#     --model-main_dim 7 \
#     --data-target_num 7 \
#     --model-d_model 64 \
#     --model-apn_te_dim 16 \
#     --model-apn_npatch 8 \
#     --train-lr 0.0005 \
#     --model-e_layers 3 \
#     --model-min_patch_size 4 \
#     --model-d_ff 128 \
#     --model-loss-criterion mse \
#     --data-split_dataset "[0.6, 0.2, 0.2]" \
#     --data-batch_size 128 \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS

# # ============================================================
# # DLinear_adapter
# # ============================================================

# # DLinear - exchange
# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "ETTh1" \
#     --config_filename ./Model_Config/dlinear.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --train-lradj ExponentialDecayLR \
#     --data-root_path dataset/ETT-small \
#     --data-data_path $dataset_name \
#     --model_name DLinear_adapter \
#     --model-threshold 0.5 \
#     --model-enc_in 7 \
#     --model-seq_len 96 \
#     --model-pred_len 96 \
#     --GPU-gpu 1 \
#     --model-main_dim 7 \
#     --data-target_num 7 \
#     --model-d_model 64 \
#     --train-lr 0.0005 \
#     --model-e_layers 3 \
#     --model-min_patch_size 4 \
#     --model-d_ff 128 \
#     --model-loss-criterion mse \
#     --data-split_dataset "[0.6, 0.2, 0.2]" \
#     --data-batch_size 128 \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS


# # ============================================================
# # tim_adapter
# # ============================================================

# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "ETTh1" \
#     --config_filename ./Model_Config/tim.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --train-lradj ExponentialDecayLR \
#     --data-root_path dataset/ETT-small \
#     --data-data_path $dataset_name \
#     --model_name TimeMosaic_adapter \
#     --model-threshold 0.5 \
#     --model-enc_in 7 \
#     --model-seq_len 96 \
#     --model-pred_len 192 \
#     --GPU-gpu 1 \
#     --model-main_dim 7 \
#     --data-target_num 7 \
#     --model-d_model 64 \
#     --model-d_ff 128 \
#     --model-n_heads 2 \
#     --model-patch_len_list "[8,16,32]" \
#     --model-num_latent_token 4 \
#     --model-pre96 32 \
#     --train-lr 0.0005 \
#     --model-e_layers 3 \
#     --model-min_patch_size 4 \
#     --model-loss-criterion mse \
#     --data-split_dataset "[0.6, 0.2, 0.2]" \
#     --data-batch_size 128 \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS

# done


for data_path in "electricity.csv"
do

# python ./Run/train.py \
#     --config_filename ./Model_Config/BAOWU/PatchTST.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --data-data_path $data_path \
#     --data-root_path "dataset/electricity" \
#     --data-data_name "electricity" \
#     --exp_name irregular \
#     --model-main_dim 321 \
#     --data-split_dataset "[0.7, 0.1, 0.2]" \
#     --model-enc_in 321 \
#     --data-target_num 321 \
#     --model-seq_len 96 \
#     --model-pred_len 96 \
#     --data-batch_size 32 \
#     --GPU-gpu 1 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS

python ./Run/eval.py \
    --config_filename ./Model_Config/BAOWU/PDF.yaml \
    --des 1 \
    --random_seed 2021 \
    --data-data_path $data_path \
    --data-root_path "dataset/electricity" \
    --data-data_name "electricity" \
    --exp_name irregular \
    --model-main_dim 321 \
    --model-num_nodes 321 \
    --model-c_in 321 \
    --model-enc_in 321 \
    --model-d_ff 32 \
    --model-d_model 128 \
    --model-dropout 0.45 \
    --model-e_layers 3 \
    --model-fc_dropout 0.15 \
    --model-kernel_list "[3, 5, 7, 7]" \
    --model-n_heads 32 \
    --model-patch_len "[1, 2, 3, 16, 48]" \
    --train-pct_start 0.2 \
    --model-period "[8, 12, 24, 180, 720]" \
    --model-stride "[1, 2, 3, 16, 48]" \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-target_num 321 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --data-batch_size 32 \
    --GPU-gpu 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "electricity" \
#     --config_filename ./Model_Config/tim.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --train-lradj ExponentialDecayLR \
#     --data-root_path dataset/electricity \
#     --data-data_path electricity.csv \
#     --model_name TimeMosaic_adapter \
#     --model-threshold 0.5 \
#     --model-enc_in 8 \
#     --model-seq_len 96 \
#     --model-pred_len 96 \
#     --GPU-gpu 1 \
#     --model-main_dim 8 \
#     --data-target_num 8 \
#     --model-d_model 64 \
#     --model-d_ff 128 \
#     --model-n_heads 2 \
#     --model-patch_len_list "[8,16,32]" \
#     --model-num_latent_token 4 \
#     --model-pre96 32 \
#     --train-lr 0.0005 \
#     --model-e_layers 3 \
#     --model-min_patch_size 4 \
#     --model-loss-criterion mse \
#     --data-split_dataset "[0.7, 0.1, 0.2]" \
#     --data-batch_size 32 \
#     --model-time 1 \
#     --model-catt 1 \
#     --model-gap 1 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS


# # APN - exchange
# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "ETTh1" \
#     --config_filename ./Model_Config/apn.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --train-lradj ExponentialDecayLR \
#     --data-root_path dataset/ETT-small \
#     --data-data_path $dataset_name \
#     --model_name APN_adapter \
#     --model-threshold 0.5 \
#     --model-enc_in 8 \
#     --model-seq_len 96 \
#     --model-pred_len 96 \
#     --GPU-gpu 1 \
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

# # ============================================================
# # DLinear_adapter
# # ============================================================

# # DLinear - exchange
# python ./Run/train.py \
#     --exp_name "irregular" \
#     --data-data_name "ETTh1" \
#     --config_filename ./Model_Config/dlinear.yaml \
#     --des 1 \
#     --random_seed 2021 \
#     --train-lradj ExponentialDecayLR \
#     --data-root_path dataset/ETT-small \
#     --data-data_path $dataset_name \
#     --model_name DLinear_adapter \
#     --model-threshold 0.5 \
#     --model-enc_in 8 \
#     --model-seq_len 96 \
#     --model-pred_len 96 \
#     --GPU-gpu 1 \
#     --model-main_dim 8 \
#     --data-target_num 8 \
#     --model-d_model 64 \
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

done