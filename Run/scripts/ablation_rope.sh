#!/bin/bash
# 需求5: RoFormer旋转位置编码(RoPE)消融实验脚本
# 在human_activity、exchange、exchange_freq、exchange_masked_20pct上进行消融
# 通过 --model-time_embed_type rope 切换为RoPE时间编码

CHECKPOINTS="./Run/TiWeaver/repo/checkpoints"
LOGS="./Run/TiWeaver/repo/logs"

# ============================================================
# RoPE消融 - HumanActivity
# ============================================================
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "irregular" \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --data-data_path HumanActivity \
    --random_seed 2024 \
    --train-lradj ExponentialDecayLR \
    --des 1 \
    --model_name dynamic_patching_model_GAN \
    --model-seq_len 3000 \
    --model-pred_len 1000 \
    --model-e_layers 3 \
    --model-threshold 0.8 \
    --model-enc_in 12 \
    --model-main_dim 12 \
    --data-target_num 12 \
    --model-d_model 64 \
    --train-lr 0.0003 \
    --data-batch_size 4 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS \
    --model-loss-criterion mse \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --model-time_embed_type rope \
    --GPU-gpu 2

# ============================================================
# RoPE消融 - exchange
# ============================================================
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange.csv \
    --model_name dynamic_patching_model_GAN \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 96 \
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
    --model-time_embed_type rope \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# ============================================================
# RoPE消融 - exchange_freq
# ============================================================
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_freq.csv \
    --model_name dynamic_patching_model_GAN \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 96 \
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
    --data-batch_size 16 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --model-time_embed_type rope \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# ============================================================
# RoPE消融 - exchange_masked_20pct
# ============================================================
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_masked_20pct.csv \
    --model_name dynamic_patching_model_GAN \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 2 \
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
    --model-time_embed_type rope \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS
