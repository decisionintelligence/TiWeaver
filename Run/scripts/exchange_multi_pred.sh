#!/bin/bash
# Exchange系列数据集多pred_len实验脚本
# 数据集: exchange, exchange_freq, exchange_masked_20pct
# pred_len: 96, 192, 336, 720

model_name="dynamic_patching_model_GAN"
CHECKPOINTS="./Run/TiWeaver/exchange_mulstep/checkpoints"
LOGS="./Run/TiWeaver/exchange_mulstep/logs"

# ============================================================
# exchange.csv
# ============================================================

# exchange pred_len=336
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange.csv \
    --model_name $model_name \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 336 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
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
    --data-batch_size 128 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# exchange pred_len=720
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange.csv \
    --model_name $model_name \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 720 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
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
    --data-batch_size 64 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

exit 0

# ============================================================
# exchange_freq.csv
# ============================================================

# exchange_freq pred_len=96
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_freq.csv \
    --model_name $model_name \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-d_model 64 \
    --model-d_ff 128 \
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
    --config_filename ./Model_Config/TiWeaver.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_freq.csv \
    --model_name $model_name \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 192 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-d_model 64 \
    --model-d_ff 128 \
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
    --config_filename ./Model_Config/TiWeaver.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_freq.csv \
    --model_name $model_name \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 336 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-d_model 64 \
    --model-d_ff 128 \
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
    --config_filename ./Model_Config/TiWeaver.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_freq.csv \
    --model_name $model_name \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 720 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --train-lr 0.0005 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 8 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# ============================================================
# exchange_masked_20pct.csv
# ============================================================

# exchange_masked_20pct pred_len=96
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_masked_20pct.csv \
    --model_name $model_name \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --train-lr 0.001 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 128 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# exchange_masked_20pct pred_len=192
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_masked_20pct.csv \
    --model_name $model_name \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 192 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --train-lr 0.001 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 128 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# exchange_masked_20pct pred_len=336
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_masked_20pct.csv \
    --model_name $model_name \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 336 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --train-lr 0.001 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 128 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

# exchange_masked_20pct pred_len=720
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "exchange_ir" \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/exchange \
    --data-data_path exchange_masked_20pct.csv \
    --model_name $model_name \
    --model-threshold 0.5 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len 720 \
    --GPU-gpu 2 \
    --model-main_dim 8 \
    --data-target_num 8 \
    --train-lr 0.001 \
    --model-e_layers 3 \
    --model-min_patch_size 4 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --data-batch_size 64 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS
