#!/bin/bash
# ETTh1/ETTm1/electricity 及其变体 pred_len=96 实验脚本
# 数据集: ETTh1, ETTm1, electricity (原始/freq/masked_20pct)
# 共 3 x 3 = 9 个实验

model_name="dynamic_patching_model_GAN"
CHECKPOINTS="./Run/TiWeaver/ettm/checkpoints"
LOGS="./Run/TiWeaver/ettm/logs"

# ============================================================
# ETTh1 (7 variables, 17420 rows, hourly)
# ============================================================

# for tau in 0.4 0.5 0.6 0.8; do
for tau in 0.6 0.4; do
for min_patch_size in 24 16 32 8; do
for batch_size in 512 256 128 64 32; do
for lr in 0.0005 0.0001; do
for data_name in "ETTm1.csv"; do
# ETTm1 原始
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "ETTm1" \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/ETT-small \
    --data-data_path $data_name \
    --model_name $model_name \
    --model-threshold $tau \
    --model-enc_in 7 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 1 \
    --model-main_dim 7 \
    --data-target_num 7 \
    --train-lr $lr \
    --model-e_layers 3 \
    --model-min_patch_size $min_patch_size \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.6, 0.2, 0.2]" \
    --data-batch_size $batch_size \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS
done
done
done
done
done