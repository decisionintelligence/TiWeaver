#!/bin/bash
# ETTh1/ETTm1/electricity 及其变体 pred_len=96 实验脚本
# 数据集: ETTh1, ETTm1, electricity (原始/freq/masked_20pct)
# 共 3 x 3 = 9 个实验

model_name="dynamic_patching_model_GAN"
CHECKPOINTS="./Run/TiWeaver/etth/checkpoints"
LOGS="./Run/TiWeaver/etth/logs"

# ============================================================
# ETTh1 (7 variables, 17420 rows, hourly)
# ============================================================

# for tau in 0.4 0.5 0.6 0.8; do
for tau in 0.5; do
for batch_size in 32 64 128; do
for lr in 0.0005 0.001 0.0001; do
for min_patch_size in 12 8; do
for data_name in "ETTh1.csv" "ETTh1_masked_20pct.csv" "ETTh1_freq.csv"; do
# ETTh1 原始
python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "ETTh1" \
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