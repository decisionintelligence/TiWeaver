#!/bin/bash
# Baseline预测模型实验脚本
# 模型: APN_adapter, DLinear_adapter
# 数据集: human_activity, exchange, exchange_freq, exchange_masked_20pct

CHECKPOINTS="./Run/baseline/tst/checkpoints"
LOGS="./Run/baseline/tst/logs"

# ============================================================
# APN_adapter
# ============================================================

for data_path in "ETTm1.csv" "ETTm1_freq.csv" "ETTm1_masked_20pct.csv" 
do

python ./Run/train.py \
    --config_filename ./Model_Config/BAOWU/PatchTST.yaml \
    --des 2021 \
    --random_seed 2021 \
    --data-data_path $data_path \
    --data-root_path "dataset/ETT-small" \
    --data-data_name "ETTm1" \
    --exp_name irregular \
    --model-main_dim 7 \
    --data-split_dataset "[0.6, 0.2, 0.2]" \
    --model-enc_in 7 \
    --data-target_num 7 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 1 \
    --data-batch_size 512 \
    --train-lr 0.0001 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

done

done
