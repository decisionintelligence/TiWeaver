#!/bin/bash
# 需求4: 线性插值 + PatchTST 预测实验脚本
# 将不规则数据集通过线性插值转为规则数据集，使用PatchTST进行预测
# 数据集: human_activity

CHECKPOINTS="./Run/baseline/linear/checkpoints"
LOGS="./Run/baseline/linear/logs"

# PatchTST_adapter (with linear interpolation) - HumanActivity
python -m debugpy --listen localhost:8891 --wait-for-client ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "irregular" \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --data-data_path HumanActivity \
    --random_seed 2024 \
    --train-lradj ExponentialDecayLR \
    --des 1 \
    --model_name PatchTST_adapter \
    --model-seq_len 3000 \
    --model-pred_len 1000 \
    --model-e_layers 3 \
    --model-threshold 0.8 \
    --model-enc_in 12 \
    --model-main_dim 12 \
    --data-target_num 12 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-n_heads 2 \
    --model-min_patch_size 16 \
    --train-lr 0.0003 \
    --train-epochs 20 \
    --train-patience 3 \
    --data-batch_size 128 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS \
    --model-loss-criterion mse \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --GPU-gpu 2