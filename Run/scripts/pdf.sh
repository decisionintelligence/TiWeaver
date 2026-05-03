#!/bin/bash

CHECKPOINTS="./Run/baseline/pdf/checkpoints"
LOGS="./Run/baseline/pdf/logs"

for random_seed in 2021
do

for data_path in "ETTh1_freq.csv" "ETTh1_masked_20pct.csv" "ETTh1.csv"
do

python ./Run/train.py \
    --config_filename ./Model_Config/BAOWU/PDF.yaml \
    --des $random_seed \
    --random_seed $random_seed \
    --data-data_path $data_path \
    --data-root_path "dataset/ETT-small" \
    --data-data_name "ETTh1" \
    --exp_name irregular \
    --model-main_dim 7 \
    --model-d_ff 128 \
    --model-d_model 128 \
    --model-dropout 0.25 \
    --model-e_layers 3 \
    --model-fc_dropout 0.15 \
    --model-kernel_list "[3, 7, 11]" \
    --model-n_heads 4 \
    --model-patch_len "[1]" \
    --train-pct_start 0.2 \
    --model-period "[24]" \
    --model-stride "[1]" \
    --data-split_dataset "[0.6, 0.2, 0.2]" \
    --data-target_num 7 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

done

for data_path in "ETTm1_freq.csv" "ETTm1_masked_20pct.csv" "ETTm1.csv"
do

python ./Run/train.py \
    --config_filename ./Model_Config/BAOWU/PDF.yaml \
    --des $random_seed \
    --random_seed $random_seed \
    --data-data_path $data_path \
    --data-root_path "dataset/ETT-small" \
    --data-data_name "ETTm1" \
    --exp_name irregular \
    --model-main_dim 7 \
    --model-d_ff 48 \
    --model-d_model 48 \
    --model-dropout 0.5 \
    --model-e_layers 3 \
    --model-fc_dropout 0.25 \
    --model-kernel_list "[5, 7, 11, 15]" \
    --model-n_heads 16 \
    --model-patch_len "[3, 6, 16, 32, 48]" \
    --train-pct_start 0.2 \
    --train-lradj "TST" \
    --model-period "[48, 90, 102, 360, 720]" \
    --model-stride "[3, 6, 16, 32, 48]" \
    --data-split_dataset "[0.6, 0.2, 0.2]" \
    --data-target_num 7 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

done


for data_path in "electricity_freq.csv" "electricity_masked_20pct.csv" "electricity.csv"
do

python ./Run/train.py \
    --config_filename ./Model_Config/BAOWU/PDF.yaml \
    --des $random_seed \
    --random_seed $random_seed \
    --data-data_path $data_path \
    --data-root_path "dataset/electricity" \
    --data-data_name "electricity" \
    --exp_name irregular \
    --model-main_dim 8 \
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
    --data-target_num 8 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

done


for pred_len in 192 336 720
do 
for data_path in "exchange.csv" "exchange_freq.csv" "exchange_masked_20pct.csv"
do

python ./Run/train.py \
    --config_filename ./Model_Config/BAOWU/PDF.yaml \
    --des $random_seed \
    --random_seed $random_seed \
    --data-data_path $data_path \
    --data-root_path "dataset/exchange" \
    --data-data_name "exchange_ir" \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --exp_name irregular \
    --model-main_dim 8 \
    --data-target_num 8 \
    --model-enc_in 8 \
    --model-seq_len 96 \
    --model-pred_len $pred_len \
    --model-d_ff 256 \
    --model-d_model 128 \
    --model-dropout 0.3 \
    --model-e_layers 1 \
    --model-fc_dropout 0.15 \
    --model-kernel_list "[3, 7, 9, 11]" \
    --model-n_heads 32 \
    --model-patch_len "[1]" \
    --train-pct_start 0.2 \
    --model-period "[24]" \
    --model-stride "[1]" \
    --GPU-gpu 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS
done
done

done