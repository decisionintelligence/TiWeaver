#!/bin/bash
CHECKPOINTS="./Run/baseline/tst/checkpoints"
LOGS="./Run/baseline/tst/logs"

for random_seed in 2021          #2022 2023
do

# for data_path in "ETTh1_freq.csv" "ETTh1_masked_20pct.csv" "ETTh1.csv"
# do

# python ./Run/train.py \
#     --config_filename ./Model_Config/BAOWU/PatchTST.yaml \
#     --des $random_seed \
#     --random_seed $random_seed \
#     --data-data_path $data_path \
#     --data-root_path "dataset/ETT-small" \
#     --data-data_name "ETTh1" \
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

# done

for data_path in "ETTm1_freq.csv" "ETTm1_masked_20pct.csv" "ETTm1.csv"
do

python ./Run/train.py \
    --config_filename ./Model_Config/BAOWU/PatchTST.yaml \
    --des $random_seed \
    --random_seed $random_seed \
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
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

done

for data_path in "electricity_freq.csv" "electricity_masked_20pct.csv" "electricity.csv"
do

python ./Run/train.py \
    --config_filename ./Model_Config/BAOWU/PatchTST.yaml \
    --des $random_seed \
    --random_seed $random_seed \
    --data-data_path $data_path \
    --data-root_path "dataset/electricity" \
    --data-data_name "electricity" \
    --exp_name irregular \
    --model-main_dim 8 \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --model-enc_in 8 \
    --data-target_num 8 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS

done

for pred_len in 192 336 720
do
for data_path in "exchange_freq.csv" "exchange_masked_20pct.csv" "exchange.csv"
do

python ./Run/train.py \
    --config_filename ./Model_Config/BAOWU/PatchTST.yaml \
    --des $random_seed \
    --random_seed $random_seed \
    --data-data_path $data_path \
    --data-root_path "dataset/exchange" \
    --data-data_name "exchange_ir" \
    --data-split_dataset "[0.7, 0.1, 0.2]" \
    --exp_name irregular \
    --model-main_dim 8 \
    --model-enc_in 8 \
    --model-d_model 128 \
    --model-d_ff 256 \
    --model-enc_in 8 \
    --data-target_num 8 \
    --model-seq_len 96 \
    --model-pred_len $pred_len \
    --train-lr 0.05 \
    --GPU-gpu 1 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS
    
done
done

done