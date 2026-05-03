# CHECKPOINTS="./Run/tese/tim/checkpoints"
# LOGS="./Run/test/tim/logs"
# # -m debugpy --listen localhost:8891 --wait-for-client 
# python -m debugpy --listen localhost:8891 --wait-for-client ./Run/train.py \
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


model_name="dynamic_patching_model_GAN"
CHECKPOINTS="./Run/TiWeaver/test/checkpoints"
LOGS="./Run/TiWeaver/test/logs"

python ./Run/train.py \
    --exp_name "irregular" \
    --data-data_name "ETTm1" \
    --config_filename ./Model_Config/TiWeaver.yaml \
    --des 1 \
    --random_seed 2021 \
    --train-lradj ExponentialDecayLR \
    --data-root_path dataset/ETT-small \
    --data-data_path "ETTm1_masked_20pct.csv" \
    --model_name $model_name \
    --model-threshold 0.6 \
    --model-enc_in 7 \
    --model-seq_len 96 \
    --model-pred_len 96 \
    --GPU-gpu 3 \
    --model-main_dim 7 \
    --data-target_num 7 \
    --train-lr 0.001 \
    --model-e_layers 3 \
    --model-min_patch_size 24 \
    --model-d_model 64 \
    --model-d_ff 128 \
    --model-time 1 \
    --model-catt 1 \
    --model-gap 1 \
    --model-loss-criterion mse \
    --data-split_dataset "[0.6, 0.2, 0.2]" \
    --data-batch_size 32 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS