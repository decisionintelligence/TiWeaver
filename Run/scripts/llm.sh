
CHECKPOINTS="./Run/baseline/llm/checkpoints"
LOGS="./Run/baseline/llm/logs"

# python -m debugpy --listen localhost:8891 --wait-for-client ./Run/eval.py \


python ./Run/train.py \
    --exp_name "irregular_ROSE" \
    --data-data_name "irregular" \
    --config_filename "./Model_Config/BAOWU/ROSE.yaml" \
    --des 1 \
    --data-data_path "HumanActivity" \
    --random_seed 2024 \
    --model-enc_in 12 \
    --model-main_dim 12 \
    --data-target_num 12 \
    --model-seq_len 3000 \
    --model-pred_len 1000 \
    --checkpoints $CHECKPOINTS \
    --log_base_dir $LOGS \
    --model-loss-criterion "mse" \
    --GPU-gpu 2

# python ./Run/eval.py \
#     --exp_name "irregular_ROSE" \
#     --data-data_name "irregular" \
#     --config_filename "./Model_Config/BAOWU/ROSE.yaml" \
#     --des 1 \
#     --data-data_path "HumanActivity" \
#     --random_seed 2024 \
#     --model-enc_in 12 \
#     --model-main_dim 12 \
#     --model-zero_shot 1 \
#     --model-d_model 128 \
#     --data-target_num 12 \
#     --model-seq_len 3000 \
#     --model-pred_len 1000 \
#     --checkpoints $CHECKPOINTS \
#     --log_base_dir $LOGS \
#     --model-loss-criterion "mse" \
#     --GPU-gpu 2