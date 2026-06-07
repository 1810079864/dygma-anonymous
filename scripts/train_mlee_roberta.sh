#!/bin/bash

LR=${LR:-3e-5}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-roberta-large}
CUDA_DEVICE=${CUDA_DEVICE:-0}
SEEDS=${SEEDS:-"22 42 66 99 111 1234"}

for SEED in $SEEDS
do
    work_path=exps/mlee/$SEED/$LR
    mkdir -p "$work_path"

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u engine.py \
        --model_type=DyGMA \
        --dataset_type=MLEE \
        --context_representation=decoder \
        --model_name_or_path="$MODEL_NAME_OR_PATH" \
        --role_path=./data/MLEE/MLEE_role_name_mapping.json \
        --prompt_path=./data/prompts/prompts_MLEE_full.csv \
        --seed=$SEED \
        --output_dir="$work_path" \
        --learning_rate=$LR \
        --batch_size 4 \
        --max_steps=10000 \
        --max_enc_seq_length 512 \
        --max_prompt_seq_length 512 \
        --max_span_length 10 \
        --window_size 250 \
        --bipartite \
        --lamb 0.1 \
        --warmup_steps 0.2 \
        --use_dep_role_order \
        --dep_order_weight 0.15 \
        --role_planner_diff_weight 0.25 \
        --role_memory_coref_weight 0.1 \
        --use_dep_memory \
        --role_memory_write_threshold 0.8
done
