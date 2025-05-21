#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Define pruning methods and their corresponding pruning ratios
declare -A pruning_ratios=(
    ["random"]=0.10
    ["l1"]=0.05
    ["l2"]=0.05
    ["taylor"]=0.10
)

# List of base model paths
base_models=(
    "../../TrainedCheckpoint/Llama-3-8B/lora/sft/EverTracer/xsum/target/100/merge_lora"
)

for base_model in "${base_models[@]}"; do
    # Get the last folder name from the base model path
    model_name=$(basename "$base_model")
    prune_rate_zero_done=true

    for pruner_type in "${!pruning_ratios[@]}"; do
        pruning_ratio=${pruning_ratios[$pruner_type]}
        save_ckpt_log_name="${base_model}/prune/${pruner_type}-${pruning_ratio}"
        if [ ! -d "$save_ckpt_log_name" ]; then
            echo "Save directory $save_ckpt_log_name does not exist, creating..."
            mkdir -p "$save_ckpt_log_name"
        fi
        echo "Starting model: $model_name, pruner_type = $pruner_type, pruning_ratio = $pruning_ratio"
        python ./LLM-Pruner/hf_prune.py --pruning_ratio "$pruning_ratio" \
            --block_wise \
            --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
            --block_attention_layer_start 4 --block_attention_layer_end 30 \
            --pruner_type "$pruner_type" \
            --device cpu --eval_device cuda:0 \
            --base_model "$base_model" \
            --save_ckpt_log_name "$save_ckpt_log_name" \
            --save_model
    done
done
