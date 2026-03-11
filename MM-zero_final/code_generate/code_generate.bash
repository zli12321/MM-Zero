#!/bin/bash
# Launch code generation across 8 GPUs in parallel.
# Usage: bash MM-zero_final/code_generate/code_generate.bash <model_path> <experiment_name>

model_name=$1
save_name=$2

export VLLM_DISABLE_COMPILE_CACHE=1

for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python MM-zero_final/code_generate/code_generate.py \
        --model $model_name --suffix $i --save_name $save_name &
done

wait
echo "Code generation finished for all GPUs."
