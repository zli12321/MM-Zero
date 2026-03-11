#!/bin/bash
# Launch proposal generation across 8 GPUs in parallel.
# Usage: bash MM-zero_final/proposal_generate/proposal_generate.bash <model_path> <num_samples_per_gpu> <save_name>

model_name=$1
num_samples=$2
save_name=$3

export VLLM_DISABLE_COMPILE_CACHE=1

echo "Generating proposals with model: $model_name"
echo "  Samples per GPU: $num_samples"
echo "  Save name: $save_name"

for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python MM-zero_final/proposal_generate/proposal_generate.py \
        --model $model_name --suffix $i --num_samples $num_samples --save_name $save_name &
done

wait
echo "Proposal generation finished for all GPUs."
