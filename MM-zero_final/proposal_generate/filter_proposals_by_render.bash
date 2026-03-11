#!/bin/bash
# Filter proposals by render success rate across 8 GPUs in parallel.
# Each GPU processes its own shard file (save_name_0.json .. save_name_7.json).
# Usage: bash filter_proposals_by_render.bash <codegen_model> <save_name> [n_samples] [min_rate] [max_rate] [render_workers]

codegen_model=$1
save_name=$2
n_samples=${3:-8}
min_rate=${4:-0.25}
max_rate=${5:-0.75}
render_workers=${6:-8}

echo "[filter] Filtering proposals across 8 GPUs (n_samples=$n_samples, rate=[$min_rate, $max_rate])"

for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python MM-zero_final/proposal_generate/filter_proposals_by_render.py \
        --codegen_model $codegen_model \
        --save_name $save_name \
        --suffix $i \
        --n_samples $n_samples \
        --min_render_rate $min_rate \
        --max_render_rate $max_rate \
        --render_workers $render_workers &
done

wait
echo "[filter] All 8 filter shards finished."

# Print combined summary
python3 -c "
import json, os, glob
save_name = '${save_name}'
storage = os.environ.get('STORAGE_PATH', '')
total = 0
for i in range(8):
    fpath = f'{storage}/generated_proposals/{save_name}_{i}.json'
    if os.path.exists(fpath):
        with open(fpath) as f:
            data = json.load(f)
            total += len(data)
print(f'[filter] Total proposals after filtering: {total}')
"
