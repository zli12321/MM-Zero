#!/bin/bash
# ==========================================================================
# Run evaluation with DATA PARALLELISM: 8 copies of the model (one per GPU),
# each processing 1/8 of each dataset. ~8x faster than tensor parallelism
# for a 4B model.
#
# For each model checkpoint:
#   1. Launch 8 workers (shard 0-7), one per GPU
#   2. Wait for all to finish
#   3. Merge shard JSONL files into one per dataset
#   4. Compute final accuracy
#
# Usage:
#   cd /workspace/Self-Agent
#   bash run_eval.sh
# ==========================================================================

set -x
set -uo pipefail
export PYTHONUNBUFFERED=1

# ---- Paths ----
# STORAGE="${STORAGE:-/workspace/selfAgent_Storage_qwen3vl_8b_round6_reward_optimization}"
STORAGE="${STORAGE:-/workspace/selfAgent_Storage_svg_long_round6_filter}"
OUTPUT_DIR="${STORAGE}/eval_responses"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NUM_GPUS=8

# ---- LLM Judge (used at the end after all evals) ----
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-14B-Instruct}"

# ---- Datasets ----
DATASETS=(
  "zli12321/MMSI"
  "zli12321/mathverse"
  "zli12321/mathvision"
  "zli12321/mathvista"
  "zli12321/mm-vet"
  "zli12321/mmmu_pro_4_options"
  "zli12321/visnumbench"
  "zli12321/mmmu_pro_10options"
  "zli12321/mmmu-pro-vision"
  "zli12321/hallusionbench"
  "zli12321/MMMU"
  "zli12321/ChartQA"
)
DS_ARGS="${DATASETS[*]}"

# ---- Base model + solver checkpoints ----
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
declare -a MODEL_NAMES=()
declare -a MODEL_PATHS=()

MODEL_NAMES+=("base")
MODEL_PATHS+=("${BASE_MODEL}")

for dir in "${STORAGE}/models"/*solver_*/global_step_*/actor/huggingface; do
  [ -d "$dir" ] || continue
  solver_part=$(echo "$dir" | grep -oP 'solver_v[0-9]+')
  step=$(echo "$dir" | grep -oP 'global_step_\K[0-9]+')
  name="${solver_part}_step${step}"
  MODEL_NAMES+=("${name}")
  MODEL_PATHS+=("${dir}")
done

echo "============================================"
echo "Models to evaluate: ${#MODEL_NAMES[@]}"
for i in "${!MODEL_NAMES[@]}"; do
  echo "  [${i}] ${MODEL_NAMES[$i]}"
done
echo "Datasets: ${DATASETS[*]}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Strategy: ${NUM_GPUS}-way data parallelism (TP=1)"
echo "============================================"

mkdir -p "${OUTPUT_DIR}"
> "${OUTPUT_DIR}/accuracy_summary.jsonl"

SCRIPT_START=$SECONDS

# ---- Merge function: combine shards and compute accuracy ----
merge_shards() {
  local save_dir="$1"
  local short_name="$2"
  local model_name="$3"
  local model_path="$4"

  local merged="${save_dir}/${short_name}.jsonl"
  > "${merged}"

  for shard_file in "${save_dir}/${short_name}".shard*.jsonl; do
    [ -f "${shard_file}" ] || continue
    cat "${shard_file}" >> "${merged}"
    rm -f "${shard_file}"
  done

  python3 -c "
import json, sys
correct = total = 0
with open('${merged}') as f:
    for line in f:
        d = json.loads(line)
        total += 1
        if d.get('correct'):
            correct += 1
acc = correct / total * 100 if total > 0 else 0.0
print(f'  {\"${short_name}\"}: {correct}/{total} = {acc:.2f}%')
with open('${OUTPUT_DIR}/accuracy_summary.jsonl', 'a') as f:
    f.write(json.dumps({
        'model': '${model_name}',
        'model_path': '${model_path}',
        'dataset': '${short_name}',
        'accuracy': round(acc, 2),
        'correct': correct,
        'total': total,
    }) + '\n')
"
}

# ---- Run each model ----
for i in "${!MODEL_NAMES[@]}"; do
  name="${MODEL_NAMES[$i]}"
  path="${MODEL_PATHS[$i]}"

  MODEL_START=$SECONDS

  echo ""
  echo ">>> [$(( i + 1 ))/${#MODEL_NAMES[@]}] Evaluating ${name} with ${NUM_GPUS}-way data parallelism..."
  echo ""

  pids=()
  for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=${gpu} python3 "${SCRIPT_DIR}/eval_generate.py" \
      --model_path "${path}" \
      --save_name "${name}" \
      --datasets ${DS_ARGS} \
      --output_dir "${OUTPUT_DIR}" \
      --n 1 \
      --temperature 0.6 \
      --top_p 0.95 \
      --gpu_mem_util 0.85 \
      --shard_id ${gpu} \
      --num_shards ${NUM_GPUS} \
      > "${OUTPUT_DIR}/${name}.shard${gpu}.log" 2>&1 &
    pids+=($!)
    echo "  Launched shard ${gpu} on GPU ${gpu} (PID ${pids[-1]})"
  done

  echo "  Waiting for all ${NUM_GPUS} shards..."
  echo ""

  # ---- Progress monitor: poll logs every 10s until all shards finish ----
  # (set +x so the table prints as one block; otherwise trace mixes with rows)
  set +x
  while true; do
    alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        ((alive++))
      fi
    done
    [ $alive -eq 0 ] && break

    # Print progress: dataset name + item progress (current/total)
    elapsed_s=$(( SECONDS - MODEL_START ))
    elapsed_m=$(( elapsed_s / 60 ))
    elapsed_rem=$(( elapsed_s % 60 ))
    printf "\r\033[K"
    echo "  ┌─ Eval progress (${alive}/${NUM_GPUS} shards active) ── elapsed: ${elapsed_m}m ${elapsed_rem}s ──────────────"
    echo "  │  GPU  │  Dataset (file)        │  Progress (items)     │  Chunk     │  Done"
    echo "  ├───────┼────────────────────────┼───────────────────────┼────────────┼───────"
    for gpu in $(seq 0 $((NUM_GPUS - 1))); do
      logf="${OUTPUT_DIR}/${name}.shard${gpu}.log"
      if [ ! -f "$logf" ]; then
        dataset="—"
        progress="—"
        chunk_info="—"
        done_count="0"
      elif kill -0 "${pids[$gpu]}" 2>/dev/null; then
        # Current dataset short name (e.g. MMSI, OmniSpatial); -- so pattern starting with --- is not treated as options
        dataset=$(grep -oP -- '--- Dataset: \S+ \(\K[^)]+' "$logf" 2>/dev/null | tail -1)
        [ -z "$dataset" ] && dataset="loading..."
        # Item progress from "[shard N] chunk A/B (current/total)..."
        progress=$(grep -oP '\[shard [0-9]+\] chunk [0-9]+/[0-9]+ \K\([0-9]+/[0-9]+\)' "$logf" 2>/dev/null | tail -1)
        [ -z "$progress" ] && progress="—"
        # Chunk X/Y
        chunk_info=$(grep -oP '\[shard [0-9]+\] chunk \K[0-9]+/[0-9]+' "$logf" 2>/dev/null | tail -1)
        [ -z "$chunk_info" ] && chunk_info="—"
        done_count=$(grep -c 'Done in' "$logf" 2>/dev/null); done_count=${done_count:-0}
      else
        dataset="(finished)"
        progress="—"
        chunk_info="—"
        done_count=$(grep -c 'Done in' "$logf" 2>/dev/null); done_count=${done_count:-0}
      fi
      printf "  │  %2d   │  %-22s │  %-21s │  %-10s │  %s/%s\n" \
        "$gpu" "$dataset" "$progress" "$chunk_info" "$done_count" "${#DATASETS[@]}"
    done
    echo "  └───────┴────────────────────────┴───────────────────────┴────────────┴───────"
    echo ""

    sleep 10
  done
  set -x

  # Collect exit codes
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "  WARNING: Worker PID ${pid} exited with error"
      ((failed++)) || true
    fi
  done

  if [ $failed -gt 0 ]; then
    echo "  WARNING: ${failed} shard(s) failed for ${name}"
  fi

  echo "  Merging shard results..."
  for DS in "${DATASETS[@]}"; do
    short="${DS##*/}"
    merge_shards "${OUTPUT_DIR}/${name}" "${short}" "${name}" "${path}"
  done

  model_elapsed=$(( SECONDS - MODEL_START ))
  model_min=$(( model_elapsed / 60 ))
  model_sec=$(( model_elapsed % 60 ))
  echo "<<< ${name} finished in ${model_min}m ${model_sec}s."
done

total_elapsed=$(( SECONDS - SCRIPT_START ))
total_min=$(( total_elapsed / 60 ))
total_sec=$(( total_elapsed % 60 ))

echo ""
echo "============================================"
echo "All evaluations complete. Total time: ${total_min}m ${total_sec}s"
echo "============================================"
echo ""
echo "Accuracy Summary:"
echo "-------------------------------------------"
printf "%-25s %-20s %s\n" "Model" "Dataset" "Accuracy"
echo "-------------------------------------------"

if [ -f "${OUTPUT_DIR}/accuracy_summary.jsonl" ]; then
  while IFS= read -r line; do
    model=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['model'])")
    dataset=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['dataset'])")
    acc=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['accuracy']:.2f}%\")")
    count=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['correct']}/{d['total']}\")")
    printf "%-25s %-20s %s (%s)\n" "$model" "$dataset" "$acc" "$count"
  done < "${OUTPUT_DIR}/accuracy_summary.jsonl"
fi

echo "-------------------------------------------"
echo ""
echo "Response files:  ${OUTPUT_DIR}/<model_name>/<dataset>.jsonl"
echo "Summary file:    ${OUTPUT_DIR}/accuracy_summary.jsonl"
echo "Per-shard logs:  ${OUTPUT_DIR}/<model_name>.shard<N>.log"

# ==========================================================================
# LLM Judge: score eval responses using a judge model (data-parallel, 8 GPUs)
# ==========================================================================
echo ""
echo "============================================"
echo "Running LLM Judge"
echo "  JUDGE_MODEL: ${JUDGE_MODEL}"
echo "  EVAL_DIR:    ${OUTPUT_DIR}"
echo "  NUM_SHARDS:  ${NUM_GPUS}"
echo "============================================"

for i in $(seq 0 $((NUM_GPUS - 1))); do
  echo "Starting judge shard $i on GPU $i..."
  CUDA_VISIBLE_DEVICES=$i python3 "$SCRIPT_DIR/llm_judge_eval.py" \
    --eval_responses_dir "$OUTPUT_DIR" \
    --judge_model "$JUDGE_MODEL" \
    --shard_id $i \
    --num_shards $NUM_GPUS &
done
wait
echo ""
echo "All judge shards done. Merging..."
python3 "$SCRIPT_DIR/llm_judge_eval.py" --eval_responses_dir "$OUTPUT_DIR" --merge_only
echo "Done. LLM judge summary: ${OUTPUT_DIR}/llm_accuracy_summary.jsonl"

# ==========================================================================
# Accuracy comparison vs base model
# ==========================================================================
echo ""
echo "============================================"
echo "Running accuracy comparison vs base"
echo "============================================"
python3 "$SCRIPT_DIR/eval_accuracy_comparison.py" "${OUTPUT_DIR}/llm_accuracy_summary.jsonl"
