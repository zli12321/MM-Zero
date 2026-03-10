#!/bin/bash
# Evaluate rendered images with hard questions using the Solver model.
# Runs in parallel across 8 GPUs.
# Usage: bash SelfAgent/question_evaluate/evaluate_imagefree.sh <solver_model_path> <experiment_name>
#
# Timeout: After this many seconds from script start, any evaluation worker still running is killed.
# Set EVALUATE_TIMEOUT_SEC to override (default 3600 = 1 hour).

model_name=$1
save_name=$2

timeout_duration=${EVALUATE_TIMEOUT_SEC:-3600}
echo "[EVALUATE] Starting 8 workers (GPUs 0-7). Timeout: ${timeout_duration}s. Progress: [0]..[7] 'EVALUATE chunk X/Y' and 'EVALUATE progress'."
pids=()

for i in {0..7}; do
  CUDA_VISIBLE_DEVICES=$i python -u SelfAgent/question_evaluate/evaluate_imagefree.py \
      --model $model_name --suffix $i --save_name $save_name &
  pids[$i]=$!
done

# Wait for the first GPU to finish
wait ${pids[0]}
echo "[EVALUATE] Task 0 finished."

(
  sleep $timeout_duration
  echo "[EVALUATE] Timeout reached (${timeout_duration}s). Killing remaining tasks..."
  for i in {1..7}; do
    if kill -0 ${pids[$i]} 2>/dev/null; then
      kill -9 ${pids[$i]} 2>/dev/null
      echo "Killed task $i"
    fi
  done
) &

for i in {1..7}; do
  wait ${pids[$i]} 2>/dev/null
done

echo "[EVALUATE] All evaluation tasks finished."
