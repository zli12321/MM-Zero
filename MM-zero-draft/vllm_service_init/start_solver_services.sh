#!/bin/bash
# =============================================================================
# Launch Solver vLLM services on GPUs 4-7 for CodeGen training reward.
#
# Usage: bash SelfAgent/vllm_service_init/start_solver_services.sh <model_path> <run_id> [max_model_len]
#   max_model_len: optional; use for 40GB GPUs (e.g. 32768).
#   Set SOLVER_GPU_MEM_UTIL env var to control gpu_mem_util (default 0.45).
# =============================================================================

model_path=$1
run_id=$2
max_model_len=${3:-}
gpu_mem_util=${SOLVER_GPU_MEM_UTIL:-0.45}
export VLLM_DISABLE_COMPILE_CACHE=1
EXTRA=" --gpu_mem_util $gpu_mem_util"
if [ -n "$max_model_len" ]; then
    EXTRA="$EXTRA --max_model_len $max_model_len"
fi
echo "[solver_services] gpu_mem_util=$gpu_mem_util, max_model_len=${max_model_len:-auto}, EXTRA='$EXTRA'"

# nohup so Solver survives when this script exits (avoid SIGHUP killing services)
LOG_DIR="${STORAGE_PATH:-.}/temp_results"
mkdir -p "$LOG_DIR"
# PIDs for cleanup (codegen_train.sh kills these when training finishes)
PIDS_FILE="$LOG_DIR/solver_service_pids.env"

nohup env CUDA_VISIBLE_DEVICES=4 python SelfAgent/vllm_service_init/start_vllm_server.py --port 6000 --model_path $model_path $EXTRA >> "$LOG_DIR/solver_6000.log" 2>&1 &
SOLVER_PID_0=$!
nohup env CUDA_VISIBLE_DEVICES=5 python SelfAgent/vllm_service_init/start_vllm_server.py --port 6001 --model_path $model_path $EXTRA >> "$LOG_DIR/solver_6001.log" 2>&1 &
SOLVER_PID_1=$!
nohup env CUDA_VISIBLE_DEVICES=6 python SelfAgent/vllm_service_init/start_vllm_server.py --port 6002 --model_path $model_path $EXTRA >> "$LOG_DIR/solver_6002.log" 2>&1 &
SOLVER_PID_2=$!
nohup env CUDA_VISIBLE_DEVICES=7 python SelfAgent/vllm_service_init/start_vllm_server.py --port 6003 --model_path $model_path $EXTRA >> "$LOG_DIR/solver_6003.log" 2>&1 &
SOLVER_PID_3=$!

{
    echo "SOLVER_PID_0=$SOLVER_PID_0"
    echo "SOLVER_PID_1=$SOLVER_PID_1"
    echo "SOLVER_PID_2=$SOLVER_PID_2"
    echo "SOLVER_PID_3=$SOLVER_PID_3"
} > "$PIDS_FILE"
echo "[solver_services] Launched on ports 6000-6003. PIDs in $PIDS_FILE. Logs: $LOG_DIR/solver_*.log"
