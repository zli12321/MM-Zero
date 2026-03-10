#!/bin/bash
# =============================================================================
# Launch both CodeGen and Solver vLLM services for Proposer training.
#
# GPU Layout:
#   GPUs 4-5: CodeGen service — generates matplotlib code
#   GPUs 6-7: Solver service  — evaluates rendered images
#
# Ports: vLLM uses 70XX/70XX+1 (CodeGen) and 60YY/60YY+1 (Solver) with XX,YY random 10-98,
# so 6000-6001 and 7000-7001 stay free for GPU server. Chosen ports written to
# temp_results/proposer_service_ports.env. Retry with new random if any port is in use.
#
# Usage: bash SelfAgent/vllm_service_init/start_proposer_services.sh <codegen_model> <solver_model> [max_model_len]
# =============================================================================

codegen_model_path=$1
solver_model_path=$2
max_model_len=${3:-}

export VLLM_DISABLE_COMPILE_CACHE=1

LOG_DIR="${STORAGE_PATH:-.}/temp_results"
mkdir -p "$LOG_DIR"
PORTS_FILE="$LOG_DIR/proposer_service_ports.env"

port_in_use() {
    local p=$1
    python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
try:
    exit(0 if s.connect_ex(('127.0.0.1', $p)) == 0 else 1)
finally:
    s.close()
" 2>/dev/null
}

# Pick random 70XX,70XX+1 and 60YY,60YY+1 (XX,YY in 10-98 to avoid 6000/6001, 7000/7001)
max_attempts=30
for attempt in $(seq 1 $max_attempts); do
    code_xx=$((RANDOM % 89 + 10))   # 10..98
    solve_yy=$((RANDOM % 89 + 10))  # 10..98
    CODEGEN_PORT_0=$((7000 + code_xx))
    CODEGEN_PORT_1=$((7000 + code_xx + 1))
    SOLVER_PORT_0=$((6000 + solve_yy))
    SOLVER_PORT_1=$((6000 + solve_yy + 1))
    if port_in_use $CODEGEN_PORT_0 || port_in_use $CODEGEN_PORT_1 || \
       port_in_use $SOLVER_PORT_0 || port_in_use $SOLVER_PORT_1; then
        [ $attempt -lt $max_attempts ] || { echo "[proposer_services] ERROR: Could not find free ports after $max_attempts attempts"; exit 1; }
        continue
    fi
    break
done

echo "export CODEGEN_PORT_0=$CODEGEN_PORT_0"  > "$PORTS_FILE"
echo "export CODEGEN_PORT_1=$CODEGEN_PORT_1" >> "$PORTS_FILE"
echo "export SOLVER_PORT_0=$SOLVER_PORT_0"   >> "$PORTS_FILE"
echo "export SOLVER_PORT_1=$SOLVER_PORT_1"   >> "$PORTS_FILE"
echo "[proposer_services] Ports: CodeGen ${CODEGEN_PORT_0},${CODEGEN_PORT_1} Solver ${SOLVER_PORT_0},${SOLVER_PORT_1} (70${code_xx}xx 60${solve_yy}xx) ($PORTS_FILE)"

# Use 0.5 so vLLM fits when GPU has other use (e.g. ~21 GiB free). Override with GPU_MEM_UTIL=0.8 if GPU is dedicated.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
CODEGEN_EXTRA=" --gpu_mem_util $GPU_MEM_UTIL"
SOLVER_EXTRA=" --gpu_mem_util $GPU_MEM_UTIL"
if [ -n "$max_model_len" ]; then
    CODEGEN_EXTRA="$CODEGEN_EXTRA --max_model_len $max_model_len"
    SOLVER_EXTRA="$SOLVER_EXTRA --max_model_len $max_model_len"
    echo "[proposer_services] Using max_model_len=$max_model_len (40GB tier), gpu_mem_util=$GPU_MEM_UTIL"
fi

CODEGEN_LOG="$LOG_DIR/codegen_${CODEGEN_PORT_0}.log"
CODEGEN_LOG_1="$LOG_DIR/codegen_${CODEGEN_PORT_1}.log"
SOLVER_LOG_0="$LOG_DIR/solver_${SOLVER_PORT_0}.log"
SOLVER_LOG_1="$LOG_DIR/solver_${SOLVER_PORT_1}.log"

# PIDs for cleanup (proposer_train.sh kills these when training finishes)
PIDS_FILE="$LOG_DIR/proposer_service_pids.env"

echo "[proposer_services] Starting CodeGen on GPUs 4-5 (ports $CODEGEN_PORT_0, $CODEGEN_PORT_1)..."
nohup env CUDA_VISIBLE_DEVICES=4 python SelfAgent/vllm_service_init/start_codegen_server.py --port $CODEGEN_PORT_0 --model_path $codegen_model_path $CODEGEN_EXTRA >> "$CODEGEN_LOG" 2>&1 &
CODEGEN_PID_0=$!
nohup env CUDA_VISIBLE_DEVICES=5 python SelfAgent/vllm_service_init/start_codegen_server.py --port $CODEGEN_PORT_1 --model_path $codegen_model_path $CODEGEN_EXTRA >> "$CODEGEN_LOG_1" 2>&1 &
CODEGEN_PID_1=$!

echo "[proposer_services] Starting Solver on GPUs 6-7 (ports $SOLVER_PORT_0, $SOLVER_PORT_1)..."
nohup env CUDA_VISIBLE_DEVICES=6 python SelfAgent/vllm_service_init/start_vllm_server.py --port $SOLVER_PORT_0 --model_path $solver_model_path $SOLVER_EXTRA >> "$SOLVER_LOG_0" 2>&1 &
SOLVER_PID_0=$!
nohup env CUDA_VISIBLE_DEVICES=7 python SelfAgent/vllm_service_init/start_vllm_server.py --port $SOLVER_PORT_1 --model_path $solver_model_path $SOLVER_EXTRA >> "$SOLVER_LOG_1" 2>&1 &
SOLVER_PID_1=$!

{
    echo "CODEGEN_PID_0=$CODEGEN_PID_0"
    echo "CODEGEN_PID_1=$CODEGEN_PID_1"
    echo "SOLVER_PID_0=$SOLVER_PID_0"
    echo "SOLVER_PID_1=$SOLVER_PID_1"
} > "$PIDS_FILE"
echo "[proposer_services] Launched. PIDs in $PIDS_FILE. Logs: $CODEGEN_LOG $CODEGEN_LOG_1 $SOLVER_LOG_0 $SOLVER_LOG_1"
