#!/bin/bash
# =============================================================================
# Launch both CodeGen and Solver vLLM services for Proposer training.
#
# GPU Layout (3+2+3 split):
#   GPUs 0-2: Training (not managed here)
#   GPUs 3-4: CodeGen service — generates SVG code (2 instances)
#   GPUs 5-7: Solver service  — evaluates rendered images (3 instances)
#
# Ports: CodeGen 70XX, 70XX+1; Solver 60YY..60YY+3. Chosen ports in proposer_service_ports.env.
#
# Usage: bash MM-zero_final/vllm_service_init/start_proposer_services.sh <codegen_model> <solver_model> [max_model_len]
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

# Pick random 70XX, 70XX+1 (CodeGen) and 60YY..60YY+2 (Solver)
max_attempts=30
for attempt in $(seq 1 $max_attempts); do
    code_xx=$((RANDOM % 87 + 10))   # 10..96
    solve_yy=$((RANDOM % 87 + 10))  # 10..96 so 60YY+2 <= 6098
    CODEGEN_PORT_0=$((7000 + code_xx))
    CODEGEN_PORT_1=$((7000 + code_xx + 1))
    SOLVER_PORT_0=$((6000 + solve_yy))
    SOLVER_PORT_1=$((6000 + solve_yy + 1))
    SOLVER_PORT_2=$((6000 + solve_yy + 2))
    if port_in_use $CODEGEN_PORT_0 || port_in_use $CODEGEN_PORT_1 || \
       port_in_use $SOLVER_PORT_0 || port_in_use $SOLVER_PORT_1 || port_in_use $SOLVER_PORT_2; then
        [ $attempt -lt $max_attempts ] || { echo "[proposer_services] ERROR: Could not find free ports after $max_attempts attempts"; exit 1; }
        continue
    fi
    break
done

echo "export CODEGEN_PORT_0=$CODEGEN_PORT_0"  > "$PORTS_FILE"
echo "export CODEGEN_PORT_1=$CODEGEN_PORT_1" >> "$PORTS_FILE"
echo "export SOLVER_PORT_0=$SOLVER_PORT_0"   >> "$PORTS_FILE"
echo "export SOLVER_PORT_1=$SOLVER_PORT_1"   >> "$PORTS_FILE"
echo "export SOLVER_PORT_2=$SOLVER_PORT_2"   >> "$PORTS_FILE"
echo "[proposer_services] Ports: CodeGen ${CODEGEN_PORT_0},${CODEGEN_PORT_1} Solver ${SOLVER_PORT_0},${SOLVER_PORT_1},${SOLVER_PORT_2} ($PORTS_FILE)"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
SOLVER_N_ROLLOUTS="${SOLVER_N_ROLLOUTS:-5}"
CODEGEN_EXTRA=" --gpu_mem_util $GPU_MEM_UTIL"
SOLVER_EXTRA=" --gpu_mem_util $GPU_MEM_UTIL --n_rollouts $SOLVER_N_ROLLOUTS"
if [ -n "$max_model_len" ]; then
    CODEGEN_EXTRA="$CODEGEN_EXTRA --max_model_len $max_model_len"
    SOLVER_EXTRA="$SOLVER_EXTRA --max_model_len $max_model_len"
    echo "[proposer_services] Using max_model_len=$max_model_len, gpu_mem_util=$GPU_MEM_UTIL"
fi

PIDS_FILE="$LOG_DIR/proposer_service_pids.env"

echo "[proposer_services] Starting CodeGen on GPUs 3-4 (ports $CODEGEN_PORT_0, $CODEGEN_PORT_1)..."
nohup env CUDA_VISIBLE_DEVICES=3 python MM-zero_final/vllm_service_init/start_codegen_server.py --port $CODEGEN_PORT_0 --model_path $codegen_model_path $CODEGEN_EXTRA >> "$LOG_DIR/codegen_${CODEGEN_PORT_0}.log" 2>&1 &
CODEGEN_PID_0=$!
nohup env CUDA_VISIBLE_DEVICES=4 python MM-zero_final/vllm_service_init/start_codegen_server.py --port $CODEGEN_PORT_1 --model_path $codegen_model_path $CODEGEN_EXTRA >> "$LOG_DIR/codegen_${CODEGEN_PORT_1}.log" 2>&1 &
CODEGEN_PID_1=$!

echo "[proposer_services] Starting Solver on GPUs 5-7 (ports $SOLVER_PORT_0, $SOLVER_PORT_1, $SOLVER_PORT_2)..."
nohup env CUDA_VISIBLE_DEVICES=5 python MM-zero_final/vllm_service_init/start_vllm_server.py --port $SOLVER_PORT_0 --model_path $solver_model_path $SOLVER_EXTRA >> "$LOG_DIR/solver_${SOLVER_PORT_0}.log" 2>&1 &
SOLVER_PID_0=$!
nohup env CUDA_VISIBLE_DEVICES=6 python MM-zero_final/vllm_service_init/start_vllm_server.py --port $SOLVER_PORT_1 --model_path $solver_model_path $SOLVER_EXTRA >> "$LOG_DIR/solver_${SOLVER_PORT_1}.log" 2>&1 &
SOLVER_PID_1=$!
nohup env CUDA_VISIBLE_DEVICES=7 python MM-zero_final/vllm_service_init/start_vllm_server.py --port $SOLVER_PORT_2 --model_path $solver_model_path $SOLVER_EXTRA >> "$LOG_DIR/solver_${SOLVER_PORT_2}.log" 2>&1 &
SOLVER_PID_2=$!

{
    echo "CODEGEN_PID_0=$CODEGEN_PID_0"
    echo "CODEGEN_PID_1=$CODEGEN_PID_1"
    echo "SOLVER_PID_0=$SOLVER_PID_0"
    echo "SOLVER_PID_1=$SOLVER_PID_1"
    echo "SOLVER_PID_2=$SOLVER_PID_2"
} > "$PIDS_FILE"
echo "[proposer_services] Launched. PIDs in $PIDS_FILE."
