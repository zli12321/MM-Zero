#!/bin/bash
# Kill all running training and vLLM processes so you can restart clean.
# Usage: bash SelfAgent_svg/scripts/kill_training.sh
# Run from repo root (e.g. /workspace/Self-Agent).

set +e
STORAGE_PATH="${STORAGE_PATH:-/workspace/selfAgent_Storage_svg_round1}"

echo "[kill_training] Stopping training and vLLM services..."

# 1) Kill by saved PIDs (CodeGen / Solver from proposer phase)
for PIDS_FILE in "${STORAGE_PATH}/temp_results/proposer_service_pids.env" "${STORAGE_PATH}/temp_results/solver_service_pids.env"; do
    if [ -f "$PIDS_FILE" ]; then
        set -a
        source "$PIDS_FILE" 2>/dev/null || true
        set +a
        for pid in ${CODEGEN_PID_0:-} ${CODEGEN_PID_1:-} ${SOLVER_PID_0:-} ${SOLVER_PID_1:-} ${SOLVER_PID_2:-} ${SOLVER_PID_3:-} ${SOLVER_PID_4:-}; do
            [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        done
        rm -f "$PIDS_FILE"
    fi
done

# 2) Kill by process name (use specific patterns to avoid killing IDE/system processes)
pkill -9 -f "python.*-m vllm" 2>/dev/null || true
pkill -9 -f "vllm\.entrypoints" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f "start_vllm_server\.py" 2>/dev/null || true
pkill -9 -f "start_codegen_server\.py" 2>/dev/null || true
pkill -9 -f "start_proposer_services\.sh" 2>/dev/null || true
pkill -9 -f "start_solver_services\.sh" 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 -f "verl\.trainer\.main" 2>/dev/null || true
pkill -9 -f "proposal_generate\.py" 2>/dev/null || true
pkill -9 -f "code_generate\.py" 2>/dev/null || true
pkill -9 -f "evaluate_imagefree\." 2>/dev/null || true
pkill -9 -f "bash.*main_svg\.sh" 2>/dev/null || true
pkill -9 -f "bash.*main\.sh" 2>/dev/null || true
pkill -9 -f "bash.*proposer_train\.sh" 2>/dev/null || true
pkill -9 -f "bash.*codegen_train\.sh" 2>/dev/null || true
pkill -9 -f "bash.*solver_train\.sh" 2>/dev/null || true

# 3) Kill everything still using a GPU
echo "[kill_training] Killing all processes using GPU..."
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
done
sleep 3
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
done

# 4) GPU device files — skipped (fuser -k on /dev/nvidia* can kill IDE/system processes)

# 5) Stop Ray (cleans up Ray workers)
ray stop --force 2>/dev/null || true

echo "[kill_training] Waiting for GPU memory to be released..."
sleep 5
echo "[kill_training] Done. Run 'nvidia-smi' to confirm GPUs are free, then start training again."
