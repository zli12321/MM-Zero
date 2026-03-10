#!/bin/bash
# =============================================================================
# ImageFree Self-Play: MINI Orchestrator (Fast Debug Run)
# =============================================================================
# Identical to main.sh but with drastically reduced batch sizes for fast
# end-to-end pipeline testing. Runs only 1 iteration instead of 3.
#
# Full run (main.sh):
#   Proposer: rollout_batch_size=64, n=8    → 512 total rollouts
#   CodeGen:  rollout_batch_size=256, n=8   → 2048 total, 500 proposals/GPU
#   Solver:   rollout_batch_size=512, n=8   → 4096 total, 512 proposals/GPU
#   Iterations: 3
#
# Mini run (this script):
#   Proposer: rollout_batch_size=8, n=2     → 16 total rollouts
#   CodeGen:  rollout_batch_size=16, n=2    → 32 total, 16 proposals/GPU
#   Solver:   rollout_batch_size=16, n=2    → 32 total, 16 proposals/GPU
#   Iterations: 1
#
# Usage: bash SelfAgent/scripts/main_mini.sh
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# GPU memory tier (same as main.sh)
# -----------------------------------------------------------------------------
export GPU_MEM="${GPU_MEM:-80}"
if [ "$GPU_MEM" != "40" ] && [ "$GPU_MEM" != "80" ]; then
    echo "ERROR: GPU_MEM must be 40 or 80 (got: $GPU_MEM). Export GPU_MEM=40 or GPU_MEM=80"
    exit 1
fi
echo "[MINI] Using GPU memory tier: ${GPU_MEM}GB"

# export WANDB_MODE=offline
# export WANDB_DIR=yourpath/wandb

# -----------------------------------------------------------------------------
# MINI overrides: small batches for fast debugging
# -----------------------------------------------------------------------------
export TRAIN_STEPS="${TRAIN_STEPS:-1}"

# Proposer: 8 proposals × 2 rollouts = 16 (was 64×8=512)
export ROLLOUT_BATCH_SIZE=8
export ROLLOUT_N=2
export GLOBAL_BATCH_SIZE_PROPOSER=8

# CodeGen: 16 samples × 2 rollouts = 32 (was 256×8=2048)
export GLOBAL_BATCH_SIZE_CODEGEN=16

# Solver: 16 samples × 2 rollouts = 32 (was 512×8=4096)
export ROLLOUT_BATCH_SIZE_SOLVER=16
export ROLLOUT_N_SOLVER=2
export GLOBAL_BATCH_SIZE_SOLVER=16

# Data generation: 16 proposals per GPU (was 500/512)
export NUM_PROPOSALS_PER_GPU=16

echo "[MINI] TRAIN_STEPS=$TRAIN_STEPS"
echo "[MINI] Proposer: rollout_batch_size=$ROLLOUT_BATCH_SIZE, rollout.n=$ROLLOUT_N, global_batch_size=$GLOBAL_BATCH_SIZE_PROPOSER"
echo "[MINI] CodeGen:  rollout_batch_size=$ROLLOUT_BATCH_SIZE, rollout.n=$ROLLOUT_N, global_batch_size=$GLOBAL_BATCH_SIZE_CODEGEN"
echo "[MINI] Solver:   rollout_batch_size=$ROLLOUT_BATCH_SIZE_SOLVER, rollout.n=$ROLLOUT_N_SOLVER, global_batch_size=$GLOBAL_BATCH_SIZE_SOLVER"
echo "[MINI] Proposals per GPU: $NUM_PROPOSALS_PER_GPU"

# -----------------------------------------------------------------------------
# Render workers
# -----------------------------------------------------------------------------
if [ -z "${RENDER_MAX_WORKERS}" ]; then
    NCPU=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo "16")
    RENDER_MAX_WORKERS=$(( NCPU / 2 ))
    [ "$RENDER_MAX_WORKERS" -lt 8 ] && RENDER_MAX_WORKERS=8
    [ "$RENDER_MAX_WORKERS" -gt 24 ] && RENDER_MAX_WORKERS=24
    echo "[MINI] RENDER_MAX_WORKERS=$RENDER_MAX_WORKERS (auto: min(half of $NCPU CPUs, 24))"
else
    echo "[MINI] RENDER_MAX_WORKERS=$RENDER_MAX_WORKERS (from env)"
fi
export RENDER_MAX_WORKERS

export STORAGE_PATH="${STORAGE_PATH:-/workspace/selfAgent_Storage_mini}"
export SAVE_RENDER_EXAMPLES="${SAVE_RENDER_EXAMPLES:-1}"
export SAVE_RENDER_EXAMPLES_N="${SAVE_RENDER_EXAMPLES_N:-5}"
export HUGGINGFACENAME=""

Base_model="${Base_model:-/workspace/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
Model_abbr=Qwen2.5-VL-7B-Instruct-ImageFree-MINI

echo "[MINI] Model_abbr: $Model_abbr"
echo "[MINI] Base_model: $Base_model"
echo "[MINI] STORAGE_PATH: $STORAGE_PATH"

mkdir -p "$STORAGE_PATH/evaluation" \
         "$STORAGE_PATH/models" \
         "$STORAGE_PATH/generated_proposals" \
         "$STORAGE_PATH/generated_code" \
         "$STORAGE_PATH/rendered_images" \
         "$STORAGE_PATH/local_parquet" \
         "$STORAGE_PATH/temp_results"

# -----------------------------------------------------------------------------
# GPU cleanup (same as main.sh)
# -----------------------------------------------------------------------------
cleanup_gpu_for_next_phase() {
    echo "[mini.sh] Cleaning up GPU before next phase..."
    for PIDS_FILE in "${STORAGE_PATH}/temp_results/proposer_service_pids.env" "${STORAGE_PATH}/temp_results/solver_service_pids.env"; do
        if [ -f "$PIDS_FILE" ]; then
            set -a
            source "$PIDS_FILE" 2>/dev/null || true
            set +a
            for pid in ${CODEGEN_PID_0:-} ${CODEGEN_PID_1:-} ${SOLVER_PID_0:-} ${SOLVER_PID_1:-} ${SOLVER_PID_2:-} ${SOLVER_PID_3:-}; do
                [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
            done
            rm -f "$PIDS_FILE"
        fi
    done
    pkill -9 -f "vllm" 2>/dev/null || true
    pkill -9 -f "VLLM" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "start_vllm_server" 2>/dev/null || true
    pkill -9 -f "start_codegen_server" 2>/dev/null || true
    pkill -9 -f "start_proposer_services" 2>/dev/null || true
    pkill -9 -f "start_solver_services" 2>/dev/null || true
    pkill -9 -f "ray" 2>/dev/null || true
    pkill -9 -f "verl.trainer.main" 2>/dev/null || true
    pkill -9 -f "proposal_generate.py" 2>/dev/null || true
    pkill -9 -f "code_generate.py" 2>/dev/null || true
    pkill -9 -f "evaluate_imagefree.py" 2>/dev/null || true
    echo "[mini.sh] Killing all processes using GPU (nvidia-smi)..."
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    sleep 5
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    for dev in /dev/nvidia[0-9]*; do
        [ -e "$dev" ] && fuser -k -9 "$dev" 2>/dev/null || true
    done
    echo "[mini.sh] Waiting for GPU memory to be released..."
    sleep 10
    python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(i)
        print(f'[mini.sh] GPU cache cleared on {torch.cuda.device_count()} device(s)')
    else:
        print('[mini.sh] CUDA not available, skipping cache clear')
except Exception as e:
    print(f'[mini.sh] GPU cache clear skipped: {e}')
" 2>/dev/null || true
    sleep 3
    echo "[mini.sh] GPU cleanup done."
}

# =============================================================================
# Iteration 1 only (mini mode runs just 1 iteration for fast debugging)
# =============================================================================
echo "=========================================="
echo "[MINI] Starting Iteration 1 (only iteration)"
echo "=========================================="

# Train Proposer v1
echo "[MINI][Iter 1] Training Proposer v1..."
bash SelfAgent/scripts/proposer_train.sh \
    $Base_model \
    $Base_model \
    $Base_model \
    ${Model_abbr}_proposer_v1

cleanup_gpu_for_next_phase
sleep 5

# Train CodeGen v1
echo "[MINI][Iter 1] Training CodeGen v1..."
bash SelfAgent/scripts/codegen_train.sh \
    $Base_model \
    ${STORAGE_PATH}/models/${Model_abbr}_proposer_v1/global_step_${TRAIN_STEPS}/actor/huggingface \
    ${Model_abbr}_codegen_v1

cleanup_gpu_for_next_phase
sleep 5

# Train Solver v1
echo "[MINI][Iter 1] Training Solver v1..."
bash SelfAgent/scripts/solver_train.sh \
    $Base_model \
    ${STORAGE_PATH}/models/${Model_abbr}_proposer_v1/global_step_${TRAIN_STEPS}/actor/huggingface \
    ${STORAGE_PATH}/models/${Model_abbr}_codegen_v1/global_step_${TRAIN_STEPS}/actor/huggingface \
    ${Model_abbr}_solver_v1

cleanup_gpu_for_next_phase

echo "=========================================="
echo "[MINI] ImageFree Self-Play MINI debug run complete!"
echo "=========================================="
