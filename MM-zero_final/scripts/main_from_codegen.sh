#!/bin/bash
# =============================================================================
# ImageFree Self-Play: Start from CodeGen Training (EasyR1)
# =============================================================================
# Same as main.sh but SKIPS Proposer v1 training (assumes it's already done).
# Starts from CodeGen v1 training, then continues with Solver v1 and iterations 2-3.
#
# Prerequisites:
#   - Proposer v1 must already be trained and saved at:
#     STORAGE_PATH/models/${Model_abbr}_proposer_v1/global_step_${TRAIN_STEPS}/actor/huggingface
#
# Usage: bash MM-zero_final/scripts/main_from_codegen.sh
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# GPU memory: set to 40 or 80 (GB per GPU). Chooses configs and vLLM max_model_len.
# -----------------------------------------------------------------------------
export GPU_MEM="${GPU_MEM:-80}"
if [ "$GPU_MEM" != "40" ] && [ "$GPU_MEM" != "80" ]; then
    echo "ERROR: GPU_MEM must be 40 or 80 (got: $GPU_MEM). Export GPU_MEM=40 or GPU_MEM=80"
    exit 1
fi
echo "Using GPU memory tier: ${GPU_MEM}GB (configs: *_config_${GPU_MEM}gb.yaml)"

# export WANDB_MODE=offline
# export WANDB_DIR=yourpath/wandb

# -----------------------------------------------------------------------------
# Training steps per model (Proposer, CodeGen, Solver). Fewer steps = switch to next model sooner.
# Override with TRAIN_STEPS=... (e.g. export TRAIN_STEPS=5 for quick runs).
# -----------------------------------------------------------------------------
export TRAIN_STEPS="${TRAIN_STEPS:-1}"
echo "TRAIN_STEPS=$TRAIN_STEPS (each of Proposer, CodeGen, Solver runs this many steps per iteration)"

# -----------------------------------------------------------------------------
# Render workers: parallel matplotlib renders during Proposer reward (CPU-bound).
# Default = half of logical CPUs (leaves headroom for training/CodeGen/Solver).
# Override with RENDER_MAX_WORKERS=... (e.g. export RENDER_MAX_WORKERS=96).
# -----------------------------------------------------------------------------
if [ -z "${RENDER_MAX_WORKERS}" ]; then
    NCPU=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo "16")
    RENDER_MAX_WORKERS=$(( NCPU / 2 ))
    [ "$RENDER_MAX_WORKERS" -lt 8 ] && RENDER_MAX_WORKERS=8
    [ "$RENDER_MAX_WORKERS" -gt 24 ] && RENDER_MAX_WORKERS=24
    echo "RENDER_MAX_WORKERS=$RENDER_MAX_WORKERS (auto: min(half of $NCPU CPUs, 24))"
else
    echo "RENDER_MAX_WORKERS=$RENDER_MAX_WORKERS (from env)"
fi
export RENDER_MAX_WORKERS

export STORAGE_PATH="${STORAGE_PATH:-/workspace/selfAgent_Storage}"
# Save a sample of rendered images per reward step to STORAGE_PATH/rendered_images/examples/step_N (set to 1 to enable)
export SAVE_RENDER_EXAMPLES="${SAVE_RENDER_EXAMPLES:-1}"
export SAVE_RENDER_EXAMPLES_N="${SAVE_RENDER_EXAMPLES_N:-10}"
export HUGGINGFACENAME=""

# export HF_HOME=/workspace/.cache/huggingface
Base_model="${Base_model:-/workspace/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
Model_abbr=Qwen2.5-VL-7B-Instruct-ImageFree
echo "Model_abbr: $Model_abbr"
echo "Base_model: $Base_model"
echo "STORAGE_PATH: $STORAGE_PATH"
[ "$SAVE_RENDER_EXAMPLES" = "1" ] && echo "SAVE_RENDER_EXAMPLES=1: example images will be saved to $STORAGE_PATH/rendered_images/examples/step_N"

mkdir -p "$STORAGE_PATH/evaluation" \
         "$STORAGE_PATH/models" \
         "$STORAGE_PATH/generated_proposals" \
         "$STORAGE_PATH/generated_code" \
         "$STORAGE_PATH/rendered_images" \
         "$STORAGE_PATH/local_parquet" \
         "$STORAGE_PATH/temp_results"

# -----------------------------------------------------------------------------
# GPU cleanup between phases: kill all GPU processes so next phase starts clean.
# -----------------------------------------------------------------------------
cleanup_gpu_for_next_phase() {
    echo "[main_from_codegen] Cleaning up GPU before next phase..."
    for PIDS_FILE in "${STORAGE_PATH}/temp_results/proposer_service_pids.env" "${STORAGE_PATH}/temp_results/solver_service_pids.env"; do
        if [ -f "$PIDS_FILE" ]; then
            set -a; source "$PIDS_FILE" 2>/dev/null || true; set +a
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
    echo "[main_from_codegen] Killing all processes using GPU (nvidia-smi)..."
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    sleep 5
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    # Final safety net: kill anything still holding GPU device files open
    for dev in /dev/nvidia[0-9]*; do
        [ -e "$dev" ] && fuser -k -9 "$dev" 2>/dev/null || true
    done
    sleep 10
    python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(i)
except Exception:
    pass
" 2>/dev/null || true
    sleep 3
    echo "[main_from_codegen] GPU cleanup done."
}

# =============================================================================
# Iteration 1: Start from CodeGen (Proposer v1 already trained)
# =============================================================================
echo "=========================================="
echo "Starting Iteration 1 (from CodeGen)"
echo "=========================================="

# Verify Proposer v1 exists
PROPOSER_V1_PATH="${STORAGE_PATH}/models/${Model_abbr}_proposer_v1/global_step_${TRAIN_STEPS}/actor/huggingface"
if [ ! -d "$PROPOSER_V1_PATH" ]; then
    echo "ERROR: Proposer v1 not found at $PROPOSER_V1_PATH"
    echo "Please train Proposer v1 first, or check TRAIN_STEPS (currently $TRAIN_STEPS)."
    exit 1
fi
echo "[Iter 1] Using existing Proposer v1: $PROPOSER_V1_PATH"

# Train CodeGen v1 (CodeGen init = base model, uses Proposer v1 for proposals)
echo "[Iter 1] Training CodeGen v1..."
bash MM-zero_final/scripts/codegen_train.sh \
    $Base_model \
    ${STORAGE_PATH}/models/${Model_abbr}_proposer_v1/global_step_${TRAIN_STEPS}/actor/huggingface \
    ${Model_abbr}_codegen_v1

cleanup_gpu_for_next_phase
sleep 5

# Train Solver v1 (Solver init = base, Proposer v1 for questions, CodeGen v1 for images)
echo "[Iter 1] Training Solver v1..."
bash MM-zero_final/scripts/solver_train.sh \
    $Base_model \
    ${STORAGE_PATH}/models/${Model_abbr}_proposer_v1/global_step_${TRAIN_STEPS}/actor/huggingface \
    ${STORAGE_PATH}/models/${Model_abbr}_codegen_v1/global_step_${TRAIN_STEPS}/actor/huggingface \
    ${Model_abbr}_solver_v1

cleanup_gpu_for_next_phase
sleep 5

# =============================================================================
# Iterations 2+: Each model evolves from its previous version
# =============================================================================
for i in {2..3}; do
    prev=$((i-1))
    echo "=========================================="
    echo "Starting Iteration $i"
    echo "=========================================="

    # --- Proposer v_i ---
    # Reward uses: Solver v_{i-1} + CodeGen v_{i-1}
    # Init from: Proposer v_{i-1}
    if [ -d "${STORAGE_PATH}/models/${Model_abbr}_proposer_v${i}/global_step_${TRAIN_STEPS}/actor/huggingface" ]; then
        echo "[Iter $i] Proposer v${i} already exists, skipping..."
    else
        echo "[Iter $i] Training Proposer v${i}..."
        bash MM-zero_final/scripts/proposer_train.sh \
            ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_${TRAIN_STEPS}/actor/huggingface \
            ${STORAGE_PATH}/models/${Model_abbr}_proposer_v${prev}/global_step_${TRAIN_STEPS}/actor/huggingface \
            ${STORAGE_PATH}/models/${Model_abbr}_codegen_v${prev}/global_step_${TRAIN_STEPS}/actor/huggingface \
            ${Model_abbr}_proposer_v${i}
    fi

    cleanup_gpu_for_next_phase
    sleep 5

    # --- CodeGen v_i ---
    if [ -d "${STORAGE_PATH}/models/${Model_abbr}_codegen_v${i}/global_step_${TRAIN_STEPS}/actor/huggingface" ]; then
        echo "[Iter $i] CodeGen v${i} already exists, skipping..."
    else
        echo "[Iter $i] Training CodeGen v${i}..."
        bash MM-zero_final/scripts/codegen_train.sh \
            ${STORAGE_PATH}/models/${Model_abbr}_codegen_v${prev}/global_step_${TRAIN_STEPS}/actor/huggingface \
            ${STORAGE_PATH}/models/${Model_abbr}_proposer_v${i}/global_step_${TRAIN_STEPS}/actor/huggingface \
            ${Model_abbr}_codegen_v${i}
    fi

    cleanup_gpu_for_next_phase
    sleep 5

    # --- Solver v_i ---
    if [ -d "${STORAGE_PATH}/models/${Model_abbr}_solver_v${i}/global_step_${TRAIN_STEPS}/actor/huggingface" ]; then
        echo "[Iter $i] Solver v${i} already exists, skipping..."
    else
        echo "[Iter $i] Training Solver v${i}..."
        bash MM-zero_final/scripts/solver_train.sh \
            ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_${TRAIN_STEPS}/actor/huggingface \
            ${STORAGE_PATH}/models/${Model_abbr}_proposer_v${i}/global_step_${TRAIN_STEPS}/actor/huggingface \
            ${STORAGE_PATH}/models/${Model_abbr}_codegen_v${i}/global_step_${TRAIN_STEPS}/actor/huggingface \
            ${Model_abbr}_solver_v${i}
    fi

    cleanup_gpu_for_next_phase
    sleep 5
done

echo "=========================================="
echo "ImageFree Self-Play training complete!"
echo "=========================================="
