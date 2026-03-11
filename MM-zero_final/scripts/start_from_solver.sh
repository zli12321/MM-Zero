#!/bin/bash
# =============================================================================
# ImageFree Self-Play: Start from Solver Training (EasyR1)
# =============================================================================
# Same as main_from_codegen.sh but SKIPS both Proposer v1 and CodeGen v1.
# Starts directly with Solver v1 training, then continues with iterations 2-3.
# Use this to test the Solver pipeline after CodeGen has finished training.
#
# Prerequisites:
#   - Proposer v1 must already be trained:
#     STORAGE_PATH/models/${Model_abbr}_proposer_v1/global_step_${TRAIN_STEPS}/actor/huggingface
#   - CodeGen v1 must already be trained:
#     STORAGE_PATH/models/${Model_abbr}_codegen_v1/global_step_${TRAIN_STEPS}/actor/huggingface
#
# Usage: bash MM-zero_final/scripts/start_from_solver.sh
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
# Training steps per model (Proposer, CodeGen, Solver). Must match the steps used
# when Proposer v1 and CodeGen v1 were trained.
# -----------------------------------------------------------------------------
export TRAIN_STEPS="${TRAIN_STEPS:-1}"
echo "TRAIN_STEPS=$TRAIN_STEPS (must match existing Proposer/CodeGen checkpoints)"

# -----------------------------------------------------------------------------
# Render workers: parallel matplotlib renders during Solver data pipeline.
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
export SAVE_RENDER_EXAMPLES="${SAVE_RENDER_EXAMPLES:-1}"
export SAVE_RENDER_EXAMPLES_N="${SAVE_RENDER_EXAMPLES_N:-10}"
export HUGGINGFACENAME=""

Base_model="${Base_model:-/workspace/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
Model_abbr=Qwen2.5-VL-7B-Instruct-ImageFree
echo "Model_abbr: $Model_abbr"
echo "Base_model: $Base_model"
echo "STORAGE_PATH: $STORAGE_PATH"

mkdir -p "$STORAGE_PATH/evaluation" \
         "$STORAGE_PATH/models" \
         "$STORAGE_PATH/generated_proposals" \
         "$STORAGE_PATH/generated_code" \
         "$STORAGE_PATH/rendered_images" \
         "$STORAGE_PATH/local_parquet" \
         "$STORAGE_PATH/temp_results"

# =============================================================================
# Iteration 1: Start from Solver only (Proposer v1 and CodeGen v1 already trained)
# =============================================================================
echo "=========================================="
echo "Starting Iteration 1 (from Solver only)"
echo "=========================================="

# Verify Proposer v1 exists
PROPOSER_V1_PATH="${STORAGE_PATH}/models/${Model_abbr}_proposer_v1/global_step_${TRAIN_STEPS}/actor/huggingface"
if [ ! -d "$PROPOSER_V1_PATH" ]; then
    echo "ERROR: Proposer v1 not found at $PROPOSER_V1_PATH"
    echo "Please train Proposer v1 first, or check TRAIN_STEPS (currently $TRAIN_STEPS)."
    exit 1
fi
echo "[Iter 1] Using existing Proposer v1: $PROPOSER_V1_PATH"

# Verify CodeGen v1 exists
CODEGEN_V1_PATH="${STORAGE_PATH}/models/${Model_abbr}_codegen_v1/global_step_${TRAIN_STEPS}/actor/huggingface"
if [ ! -d "$CODEGEN_V1_PATH" ]; then
    echo "ERROR: CodeGen v1 not found at $CODEGEN_V1_PATH"
    echo "Please train CodeGen v1 first, or check TRAIN_STEPS (currently $TRAIN_STEPS)."
    exit 1
fi
echo "[Iter 1] Using existing CodeGen v1: $CODEGEN_V1_PATH"

# Train Solver v1 (Solver init = base, Proposer v1 for questions, CodeGen v1 for images)
echo "[Iter 1] Training Solver v1..."
bash MM-zero_final/scripts/solver_train.sh \
    $Base_model \
    "$PROPOSER_V1_PATH" \
    "$CODEGEN_V1_PATH" \
    ${Model_abbr}_solver_v1

sleep 10

# =============================================================================
# Iterations 2+: Each model evolves from its previous version
# =============================================================================
for i in {2..3}; do
    prev=$((i-1))
    echo "=========================================="
    echo "Starting Iteration $i"
    echo "=========================================="

    # --- Proposer v_i ---
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

    sleep 10

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

    sleep 10

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

    sleep 10
done

echo "=========================================="
echo "ImageFree Self-Play (from Solver) complete!"
echo "=========================================="
