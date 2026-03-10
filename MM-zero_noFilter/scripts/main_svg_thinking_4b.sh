#!/bin/bash
# =============================================================================
# ImageFree Self-Play: Main Orchestrator — SVG-Only Version
# =============================================================================
# Runs the full 3-model self-play loop for 3 iterations:
#   Iteration 1: Proposer v1 → CodeGen v1 → Solver v1
#   Iteration 2: Proposer v2 → CodeGen v2 → Solver v2
#   Iteration 3: Proposer v3 → CodeGen v3 → Solver v3
#
# GPU Layout: 2 GPUs for training, 6 GPUs for vLLM inference (Proposer/CodeGen).
#             Solver GRPO uses all 8 GPUs.
# Rendering:  SVG-only (cairosvg → PNG). No matplotlib/plotly/pillow.
#
# Usage: bash SelfAgent_svg/scripts/main_svg.sh
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# GPU memory: set to 40 or 80 (GB per GPU).
# -----------------------------------------------------------------------------
export GPU_MEM="${GPU_MEM:-80}"
if [ "$GPU_MEM" != "40" ] && [ "$GPU_MEM" != "80" ]; then
    echo "ERROR: GPU_MEM must be 40 or 80 (got: $GPU_MEM)."
    exit 1
fi
echo "Using GPU memory tier: ${GPU_MEM}GB"

# export WANDB_MODE=offline

# Training steps per model
export TRAIN_STEPS="${TRAIN_STEPS:-20}"
echo "TRAIN_STEPS=$TRAIN_STEPS"

# Render workers for SVG→PNG (CPU-bound cairosvg; lighter than matplotlib)
if [ -z "${RENDER_MAX_WORKERS}" ]; then
    NCPU=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo "16")
    RENDER_MAX_WORKERS=$(( NCPU / 2 ))
    [ "$RENDER_MAX_WORKERS" -lt 8 ] && RENDER_MAX_WORKERS=8
    [ "$RENDER_MAX_WORKERS" -gt 32 ] && RENDER_MAX_WORKERS=32
    echo "RENDER_MAX_WORKERS=$RENDER_MAX_WORKERS (auto)"
else
    echo "RENDER_MAX_WORKERS=$RENDER_MAX_WORKERS (from env)"
fi
export RENDER_MAX_WORKERS

export STORAGE_PATH="${STORAGE_PATH:-/workspace/selfAgent_Storage_svg_long_thinking_4b_round1_filter}"
export NUM_PROPOSALS_PER_GPU="${NUM_PROPOSALS_PER_GPU:-413}"
export NUM_PROPOSALS_PER_GPU_SOLVER="${NUM_PROPOSALS_PER_GPU_SOLVER:-625}"
# Proposer: 3 GPUs for training; batch sizes must be divisible by 3
export PROPOSER_ROLLOUT_BATCH_SIZE="${PROPOSER_ROLLOUT_BATCH_SIZE:-18}"
export GLOBAL_BATCH_SIZE_PROPOSER="${GLOBAL_BATCH_SIZE_PROPOSER:-18}"
# CodeGen rollouts per proposal. 4 = faster steps; 8 = more coverage. Affects proposer and codegen training.
export ROLLOUT_N="${ROLLOUT_N:-8}"
# Solver rollouts per question during proposer/codegen reward. 5 = faster; solver_train.sh overrides to 8.
export SOLVER_N_ROLLOUTS="${SOLVER_N_ROLLOUTS:-5}"
# Solver batch sizes
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-320}"
export ROLLOUT_BATCH_SIZE_SOLVER="${ROLLOUT_BATCH_SIZE_SOLVER:-512}"
export GLOBAL_BATCH_SIZE_SOLVER="${GLOBAL_BATCH_SIZE_SOLVER:-64}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.98}"
export CODEGEN_HTTP_TIMEOUT="${CODEGEN_HTTP_TIMEOUT:-900}"
export SAVE_RENDER_EXAMPLES="${SAVE_RENDER_EXAMPLES:-1}"
export SAVE_RENDER_EXAMPLES_N="${SAVE_RENDER_EXAMPLES_N:-10}"
export HUGGINGFACENAME=""

# XiaomiMiMo/MiMo-VL-7B-SFT-2508
export HF_HOME=/workspace/
Base_model="${Base_model:-Qwen/Qwen3-VL-4B-Thinking}"
Model_abbr=Qwen3-VL-4B-thinking-ImageFree-SVG
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

# GPU cleanup between phases
cleanup_gpu_for_next_phase() {
    echo "[main_svg] Cleaning up GPU before next phase..."
    for PIDS_FILE in "${STORAGE_PATH}/temp_results/proposer_service_pids.env" "${STORAGE_PATH}/temp_results/solver_service_pids.env"; do
        if [ -f "$PIDS_FILE" ]; then
            set -a
            source "$PIDS_FILE" 2>/dev/null || true
            set +a
            for pid in ${CODEGEN_PID_0:-} ${CODEGEN_PID_1:-} ${SOLVER_PID_0:-} ${SOLVER_PID_1:-} ${SOLVER_PID_2:-} ${SOLVER_PID_3:-} ${SOLVER_PID_4:-} ${SOLVER_PID_5:-}; do
                [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
            done
            rm -f "$PIDS_FILE"
        fi
    done
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
    echo "[main_svg] Killing GPU compute processes (nvidia-smi)..."
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    sleep 5
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    echo "[main_svg] Waiting for GPU memory and ports to be released..."
    sleep 15
    python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(i)
        print(f'[main_svg] GPU cache cleared on {torch.cuda.device_count()} device(s)')
except Exception as e:
    print(f'[main_svg] GPU cache clear skipped: {e}')
" 2>/dev/null || true
    sleep 3
    echo "[main_svg] GPU cleanup done."
}

# =============================================================================
# Iteration 1: All three models start from the base model
# =============================================================================
echo "=========================================="
echo "Starting Iteration 1 (SVG-only)"
echo "=========================================="

# Train Proposer v1
if [ -d "${STORAGE_PATH}/models/${Model_abbr}_proposer_v1/global_step_${TRAIN_STEPS}/actor/huggingface" ]; then
    echo "[Iter 1] Proposer v1 already exists, skipping..."
else
    echo "[Iter 1] Training Proposer v1..."
    bash SelfAgent_svg/scripts/proposer_train.sh \
        $Base_model \
        $Base_model \
        $Base_model \
        ${Model_abbr}_proposer_v1
fi

cleanup_gpu_for_next_phase
sleep 5

# Train CodeGen v1
export CODEGEN_EXAMPLE_STEP_OFFSET=0
if [ -d "${STORAGE_PATH}/models/${Model_abbr}_codegen_v1/global_step_${TRAIN_STEPS}/actor/huggingface" ]; then
    echo "[Iter 1] CodeGen v1 already exists, skipping..."
else
    echo "[Iter 1] Training CodeGen v1..."
    bash SelfAgent_svg/scripts/codegen_train.sh \
        $Base_model \
        ${STORAGE_PATH}/models/${Model_abbr}_proposer_v1/global_step_${TRAIN_STEPS}/actor/huggingface \
        ${Model_abbr}_codegen_v1
fi

cleanup_gpu_for_next_phase
sleep 5

# Train Solver v1
if [ -d "${STORAGE_PATH}/models/${Model_abbr}_solver_v1/global_step_${TRAIN_STEPS}/actor/huggingface" ]; then
    echo "[Iter 1] Solver v1 already exists, skipping..."
else
    echo "[Iter 1] Training Solver v1..."
    bash SelfAgent_svg/scripts/solver_train.sh \
        $Base_model \
        ${STORAGE_PATH}/models/${Model_abbr}_proposer_v1/global_step_${TRAIN_STEPS}/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_codegen_v1/global_step_${TRAIN_STEPS}/actor/huggingface \
        ${Model_abbr}_solver_v1
fi

cleanup_gpu_for_next_phase
sleep 5

# =============================================================================
# Iterations 2+: Each model evolves from its previous version
# =============================================================================
for i in {2..12}; do
    prev=$((i-1))
    echo "=========================================="
    echo "Starting Iteration $i (SVG-only)"
    echo "=========================================="

    # --- Proposer v_i ---
    if [ -d "${STORAGE_PATH}/models/${Model_abbr}_proposer_v${i}/global_step_${TRAIN_STEPS}/actor/huggingface" ]; then
        echo "[Iter $i] Proposer v${i} already exists, skipping..."
    else
        echo "[Iter $i] Training Proposer v${i}..."
        bash SelfAgent_svg/scripts/proposer_train.sh \
            ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_${TRAIN_STEPS}/actor/huggingface \
            ${STORAGE_PATH}/models/${Model_abbr}_proposer_v${prev}/global_step_${TRAIN_STEPS}/actor/huggingface \
            ${STORAGE_PATH}/models/${Model_abbr}_codegen_v${prev}/global_step_${TRAIN_STEPS}/actor/huggingface \
            ${Model_abbr}_proposer_v${i}
    fi

    cleanup_gpu_for_next_phase
    sleep 5

    # --- CodeGen v_i ---
    export CODEGEN_EXAMPLE_STEP_OFFSET=$(( (i - 1) * TRAIN_STEPS ))
    if [ -d "${STORAGE_PATH}/models/${Model_abbr}_codegen_v${i}/global_step_${TRAIN_STEPS}/actor/huggingface" ]; then
        echo "[Iter $i] CodeGen v${i} already exists, skipping..."
    else
        echo "[Iter $i] Training CodeGen v${i}..."
        bash SelfAgent_svg/scripts/codegen_train.sh \
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
        bash SelfAgent_svg/scripts/solver_train.sh \
            ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_${TRAIN_STEPS}/actor/huggingface \
            ${STORAGE_PATH}/models/${Model_abbr}_proposer_v${i}/global_step_${TRAIN_STEPS}/actor/huggingface \
            ${STORAGE_PATH}/models/${Model_abbr}_codegen_v${i}/global_step_${TRAIN_STEPS}/actor/huggingface \
            ${Model_abbr}_solver_v${i}
    fi

    cleanup_gpu_for_next_phase
    sleep 5
done

echo "=========================================="
echo "ImageFree Self-Play (SVG) training complete!"
echo "=========================================="
