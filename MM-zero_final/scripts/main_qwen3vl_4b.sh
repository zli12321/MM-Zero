#!/bin/bash
# =============================================================================
# ImageFree Self-Play: Main Orchestrator — Base-Model Fallback Version
# =============================================================================
# Same as main.sh but NEVER errors on missing checkpoints.
# If a previous iteration's checkpoint is missing, it falls back to $Base_model.
# This makes it safe to run from scratch or with partial checkpoints.
#
# Usage: bash MM-zero_final/scripts/main_base.sh
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# GPU memory
# -----------------------------------------------------------------------------
export GPU_MEM="${GPU_MEM:-80}"
if [ "$GPU_MEM" != "40" ] && [ "$GPU_MEM" != "80" ]; then
    echo "ERROR: GPU_MEM must be 40 or 80 (got: $GPU_MEM). Export GPU_MEM=40 or GPU_MEM=80"
    exit 1
fi
echo "Using GPU memory tier: ${GPU_MEM}GB (configs: *_config_${GPU_MEM}gb.yaml)"

# export WANDB_MODE=offline
# export WANDB_DIR=yourpath/wandb

export TRAIN_STEPS="${TRAIN_STEPS:-20}"
echo "TRAIN_STEPS=$TRAIN_STEPS (each of Proposer, CodeGen, Solver runs this many steps per iteration)"

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

export STORAGE_PATH="${STORAGE_PATH:-/workspace/dummy_output_qwen3vl_4b}"
export NUM_PROPOSALS_PER_GPU="${NUM_PROPOSALS_PER_GPU:-450}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.98}"
export CODEGEN_HTTP_TIMEOUT="${CODEGEN_HTTP_TIMEOUT:-600}"
# Proposer trains on 3 GPUs; batch sizes must be divisible by 3 * micro_batch (3*4=12)
export PROPOSER_ROLLOUT_BATCH_SIZE="${PROPOSER_ROLLOUT_BATCH_SIZE:-24}"
export GLOBAL_BATCH_SIZE_PROPOSER="${GLOBAL_BATCH_SIZE_PROPOSER:-24}"
# Select 4B-optimized configs (no offloading, larger micro-batches, TP=1)
export MODEL_SIZE="4b"
export SAVE_RENDER_EXAMPLES="${SAVE_RENDER_EXAMPLES:-1}"
export SAVE_RENDER_EXAMPLES_N="${SAVE_RENDER_EXAMPLES_N:-10}"
export HUGGINGFACENAME=""

Base_model="${Base_model:-Qwen/Qwen3-VL-4B-Instruct}"
Model_abbr=Qwen3-VL-4B-Instruct-ImageFree
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
# Helper: resolve a model path, falling back to Base_model if checkpoint missing
# -----------------------------------------------------------------------------
resolve_model() {
    local path="$1"
    local label="$2"
    if [ -d "$path" ]; then
        echo "$path"
    else
        echo "[main_base] WARNING: $label not found at $path — using Base_model instead" >&2
        echo "$Base_model"
    fi
}

# -----------------------------------------------------------------------------
# GPU cleanup (same as main.sh)
# -----------------------------------------------------------------------------
cleanup_gpu_for_next_phase() {
    echo "[main_base] Cleaning up GPU before next phase..."
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
    echo "[main_base] Killing all processes using GPU (nvidia-smi)..."
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
    echo "[main_base] Waiting for GPU memory to be released..."
    sleep 10
    python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(i)
        print(f'[main_base] GPU cache cleared on {torch.cuda.device_count()} device(s)')
    else:
        print('[main_base] CUDA not available, skipping cache clear')
except Exception as e:
    print(f'[main_base] GPU cache clear skipped: {e}')
" 2>/dev/null || true
    sleep 3
    echo "[main_base] GPU cleanup done."
}

# =============================================================================
# All iterations in one loop (1 through N). Uses resolve_model() so missing
# checkpoints fall back to Base_model instead of erroring.
# =============================================================================
NUM_ITERATIONS="${NUM_ITERATIONS:-6}"
START_ITERATION="${START_ITERATION:-1}"
echo "NUM_ITERATIONS=$NUM_ITERATIONS, START_ITERATION=$START_ITERATION"

# Auto-detect: scan for the earliest incomplete iteration so we can skip ahead
if [ "$START_ITERATION" -eq 1 ]; then
    for _auto_i in $(seq 1 $NUM_ITERATIONS); do
        _p="${STORAGE_PATH}/models/${Model_abbr}_proposer_v${_auto_i}/global_step_${TRAIN_STEPS}/actor/huggingface"
        _c="${STORAGE_PATH}/models/${Model_abbr}_codegen_v${_auto_i}/global_step_${TRAIN_STEPS}/actor/huggingface"
        _s="${STORAGE_PATH}/models/${Model_abbr}_solver_v${_auto_i}/global_step_${TRAIN_STEPS}/actor/huggingface"
        if [ -d "$_p" ] && [ -d "$_c" ] && [ -d "$_s" ]; then
            continue
        else
            START_ITERATION=$_auto_i
            break
        fi
    done
    echo "[resume] Auto-detected: resuming from iteration $START_ITERATION"
fi

for i in $(seq $START_ITERATION $NUM_ITERATIONS); do
    echo "=========================================="
    echo "Starting Iteration $i / $NUM_ITERATIONS"
    echo "=========================================="

    # Resolve model paths: for iteration 1 use base model, for 2+ use previous checkpoint (or base if missing)
    prev=$((i-1))
    if [ "$i" -eq 1 ]; then
        PROPOSER_INIT="$Base_model"
        CODEGEN_INIT="$Base_model"
        SOLVER_INIT="$Base_model"
        SOLVER_FOR_REWARD="$Base_model"
        CODEGEN_FOR_REWARD="$Base_model"
    else
        PROPOSER_INIT=$(resolve_model "${STORAGE_PATH}/models/${Model_abbr}_proposer_v${prev}/global_step_${TRAIN_STEPS}/actor/huggingface" "Proposer v${prev}")
        CODEGEN_INIT=$(resolve_model "${STORAGE_PATH}/models/${Model_abbr}_codegen_v${prev}/global_step_${TRAIN_STEPS}/actor/huggingface" "CodeGen v${prev}")
        SOLVER_INIT="$Base_model"
        SOLVER_FOR_REWARD=$(resolve_model "${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_${TRAIN_STEPS}/actor/huggingface" "Solver v${prev}")
        CODEGEN_FOR_REWARD=$(resolve_model "${STORAGE_PATH}/models/${Model_abbr}_codegen_v${prev}/global_step_${TRAIN_STEPS}/actor/huggingface" "CodeGen v${prev}")
    fi

    # --- Proposer v_i ---
    PHASE_RAN=0
    if [ -d "${STORAGE_PATH}/models/${Model_abbr}_proposer_v${i}/global_step_${TRAIN_STEPS}/actor/huggingface" ]; then
        echo "[Iter $i] Proposer v${i} already exists, skipping..."
    else
        echo "[Iter $i] Training Proposer v${i}..."
        bash MM-zero_final/scripts/proposer_train.sh \
            "$SOLVER_FOR_REWARD" \
            "$PROPOSER_INIT" \
            "$CODEGEN_FOR_REWARD" \
            ${Model_abbr}_proposer_v${i}
        PHASE_RAN=1
    fi

    if [ "$PHASE_RAN" -eq 1 ]; then
        cleanup_gpu_for_next_phase
        sleep 5
    fi

    # Resolve proposer path for CodeGen/Solver (just trained, or fallback)
    PROPOSER_TRAINED=$(resolve_model "${STORAGE_PATH}/models/${Model_abbr}_proposer_v${i}/global_step_${TRAIN_STEPS}/actor/huggingface" "Proposer v${i}")

    # --- CodeGen v_i ---
    PHASE_RAN=0
    if [ -d "${STORAGE_PATH}/models/${Model_abbr}_codegen_v${i}/global_step_${TRAIN_STEPS}/actor/huggingface" ]; then
        echo "[Iter $i] CodeGen v${i} already exists, skipping..."
    else
        echo "[Iter $i] Training CodeGen v${i}..."
        bash MM-zero_final/scripts/codegen_train.sh \
            "$CODEGEN_INIT" \
            "$PROPOSER_TRAINED" \
            ${Model_abbr}_codegen_v${i}
        PHASE_RAN=1
    fi

    if [ "$PHASE_RAN" -eq 1 ]; then
        cleanup_gpu_for_next_phase
        sleep 5
    fi

    # Resolve codegen path for Solver (just trained, or fallback)
    CODEGEN_TRAINED=$(resolve_model "${STORAGE_PATH}/models/${Model_abbr}_codegen_v${i}/global_step_${TRAIN_STEPS}/actor/huggingface" "CodeGen v${i}")

    # --- Solver v_i ---
    PHASE_RAN=0
    if [ -d "${STORAGE_PATH}/models/${Model_abbr}_solver_v${i}/global_step_${TRAIN_STEPS}/actor/huggingface" ]; then
        echo "[Iter $i] Solver v${i} already exists, skipping..."
    else
        echo "[Iter $i] Training Solver v${i}..."
        bash MM-zero_final/scripts/solver_train.sh \
            "$SOLVER_INIT" \
            "$PROPOSER_TRAINED" \
            "$CODEGEN_TRAINED" \
            ${Model_abbr}_solver_v${i}
        PHASE_RAN=1
    fi

    if [ "$PHASE_RAN" -eq 1 ]; then
        cleanup_gpu_for_next_phase
        sleep 5
    fi
done

echo "=========================================="
echo "ImageFree Self-Play training complete!"
echo "=========================================="
