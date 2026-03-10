#!/bin/bash
# =============================================================================
# ImageFree Self-Play: Main Orchestrator (EasyR1)
# =============================================================================
# Runs the full 3-model self-play loop for 3 iterations:
#   Iteration 1: Proposer v1 → CodeGen v1 → Solver v1
#   Iteration 2: Proposer v2 → CodeGen v2 → Solver v2
#   Iteration 3: Proposer v3 → CodeGen v3 → Solver v3
#
# All three models start from the same base VLM and co-evolve.
#
# Proposer training uses BOTH CodeGen and Solver as reward services:
#   proposer_train.sh <solver_model> <proposer_model> <codegen_model> <save_name>
#
# Usage: bash SelfAgent/scripts/main.sh
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
export TRAIN_STEPS="${TRAIN_STEPS:-5}"
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
    # Cap at 24 to limit CPU RAM usage (each render worker holds matplotlib state in memory)
    [ "$RENDER_MAX_WORKERS" -gt 24 ] && RENDER_MAX_WORKERS=24
    echo "RENDER_MAX_WORKERS=$RENDER_MAX_WORKERS (auto: min(half of $NCPU CPUs, 24))"
else
    echo "RENDER_MAX_WORKERS=$RENDER_MAX_WORKERS (from env)"
fi
export RENDER_MAX_WORKERS

## round 5 not cross step caption validation
export STORAGE_PATH="${STORAGE_PATH:-/workspace/selfAgent_Storage_qwen3vl_8b_round6_reward_optimization}"
# Proposals per GPU for CodeGen data generation (fewer = faster, more = more training data)
export NUM_PROPOSALS_PER_GPU="${NUM_PROPOSALS_PER_GPU:-1024}"
# Solver needs more proposals because many are lost to render failures + difficulty filtering;
# with ~3% yield, 1000/GPU × 8 GPUs ≈ 8000 proposals → ~240 filtered training samples
export NUM_PROPOSALS_PER_GPU_SOLVER="${NUM_PROPOSALS_PER_GPU_SOLVER:-1000}"
# Proposer-only batch sizes (reduced to avoid OOM during actor update; was 64/32)
# NOT exported globally — only used in proposer_train.sh via PROPOSER_ROLLOUT_BATCH_SIZE
export PROPOSER_ROLLOUT_BATCH_SIZE="${PROPOSER_ROLLOUT_BATCH_SIZE:-32}"
export GLOBAL_BATCH_SIZE_PROPOSER="${GLOBAL_BATCH_SIZE_PROPOSER:-16}"
# Solver batch sizes (config default is 512/128 which requires >512 training samples;
# set lower to handle small filtered datasets from the difficulty filter)
export ROLLOUT_BATCH_SIZE_SOLVER="${ROLLOUT_BATCH_SIZE_SOLVER:-64}"
export GLOBAL_BATCH_SIZE_SOLVER="${GLOBAL_BATCH_SIZE_SOLVER:-64}"
# Raise Ray's OOM kill threshold from 0.95 to 0.98 (proposer offloading + Solver vLLM use ~95% of system RAM)
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.98}"
# Save a sample of rendered images per reward step to STORAGE_PATH/rendered_images/examples/step_N (set to 1 to enable)
export CODEGEN_HTTP_TIMEOUT="${CODEGEN_HTTP_TIMEOUT:-900}"
export SAVE_RENDER_EXAMPLES="${SAVE_RENDER_EXAMPLES:-1}"
export SAVE_RENDER_EXAMPLES_N="${SAVE_RENDER_EXAMPLES_N:-10}"
export HUGGINGFACENAME=""

# export HF_HOME=/workspace/.cache/huggingface 
# Base_model="${Base_model:-/workspace/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
Base_model="${Base_model:-/workspace/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"
Model_abbr=Qwen3-VL-8B-Instruct-ImageFree
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
# GPU cleanup between phases: kill vLLM/training processes so next phase can use GPUs.
# Call this after every phase (Proposer, CodeGen, Solver) before starting the next.
# -----------------------------------------------------------------------------
cleanup_gpu_for_next_phase() {
    echo "[main.sh] Cleaning up GPU before next phase..."
    # Kill by PID files (from proposer/codegen vLLM services)
    for PIDS_FILE in "${STORAGE_PATH}/temp_results/proposer_service_pids.env" "${STORAGE_PATH}/temp_results/solver_service_pids.env"; do
        if [ -f "$PIDS_FILE" ]; then
            set -a
            # shellcheck source=/dev/null
            source "$PIDS_FILE" 2>/dev/null || true
            set +a
            for pid in ${CODEGEN_PID_0:-} ${CODEGEN_PID_1:-} ${SOLVER_PID_0:-} ${SOLVER_PID_1:-} ${SOLVER_PID_2:-} ${SOLVER_PID_3:-}; do
                [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
            done
            rm -f "$PIDS_FILE"
        fi
    done
    # Kill vLLM and service launcher processes (lowercase and uppercase to catch VLLM::EngineCore children)
    pkill -9 -f "vllm" 2>/dev/null || true
    pkill -9 -f "VLLM" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "start_vllm_server" 2>/dev/null || true
    pkill -9 -f "start_codegen_server" 2>/dev/null || true
    pkill -9 -f "start_proposer_services" 2>/dev/null || true
    pkill -9 -f "start_solver_services" 2>/dev/null || true
    # Kill Ray (GRPO training)
    pkill -9 -f "ray" 2>/dev/null || true
    # Kill training-related Python processes that may still be shutting down
    pkill -9 -f "verl.trainer.main" 2>/dev/null || true
    pkill -9 -f "proposal_generate.py" 2>/dev/null || true
    pkill -9 -f "code_generate.py" 2>/dev/null || true
    pkill -9 -f "evaluate_imagefree.py" 2>/dev/null || true
    # Kill ALL processes currently using any GPU (catches VLLM::EngineCore and orphaned children)
    echo "[main.sh] Killing all processes using GPU (nvidia-smi)..."
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
    echo "[main.sh] Waiting for GPU memory to be released..."
    sleep 10
    python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(i)
        print(f'[main.sh] GPU cache cleared on {torch.cuda.device_count()} device(s)')
    else:
        print('[main.sh] CUDA not available, skipping cache clear')
except Exception as e:
    print(f'[main.sh] GPU cache clear skipped: {e}')
" 2>/dev/null || true
    sleep 3
    echo "[main.sh] GPU cleanup done."
}

# =============================================================================
# Iteration 1: All three models start from the base model
# =============================================================================
echo "=========================================="
echo "Starting Iteration 1"
echo "=========================================="

# Train Proposer v1
# Args: <solver_model> <proposer_model> <codegen_model> <save_name>
# In iteration 1, all three roles use the base model
if [ -d "${STORAGE_PATH}/models/${Model_abbr}_proposer_v1/global_step_${TRAIN_STEPS}/actor/huggingface" ]; then
    echo "[Iter 1] Proposer v1 already exists, skipping..."
else
    echo "[Iter 1] Training Proposer v1..."
    bash SelfAgent/scripts/proposer_train.sh \
        $Base_model \
        $Base_model \
        $Base_model \
        ${Model_abbr}_proposer_v1
fi

cleanup_gpu_for_next_phase
sleep 5

# Train CodeGen v1 (CodeGen init = base model, uses Proposer v1 for proposals)
if [ -d "${STORAGE_PATH}/models/${Model_abbr}_codegen_v1/global_step_${TRAIN_STEPS}/actor/huggingface" ]; then
    echo "[Iter 1] CodeGen v1 already exists, skipping..."
else
    echo "[Iter 1] Training CodeGen v1..."
    bash SelfAgent/scripts/codegen_train.sh \
        $Base_model \
        ${STORAGE_PATH}/models/${Model_abbr}_proposer_v1/global_step_${TRAIN_STEPS}/actor/huggingface \
        ${Model_abbr}_codegen_v1
fi

cleanup_gpu_for_next_phase
sleep 5

# Train Solver v1 (Solver init = base, Proposer v1 for questions, CodeGen v1 for images)
if [ -d "${STORAGE_PATH}/models/${Model_abbr}_solver_v1/global_step_${TRAIN_STEPS}/actor/huggingface" ]; then
    echo "[Iter 1] Solver v1 already exists, skipping..."
else
    echo "[Iter 1] Training Solver v1..."
    bash SelfAgent/scripts/solver_train.sh \
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
for i in {2..6}; do
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
        bash SelfAgent/scripts/proposer_train.sh \
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
        bash SelfAgent/scripts/codegen_train.sh \
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
        bash SelfAgent/scripts/solver_train.sh \
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
