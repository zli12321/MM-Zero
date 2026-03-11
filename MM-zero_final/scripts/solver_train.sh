#!/bin/bash
# =============================================================================
# ImageFree Self-Play: Solver Training (EasyR1)
# =============================================================================
# The Solver takes rendered images + hard questions and learns to answer them.
#
# Pipeline:
#   1. Generate proposals from the Proposer (caption + easy Q + hard Q)
#   2. Generate matplotlib code from proposals using CodeGen
#   3. Compile/render code to produce images
#   4. Evaluate hard questions with the Solver to get silver labels
#   5. Filter by difficulty (score 0.3–0.8) and train Solver with GRPO
#
# Usage: bash MM-zero_final/scripts/solver_train.sh <solver_model> <proposer_model> <codegen_model> <save_name>
# =============================================================================

set -x

# -----------------------------------------------------------------------------
# Cleanup: Always free all GPUs and kill all processes on exit/error.
# This ensures clean state for next run even if training fails halfway.
# -----------------------------------------------------------------------------
cleanup_solver_training() {
    echo "[solver_train] CLEANUP: Freeing all GPUs and killing processes..."
    
    # Kill any vLLM processes (from proposal/code generation/evaluation)
    echo "[solver_train] Killing vLLM processes..."
    pkill -9 -f "python.*-m vllm" 2>/dev/null || true
    pkill -9 -f "vllm\.entrypoints" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "start_vllm_server\.py" 2>/dev/null || true
    pkill -9 -f "start_codegen_server\.py" 2>/dev/null || true
    
    # Kill any Ray processes (from GRPO training)
    echo "[solver_train] Killing Ray processes..."
    ray stop --force 2>/dev/null || true
    
    # Kill any Python processes that might be holding GPU memory
    echo "[solver_train] Killing training-related Python processes..."
    pkill -9 -f "proposal_generate\.py" 2>/dev/null || true
    pkill -9 -f "code_generate\.py" 2>/dev/null || true
    pkill -9 -f "evaluate_imagefree\." 2>/dev/null || true
    pkill -9 -f "verl\.trainer\.main" 2>/dev/null || true
    
    # Clean up any PID files
    echo "[solver_train] Cleaning up PID files..."
    rm -f "${STORAGE_PATH:-.}/temp_results/solver_service_pids.env" 2>/dev/null || true
    rm -f "${STORAGE_PATH:-.}/temp_results/proposer_service_pids.env" 2>/dev/null || true
    
    echo "[solver_train] Killing GPU compute processes (nvidia-smi)..."
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    
    # Wait a moment for processes to release GPU memory
    sleep 3
    
    # Clear GPU memory cache on all devices
    echo "[solver_train] Clearing GPU memory cache..."
    python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(i)
        print(f'GPU memory cache cleared on {torch.cuda.device_count()} device(s)')
    else:
        print('CUDA not available, skipping GPU cleanup')
except ImportError:
    print('PyTorch not available, skipping GPU cleanup')
except Exception as e:
    print(f'Error during GPU cleanup: {e}')
" 2>/dev/null || echo "GPU cleanup attempted (may have failed)"
    
    echo "[solver_train] Cleanup complete."
}
trap cleanup_solver_training EXIT INT TERM

solver_model_path=$1
proposer_model_path=$2
codegen_model_path=$3
experiment_name=$4

echo "========================================"
echo "Solver Training: $experiment_name"
echo "  Solver init model: $solver_model_path"
echo "  Proposer model: $proposer_model_path"
echo "  CodeGen model: $codegen_model_path"
echo "========================================"

export VLLM_DISABLE_COMPILE_CACHE=1

# ---------------------------------------------------------------------------
# Helper: kill all GPU processes and clear CUDA cache between sub-steps.
# Reusable within solver_train.sh so each sub-step gets a clean GPU.
# ---------------------------------------------------------------------------
_kill_gpu_processes() {
    local label="${1:-solver_train}"
    echo "[$label] Killing named processes..."
    pkill -9 -f "proposal_generate\.py" 2>/dev/null || true
    pkill -9 -f "code_generate\.py" 2>/dev/null || true
    pkill -9 -f "evaluate_imagefree\." 2>/dev/null || true
    pkill -9 -f "python.*-m vllm" 2>/dev/null || true
    pkill -9 -f "vllm\.entrypoints" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "start_vllm_server\.py" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    pkill -9 -f "verl\.trainer\.main" 2>/dev/null || true
    echo "[$label] Killing GPU compute processes (nvidia-smi)..."
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    sleep 5
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    sleep 5
    python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(i)
        print(f'[$label] GPU cache cleared on {torch.cuda.device_count()} device(s)')
except Exception:
    pass
" 2>/dev/null || true
    sleep 3
    echo "[$label] GPU cleanup done."
}

# Solver training uses 8 rollouts (override any lower value from proposer/codegen phases)
export SOLVER_N_ROLLOUTS=8

# Clean up any leftover GPU memory from previous runs
echo "[Pre-flight] Cleaning up GPU memory..."
_kill_gpu_processes "Pre-flight"

# Step 1: Generate proposals (caption + easy Q + hard Q)
# Solver uses its own count (NUM_PROPOSALS_PER_GPU_SOLVER) because many proposals are
# lost to render failures + difficulty filtering, so we need a larger starting pool.
NUM_PROPOSALS_PER_GPU_SOLVER=${NUM_PROPOSALS_PER_GPU_SOLVER:-${NUM_PROPOSALS_PER_GPU:-512}}
echo ""
echo "========== PHASE 1/6: PROPOSALS =========="
echo "[Step 1/6] Generating proposals (${NUM_PROPOSALS_PER_GPU_SOLVER} per GPU)..."
bash MM-zero_final/proposal_generate/proposal_generate.bash $proposer_model_path $NUM_PROPOSALS_PER_GPU_SOLVER $experiment_name

# Clean GPUs after proposal generation (8 vLLM processes on GPUs 0-7)
_kill_gpu_processes "Step 1→2"

# Step 2: Generate code from proposals using CodeGen
echo ""
echo "========== PHASE 2/6: CODE GENERATION =========="
echo "[Step 2/6] Generating code from proposals..."
bash MM-zero_final/code_generate/code_generate.bash $codegen_model_path $experiment_name

# Clean GPUs after code generation (8 vLLM processes on GPUs 0-7)
_kill_gpu_processes "Step 2→3"

# Step 3: Compile/render code to images (parallel with pre-loaded matplotlib).
# Cap: once >=70% of images render successfully (or >=2000 OK), remaining items get at most 3 min; then rest are skipped.
# Fallback: once 90% of tasks have completed, remaining get 3 min so we don't hang on stuck tasks.
echo ""
echo "========== PHASE 3/6: RENDERING =========="
echo "[Step 3/6] Rendering code to images (progress: done/total — OK count; 70%% cap + 3 min for rest)..."
RENDER_WORKERS=${RENDER_MAX_WORKERS:-16}
python MM-zero_final/code_render/render_code.py --experiment_name $experiment_name --workers $RENDER_WORKERS --timeout 30

sleep 5

# Step 4: Evaluate with Solver to get silver labels + score difficulty
echo ""
echo "========== PHASE 4/6: EVALUATION =========="
echo "[Step 4/6] Evaluating rendered images with Solver (8 workers, 10 rollouts; progress: [0]..[7] chunk X/Y)..."
EVAL_NUM_SAMPLES=${EVAL_NUM_SAMPLES:-10}
bash MM-zero_final/question_evaluate/evaluate_imagefree.sh $solver_model_path $experiment_name $EVAL_NUM_SAMPLES

# Clean GPUs after evaluation (8 vLLM processes on GPUs 0-7)
_kill_gpu_processes "Step 4→5"

# Step 5: Filter by difficulty and save as parquet
echo ""
echo "========== PHASE 5/6: UPLOAD PARQUET =========="
echo "[Step 5/6] Filtering and saving training data..."
python MM-zero_final/question_evaluate/upload_imagefree.py \
    --min_easy_consistency 0.5 \
    --min_hard_consistency 0.25 \
    --max_hard_consistency 0.75 \
    --save_name ${experiment_name}

sleep 5

# Steps per run (from main.sh or env; default 20)
TRAIN_STEPS=${TRAIN_STEPS:-20}

# Optional: override batch sizes via env for faster debugging (leave unset to use config default)
ROLLOUT_BATCH_SIZE_SOLVER=${ROLLOUT_BATCH_SIZE_SOLVER:-}
ROLLOUT_N_SOLVER=${ROLLOUT_N_SOLVER:-}
GLOBAL_BATCH_SIZE_SOLVER=${GLOBAL_BATCH_SIZE_SOLVER:-}

# Auto-clamp batch sizes if dataset is too small (prevents "assert len(train_dataloader) >= 1")
TRAIN_PARQUET="${STORAGE_PATH}/local_parquet/${experiment_name}_train.parquet"
if [ -f "$TRAIN_PARQUET" ]; then
    NUM_SAMPLES=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$TRAIN_PARQUET')))" 2>/dev/null || echo "0")
    echo "[solver_train] Training dataset has $NUM_SAMPLES samples."
    # rollout_batch_size must be <= NUM_SAMPLES; clamp if needed
    RBS=${ROLLOUT_BATCH_SIZE_SOLVER:-512}
    GBS=${GLOBAL_BATCH_SIZE_SOLVER:-128}
    if [ "$NUM_SAMPLES" -gt 0 ] && [ "$RBS" -gt "$NUM_SAMPLES" ]; then
        # Round down to largest power-of-2 that fits, minimum 16
        ROLLOUT_BATCH_SIZE_SOLVER=$(python3 -c "n=$NUM_SAMPLES; v=16; [exec('v*=2') for _ in range(20) if v*2<=n]; print(v)")
        echo "[solver_train] WARNING: Clamped rollout_batch_size from $RBS to $ROLLOUT_BATCH_SIZE_SOLVER (dataset has only $NUM_SAMPLES samples)."
    fi
    if [ -n "$ROLLOUT_BATCH_SIZE_SOLVER" ] && [ "$GBS" -gt "${ROLLOUT_BATCH_SIZE_SOLVER:-$GBS}" ]; then
        GLOBAL_BATCH_SIZE_SOLVER=$ROLLOUT_BATCH_SIZE_SOLVER
        echo "[solver_train] WARNING: Clamped global_batch_size to $GLOBAL_BATCH_SIZE_SOLVER to match rollout_batch_size."
    fi
fi

EXTRA_ARGS=""
[ -n "$ROLLOUT_BATCH_SIZE_SOLVER" ] && EXTRA_ARGS="$EXTRA_ARGS data.rollout_batch_size=$ROLLOUT_BATCH_SIZE_SOLVER"
[ -n "$ROLLOUT_N_SOLVER" ] && EXTRA_ARGS="$EXTRA_ARGS worker.rollout.n=$ROLLOUT_N_SOLVER"
[ -n "$GLOBAL_BATCH_SIZE_SOLVER" ] && EXTRA_ARGS="$EXTRA_ARGS worker.actor.global_batch_size=$GLOBAL_BATCH_SIZE_SOLVER worker.critic.global_batch_size=$GLOBAL_BATCH_SIZE_SOLVER"

# Step 6: Train Solver with GRPO (config by GPU_MEM: 40gb or 80gb)
echo ""
echo "========== PHASE 6/6: SOLVER GRPO TRAINING =========="
GPU_MEM=${GPU_MEM:-40}
echo "[Step 6/6] Training Solver with GRPO (max_steps=$TRAIN_STEPS, rollout_batch_size=${ROLLOUT_BATCH_SIZE_SOLVER:-config})..."
python3 -m verl.trainer.main \
    config=MM-zero_final/configs/imagefree_solver_config_${GPU_MEM}gb${MODEL_SIZE:+_${MODEL_SIZE}}.yaml \
    data.max_response_length=4096 \
    data.train_files=${STORAGE_PATH}/local_parquet/${experiment_name}_train.parquet \
    data.val_files=hiyouga/geometry3k@test \
    data.format_prompt=./MM-zero_final/format_prompt/solver.jinja \
    worker.actor.model.model_path=$solver_model_path \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.reward.reward_function=./MM-zero_final/reward_function/cot_val_solver.py:compute_score \
    trainer.total_epochs=10 \
    trainer.max_steps=$TRAIN_STEPS \
    trainer.save_freq=$TRAIN_STEPS \
    $EXTRA_ARGS \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/

# Clean GPUs after GRPO training (Ray/vLLM processes)
_kill_gpu_processes "Step 6 done"

# Merge FSDP shards (only if training produced a checkpoint)
ACTOR_DIR="${STORAGE_PATH}/models/${experiment_name}/global_step_${TRAIN_STEPS}/actor"
if [ -d "$ACTOR_DIR" ]; then
    # Ensure huggingface/ dir has config.json (FSDP checkpoint manager sometimes fails to save it)
    HUGGINGFACE_DIR="${ACTOR_DIR}/huggingface"
    if [ ! -f "$HUGGINGFACE_DIR/config.json" ]; then
        echo "huggingface/config.json missing — copying from base model ($solver_model_path)..."
        mkdir -p "$HUGGINGFACE_DIR"
        for f in config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json \
                 added_tokens.json merges.txt vocab.json preprocessor_config.json chat_template.jinja \
                 video_preprocessor_config.json; do
            [ -f "$solver_model_path/$f" ] && cp "$solver_model_path/$f" "$HUGGINGFACE_DIR/"
        done
    fi
    echo "Merging model shards..."
    python scripts/model_merger.py --local_dir "$ACTOR_DIR"
    # After merging, model is in actor/huggingface; verify it exists
    if [ ! -f "$HUGGINGFACE_DIR/config.json" ]; then
        echo "ERROR: Merged model not found at $HUGGINGFACE_DIR (merge may have failed)."
        exit 1
    fi
    # Remove FSDP shard .pt files to save storage (merged weights are in huggingface/)
    PT_COUNT=$(find "$ACTOR_DIR" -maxdepth 1 -name '*.pt' | wc -l)
    if [ "$PT_COUNT" -gt 0 ]; then
        echo "Removing $PT_COUNT FSDP shard .pt files from $ACTOR_DIR to save storage..."
        rm -f "$ACTOR_DIR"/*.pt
    fi
else
    echo "ERROR: No checkpoint found at $ACTOR_DIR (Solver training may have failed). Skipping model merge."
    exit 1
fi

sleep 10

# Explicit cleanup before exit (trap will also run, but this ensures it happens)
echo "[solver_train] Final cleanup before exit..."
cleanup_solver_training

echo "Solver training finished: $experiment_name"
