#!/bin/bash
# =============================================================================
# ImageFree Self-Play: Proposer Training (EasyR1)
# =============================================================================
# The Proposer generates chart descriptions (captions), easy questions, and hard questions.
# It is text-only — no images needed.
#
# Reward pipeline (per proposer output):
#   1. Caption + questions → CodeGen service → matplotlib code (N=8 samples)
#   2. Code → render subprocess → image
#   3. For each successful render:
#      a) image + easy_q → Solver (8 rollouts) → solvability score
#      b) image + hard_q → Solver (8 rollouts) → difficulty score
#   4. reward = (sum of contributions) / N
#
# Services:
#   GPUs 4-5: CodeGen vLLM service (ports 7000-7001)
#   GPUs 6-7: Solver  vLLM service (ports 6000-6001)
#
# Usage: bash SelfAgent/scripts/proposer_train.sh \
#            <solver_model> <proposer_model> <codegen_model> <save_name>
# =============================================================================

set -x

# -----------------------------------------------------------------------------
# Cleanup: always kill vLLM services (CodeGen + Solver) on exit so next phase has fresh GPU memory.
# Runs on normal exit, failure, or Ctrl+C. Kill by PIDs first, then pkill fallback.
# -----------------------------------------------------------------------------
kill_proposer_vllm_services() {
    echo "[proposer_train] Stopping vLLM services (CodeGen + Solver) to free GPUs 4-7..."
    PIDS_FILE="${STORAGE_PATH:-.}/temp_results/proposer_service_pids.env"
    if [ -f "$PIDS_FILE" ]; then
        set -a
        # shellcheck source=/dev/null
        source "$PIDS_FILE" 2>/dev/null || true
        set +a
        for pid in $CODEGEN_PID_0 $CODEGEN_PID_1 $SOLVER_PID_0 $SOLVER_PID_1; do
            [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        done
        rm -f "$PIDS_FILE"
    fi
    pkill -9 -f "start_vllm_server" 2>/dev/null || true
    pkill -9 -f "start_codegen_server" 2>/dev/null || true
    pkill -9 -f "vllm" 2>/dev/null || true
    pkill -9 -f "VLLM" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    # Kill ALL processes using any GPU (catches VLLM::EngineCore and orphaned children)
    echo "[proposer_train] Killing all processes using GPU (nvidia-smi)..."
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    # Final safety net: kill anything still holding GPU device files open
    for dev in /dev/nvidia[0-9]*; do
        [ -e "$dev" ] && fuser -k -9 "$dev" 2>/dev/null || true
    done
    echo "[proposer_train] vLLM services stopped."
}
trap kill_proposer_vllm_services EXIT INT TERM

solver_model_path=$1
proposer_model_path=$2
codegen_model_path=$3
save_path=$4

echo "========================================"
echo "Proposer Training: $save_path"
echo "  Solver model (for reward):  $solver_model_path"
echo "  CodeGen model (for reward): $codegen_model_path"
echo "  Proposer init model:        $proposer_model_path"
echo "========================================"

# Generate unique RUN_ID
RUN_ID=$(date +%s%N)
export RUN_ID
echo "RUN_ID=$RUN_ID"

# Launch BOTH CodeGen and Solver as vLLM services
# CodeGen on GPUs 4-5 (ports 7000-7001), Solver on GPUs 6-7 (ports 6000-6001)
# Pass max_model_len for 40GB GPUs so KV cache fits (see main.sh GPU_MEM)
GPU_MEM=${GPU_MEM:-40}
VLLM_MAX_LEN=""
if [ "$GPU_MEM" = "40" ]; then
    VLLM_MAX_LEN="32768"
fi
bash SelfAgent/vllm_service_init/start_proposer_services.sh "$codegen_model_path" "$solver_model_path" "$VLLM_MAX_LEN"
echo "CodeGen + Solver services started"

# Load chosen ports (default or alternates if 7000-7001/6000-6001 were in use)
if [ -f "${STORAGE_PATH:-.}/temp_results/proposer_service_ports.env" ]; then
    set -a
    source "${STORAGE_PATH:-.}/temp_results/proposer_service_ports.env"
    set +a
fi
export CODEGEN_PORT_0=${CODEGEN_PORT_0:-7010}
export CODEGEN_PORT_1=${CODEGEN_PORT_1:-7011}
export SOLVER_PORT_0=${SOLVER_PORT_0:-6010}
export SOLVER_PORT_1=${SOLVER_PORT_1:-6011}

# Wait for vLLM to load models and finish torch.compile + CUDA graph capture (often 60–90s total).
echo "[proposer_train] Waiting 90s for vLLM engines to initialize (model load + compile)..."
sleep 90

# Quick check: our CodeGen and Solver ports must be reachable or reward will get "CodeGen results not found".
echo "[proposer_train] Checking CodeGen and Solver services..."
for port in $CODEGEN_PORT_0 $CODEGEN_PORT_1 $SOLVER_PORT_0 $SOLVER_PORT_1; do
    if curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "http://127.0.0.1:${port}/" 2>/dev/null | grep -q '[0-9]'; then
        echo "  port $port: reachable"
    else
        echo "  port $port: NOT reachable (check logs in ${STORAGE_PATH:-.}/temp_results/)"
    fi
done

# Steps per run (from main.sh or env; default 20)
TRAIN_STEPS=${TRAIN_STEPS:-20}
# Optional: fewer proposals per step = faster step (set in env or leave unset to use config default)
# ROLLOUT_BATCH_SIZE=32  →  data.rollout_batch_size=32
# ROLLOUT_N=4            →  worker.rollout.n=4 (fewer CodeGen/Render/Solver calls per step)
ROLLOUT_BATCH_SIZE=${PROPOSER_ROLLOUT_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE:-}}
ROLLOUT_N=${ROLLOUT_N:-8}
GLOBAL_BATCH_SIZE_PROPOSER=${GLOBAL_BATCH_SIZE_PROPOSER:-}

EXTRA_ARGS=""
[ -n "$ROLLOUT_BATCH_SIZE" ] && EXTRA_ARGS="$EXTRA_ARGS data.rollout_batch_size=$ROLLOUT_BATCH_SIZE"
[ -n "$GLOBAL_BATCH_SIZE_PROPOSER" ] && EXTRA_ARGS="$EXTRA_ARGS worker.actor.global_batch_size=$GLOBAL_BATCH_SIZE_PROPOSER worker.critic.global_batch_size=$GLOBAL_BATCH_SIZE_PROPOSER"

# Train Proposer with GRPO on GPUs 0-3 (config chosen by GPU_MEM: 40gb or 80gb)
echo "Starting Proposer GRPO training (max_steps=$TRAIN_STEPS, rollout.n=${ROLLOUT_N}, rollout_batch_size=${ROLLOUT_BATCH_SIZE:-config})..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=SelfAgent/configs/imagefree_proposer_config_${GPU_MEM}gb${MODEL_SIZE:+_${MODEL_SIZE}}${CONFIG_SUFFIX:-}.yaml \
    data.train_files=./SelfAgent/data/text_seed_prompts.parquet \
    data.val_files=hiyouga/geometry3k@test \
    data.prompt_key=problem \
    data.answer_key=answer \
    worker.actor.model.model_path=$proposer_model_path \
    worker.rollout.n=$ROLLOUT_N \
    trainer.max_steps=$TRAIN_STEPS \
    trainer.save_freq=$TRAIN_STEPS \
    $EXTRA_ARGS \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=10 \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=false

sleep 5

# Merge FSDP shards into HuggingFace format (only if training produced a checkpoint)
ACTOR_DIR="${STORAGE_PATH}/models/$save_path/global_step_${TRAIN_STEPS}/actor"
if [ -d "$ACTOR_DIR" ]; then
    # Ensure huggingface/ dir has config.json (FSDP checkpoint manager sometimes fails to save it)
    HUGGINGFACE_DIR="${ACTOR_DIR}/huggingface"
    if [ ! -f "$HUGGINGFACE_DIR/config.json" ]; then
        echo "huggingface/config.json missing — copying from base model ($proposer_model_path)..."
        mkdir -p "$HUGGINGFACE_DIR"
        for f in config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json \
                 added_tokens.json merges.txt vocab.json preprocessor_config.json chat_template.jinja \
                 video_preprocessor_config.json; do
            [ -f "$proposer_model_path/$f" ] && cp "$proposer_model_path/$f" "$HUGGINGFACE_DIR/"
        done
    fi
    echo "Merging model shards..."
    python scripts/model_merger.py --local_dir "$ACTOR_DIR"
    # After merging, model is in actor/huggingface; verify it exists
    if [ ! -f "$HUGGINGFACE_DIR/config.json" ]; then
        echo "ERROR: Merged model not found at $HUGGINGFACE_DIR (merge may have failed)."
        exit 1
    fi
else
    echo "ERROR: No checkpoint found at $ACTOR_DIR (Proposer training may have failed). Skipping model merge."
    exit 1
fi

sleep 10

# Cleanup runs automatically via trap; no need to call here (trap runs on script exit).
echo "Proposer training finished: $save_path"
