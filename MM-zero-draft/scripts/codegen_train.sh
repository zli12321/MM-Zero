#!/bin/bash
# =============================================================================
# ImageFree Self-Play: Code Generator Training (EasyR1)
# =============================================================================
# The CodeGen model takes caption + questions and generates matplotlib code.
#
# Pipeline:
#   1. Generate proposals from the Proposer
#   2. Convert proposals to a training parquet
#   3. Train CodeGen with GRPO (reward = renderability + solvability + difficulty)
#
# The Solver runs as a vLLM service on GPUs 4-7 for functional reward.
#
# Usage: bash SelfAgent/scripts/codegen_train.sh <codegen_model> <proposer_model> <save_name>
# =============================================================================

set -x

# -----------------------------------------------------------------------------
# Cleanup: always kill Solver vLLM services on exit so next phase has fresh GPU memory.
# Runs on normal exit, failure, or Ctrl+C.
# -----------------------------------------------------------------------------
kill_codegen_phase_vllm_services() {
    echo "[codegen_train] Stopping vLLM Solver services (GPUs 4-7)..."
    PIDS_FILE="${STORAGE_PATH:-.}/temp_results/solver_service_pids.env"
    if [ -f "$PIDS_FILE" ]; then
        set -a
        # shellcheck source=/dev/null
        source "$PIDS_FILE" 2>/dev/null || true
        set +a
        for pid in $SOLVER_PID_0 $SOLVER_PID_1 $SOLVER_PID_2 $SOLVER_PID_3; do
            [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        done
        rm -f "$PIDS_FILE"
    fi
    pkill -9 -f "start_vllm_server" 2>/dev/null || true
    pkill -9 -f "vllm" 2>/dev/null || true
    pkill -9 -f "VLLM" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    # Kill ALL processes using any GPU (catches VLLM::EngineCore and orphaned children)
    echo "[codegen_train] Killing all processes using GPU (nvidia-smi)..."
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
    echo "[codegen_train] vLLM Solver services stopped."
}
trap kill_codegen_phase_vllm_services EXIT INT TERM

codegen_model_path=$1
proposer_model_path=$2
save_path=$3

echo "========================================"
echo "CodeGen Training: $save_path"
echo "  CodeGen init model: $codegen_model_path"
echo "  Proposer model (for proposals): $proposer_model_path"
echo "========================================"

export VLLM_DISABLE_COMPILE_CACHE=1

# Clean up any leftover GPU memory from previous runs
echo "[Pre-flight] Cleaning up GPU memory..."
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
sleep 5
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
echo "[Pre-flight] GPU cleanup done."

# Step 1: Generate proposals from the Proposer
NUM_PROPOSALS_PER_GPU=${NUM_PROPOSALS_PER_GPU:-500}
echo "[Step 1/4] Generating proposals (${NUM_PROPOSALS_PER_GPU} per GPU)..."
bash SelfAgent/proposal_generate/proposal_generate.bash $proposer_model_path $NUM_PROPOSALS_PER_GPU $save_path

# Clean GPUs after proposal generation (8 vLLM processes on GPUs 0-7)
echo "[Step 1→2] Killing all GPU processes after proposal generation..."
pkill -9 -f "proposal_generate.py" 2>/dev/null || true
pkill -9 -f "vllm" 2>/dev/null || true
pkill -9 -f "VLLM" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
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
sleep 5
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
echo "[Step 1→2] GPU cleanup done."

# Step 2: Convert proposals to parquet for GRPO training
echo "[Step 2/4] Converting proposals to parquet..."
python -c "
import json, os, glob, pandas as pd

STORAGE_PATH = os.getenv('STORAGE_PATH')
save_path = '${save_path}'

# Collect all proposal files
all_proposals = []
for i in range(8):
    fpath = f'{STORAGE_PATH}/generated_proposals/{save_path}_{i}.json'
    if os.path.exists(fpath):
        with open(fpath) as f:
            data = json.load(f)
            for item in data:
                if item.get('caption') and item.get('easy_question') and item.get('easy_answer'):
                    hard_q = item.get('hard_question', 'N/A') or 'N/A'
                    hard_a = item.get('hard_answer', 'N/A') or 'N/A'
                    vt = item.get('visual_type', 'matplotlib') or 'matplotlib'
                    if vt not in ('matplotlib', 'plotly', 'pillow', 'svg'):
                        vt = 'matplotlib'
                    all_proposals.append({
                        'prompt_text': f\"\"\"Visual Type: {vt}\n\nChart Description:\n{item['caption']}\n\nEasy Question: {item['easy_question']}\nEasy Answer: {item['easy_answer']}\n\nHard Question: {hard_q}\nHard Answer: {hard_a}\"\"\",
                        'easy_answer': item['easy_answer'],
                        'caption': item['caption'],
                        'easy_question': item['easy_question'],
                        'hard_question': item.get('hard_question', ''),
                        'hard_answer': item.get('hard_answer', ''),
                        'visual_type': vt,
                    })

print(f'Total valid proposals: {len(all_proposals)}')
if all_proposals:
    df = pd.DataFrame(all_proposals)
    outdir = f'{STORAGE_PATH}/generated_proposals'
    os.makedirs(outdir, exist_ok=True)
    outpath = f'{outdir}/{save_path}_proposals.parquet'
    df.to_parquet(outpath, index=False)
    print(f'Saved to {outpath}')
else:
    print('ERROR: No valid proposals found!')
    exit(1)
"

sleep 5

# Step 3: Launch Solver as vLLM service for functional reward
# Free GPUs 4-7 before starting (proposal gen used 0-7; leftover memory caused 6002/6003 to fail with "Free memory < 0.8*80GiB").
echo "[Step 3/4] Cleaning GPUs 4-7 before starting Solver services..."
pkill -9 -f "proposal_generate.py" 2>/dev/null || true
pkill -9 -f "vllm" 2>/dev/null || true
pkill -9 -f "VLLM" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
pkill -9 -f "start_vllm_server" 2>/dev/null || true
echo "[Step 3/4] Killing all processes using GPU (nvidia-smi)..."
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
done
sleep 10
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
        print(f'[Step 3/4] GPU cache cleared on {torch.cuda.device_count()} device(s)')
except Exception:
    pass
" 2>/dev/null || true
sleep 5

RUN_ID=$(date +%s%N)
export RUN_ID
GPU_MEM=${GPU_MEM:-40}
VLLM_MAX_LEN=""
[ "$GPU_MEM" = "40" ] && VLLM_MAX_LEN="32768"
# Use gpu_mem_util=0.45 so each Solver needs ~36 GiB; fits when GPUs have leftover memory.
export SOLVER_GPU_MEM_UTIL=0.45
echo "[Step 3/4] Starting vLLM Solver service (RUN_ID=$RUN_ID, SOLVER_GPU_MEM_UTIL=$SOLVER_GPU_MEM_UTIL)..."
bash SelfAgent/vllm_service_init/start_solver_services.sh "$codegen_model_path" "$RUN_ID" "$VLLM_MAX_LEN"

# Wait for vLLM services to be ready with health checks
echo "[Step 3/4] Waiting for Solver services to be ready..."
LOG_DIR="${STORAGE_PATH:-.}/temp_results"
MAX_WAIT=300  # Maximum wait time in seconds
WAIT_INTERVAL=5
ELAPSED=0
ALL_READY=false

# Use a single Python script to avoid quoting issues; try both 127.0.0.1 and 0.0.0.0
# (servers bind to 127.0.0.1; in Docker/some envs checking 0.0.0.0 can help)
_check_ports_py() {
    python3 << 'PYEOF'
import socket
ports = [6000, 6001, 6002, 6003]
hosts = ['127.0.0.1', '0.0.0.0']
ready = 0
for p in ports:
    s = None
    for h in hosts:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            if s.connect_ex((h, p)) == 0:
                ready += 1
                break
        except Exception:
            pass
        finally:
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
print(ready)
PYEOF
}

while [ $ELAPSED -lt $MAX_WAIT ]; do
    READY_COUNT=$(_check_ports_py)
    
    if [ "$READY_COUNT" = "4" ]; then
        ALL_READY=true
        break
    fi
    
    echo "[Step 3/4] Waiting for Solver services... ($READY_COUNT/4 ready, ${ELAPSED}s elapsed)"
    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

if [ "$ALL_READY" = false ]; then
    echo "ERROR: Solver services failed to start after ${MAX_WAIT}s"
    echo "Check logs: $LOG_DIR/solver_*.log"
    echo "Checking service status (trying 127.0.0.1 and 0.0.0.0)..."
    for port in 6000 6001 6002 6003; do
        _ok=0
        for host in 127.0.0.1 0.0.0.0; do
            if python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2)
try:
    r = s.connect_ex(('$host', $port))
    s.close()
    exit(0 if r == 0 else 1)
except Exception:
    exit(1)
" 2>/dev/null; then
                _ok=1
                break
            fi
        done
        if [ $_ok -eq 0 ]; then
            echo "  Port $port: NOT READY"
            if [ -f "$LOG_DIR/solver_${port}.log" ]; then
                echo "    Last 10 lines of log:"
                tail -10 "$LOG_DIR/solver_${port}.log" | sed 's/^/    /'
            fi
        else
            echo "  Port $port: READY"
        fi
    done
    exit 1
fi

echo "[Step 3/4] All Solver services are ready!"

# Steps per run (from main.sh or env; default 20)
TRAIN_STEPS=${TRAIN_STEPS:-20}

# Optional: override batch sizes via env for faster debugging (leave unset to use config default)
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-}
ROLLOUT_N=${ROLLOUT_N:-8}
GLOBAL_BATCH_SIZE_CODEGEN=${GLOBAL_BATCH_SIZE_CODEGEN:-}

EXTRA_ARGS=""
[ -n "$ROLLOUT_BATCH_SIZE" ] && EXTRA_ARGS="$EXTRA_ARGS data.rollout_batch_size=$ROLLOUT_BATCH_SIZE"
[ -n "$GLOBAL_BATCH_SIZE_CODEGEN" ] && EXTRA_ARGS="$EXTRA_ARGS worker.actor.global_batch_size=$GLOBAL_BATCH_SIZE_CODEGEN worker.critic.global_batch_size=$GLOBAL_BATCH_SIZE_CODEGEN"

# Step 4: Train CodeGen with GRPO (config by GPU_MEM: 40gb or 80gb)
echo "[Step 4/4] Starting CodeGen GRPO training (max_steps=$TRAIN_STEPS, rollout.n=${ROLLOUT_N}, rollout_batch_size=${ROLLOUT_BATCH_SIZE:-config})..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=SelfAgent/configs/imagefree_codegen_config_${GPU_MEM}gb${MODEL_SIZE:+_${MODEL_SIZE}}.yaml \
    data.train_files=${STORAGE_PATH}/generated_proposals/${save_path}_proposals.parquet \
    data.val_files=hiyouga/geometry3k@test \
    data.prompt_key=prompt_text \
    data.val_prompt_key=problem \
    data.answer_key=easy_answer \
    data.val_answer_key=answer \
    worker.actor.model.model_path=$codegen_model_path \
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

# Merge FSDP shards (only if training produced a checkpoint)
ACTOR_DIR="${STORAGE_PATH}/models/$save_path/global_step_${TRAIN_STEPS}/actor"
if [ -d "$ACTOR_DIR" ]; then
    # Ensure huggingface/ dir has config.json (FSDP checkpoint manager sometimes fails to save it)
    HUGGINGFACE_DIR="${ACTOR_DIR}/huggingface"
    if [ ! -f "$HUGGINGFACE_DIR/config.json" ]; then
        echo "huggingface/config.json missing — copying from base model ($codegen_model_path)..."
        mkdir -p "$HUGGINGFACE_DIR"
        for f in config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json \
                 added_tokens.json merges.txt vocab.json preprocessor_config.json chat_template.jinja \
                 video_preprocessor_config.json; do
            [ -f "$codegen_model_path/$f" ] && cp "$codegen_model_path/$f" "$HUGGINGFACE_DIR/"
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
    echo "ERROR: No checkpoint found at $ACTOR_DIR (CodeGen training may have failed). Skipping model merge."
    exit 1
fi

sleep 10

# Clean up GPU memory before script exit (so next iteration has clean GPUs)
# First kill Solver services explicitly (trap will also do this, but doing it here ensures cleanup happens)
echo "[Cleanup] Stopping Solver vLLM services..."
PIDS_FILE="${STORAGE_PATH:-.}/temp_results/solver_service_pids.env"
if [ -f "$PIDS_FILE" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$PIDS_FILE" 2>/dev/null || true
    set +a
    for pid in $SOLVER_PID_0 $SOLVER_PID_1 $SOLVER_PID_2 $SOLVER_PID_3; do
        [ -n "$pid" ] && [ "$pid" -gt 0 ] 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    rm -f "$PIDS_FILE"
fi
pkill -9 -f "start_vllm_server" 2>/dev/null || true
sleep 3  # Give processes time to release GPU memory

# Now free GPU memory on all devices
echo "[Cleanup] Freeing GPU memory on all devices..."
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
" || echo "GPU cleanup attempted (may have failed if PyTorch unavailable)"

sleep 5

# Cleanup runs automatically via trap on script exit.
echo "CodeGen training finished: $save_path"
