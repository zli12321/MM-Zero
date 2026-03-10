# Training Speed & Memory Optimization

Levers to speed up training or fit in memory (Proposer → CodeGen → Solver pipeline).

---

## 1. **Proposer training (biggest bottleneck)**

### Per-step time
Each step = one reward batch: **rollout → CodeGen → Render → Solver (easy + hard)**.  
Most time is in **CodeGen** and **Solver**; **Render** is already parallel (CPU).

| Parameter | Where | Effect | Tradeoff |
|-----------|--------|--------|----------|
| **rollout_batch_size** | `imagefree_proposer_config_*gb.yaml` → `data.rollout_batch_size` | **# of proposals per step.** Smaller = fewer CodeGen/Render/Solver calls = **faster per step**. | Smaller ⇒ more steps to cover dataset; larger ⇒ slower step, fewer steps. |
| **worker.actor.global_batch_size** | Same config | Gradient batch size. Must divide `rollout_batch_size`. | Larger = fewer grad steps per rollout batch (faster) if memory allows. |
| **worker.rollout.n** | Overridden in `proposer_train.sh` (default 8) | Rollouts per proposal (8 → CodeGen 8 codes, 8× render, 8× Solver). | **n=4** halves CodeGen/Render/Solver cost per step but can hurt reward quality. |
| **enable_gradient_checkpointing** | `worker.actor.model` | `true` = less GPU memory, slower; `false` = faster if VRAM allows. | On 80GB you can try `false`. |
| **offload (params / optimizer)** | `worker.actor.offload` | `true` = less VRAM, slower; `false` = faster. | On 80GB try `offload_params: false`, `offload_optimizer: false`. |
| **worker.rollout.gpu_memory_utilization** | Proposer config | Higher = more vLLM batch size during rollout. | 0.7 (40GB) → 0.8 (80GB) can speed rollout. |
| **worker.rollout.max_model_len** | Proposer config | Lower = faster + less memory (40GB: 32768). | Only reduce if you don’t need long context. |

**Quick win:** Reduce **rollout_batch_size** (e.g. 64 → 32) in the proposer config to get **faster steps** (less CodeGen/Render/Solver per step). You can compensate with more steps (`TRAIN_STEPS`) or accept fewer proposals per step.

---

## 2. **CodeGen service (vLLM)**

| Parameter | Where | Effect |
|-----------|--------|--------|
| **chunk_size** | `start_codegen_server.py` `--chunk_size` (default 16) | Prompts per vLLM batch. **Larger = fewer round-trips** (e.g. 32 or 64 on 80GB). |
| **gpu_mem_util** | `start_proposer_services.sh` / `GPU_MEM_UTIL` | Higher = more memory for batch (0.5–0.8). |
| **max_model_len** | CodeGen server `--max_model_len` | Lower = faster + less memory (40GB: 32768). |
| **CODEGEN_HTTP_TIMEOUT** | env (reward phase) | How long to wait for CodeGen (HTTP + result files). **`auto`** (default) = scale with batch size (min 10 min, cap 2 h). **`0`** or **`none`** = wait until done (no hard stop; use if runs are slow and you prefer to wait). **&lt;seconds&gt;** = fixed limit (e.g. `3600`). |

---

## 3. **Solver service (vLLM)**

Same idea: **gpu_mem_util** and **max_model_len** in `start_solver_services.sh` / Solver vLLM args. Higher util and appropriate max_model_len help throughput.

---

## 4. **Render (CPU)**

| Parameter | Where | Effect |
|-----------|--------|--------|
| **RENDER_MAX_WORKERS** | `main.sh` (auto from CPU count) or env | More workers = faster render. Default = half of logical CPUs. |
| **RENDER_USE_SUBPROCESS** | env | Set to `1` to force one Python subprocess per snippet (slower, more isolated). Default = use long-lived process pool (matplotlib/plotly imported once per worker → much faster). |
| **SAVE_RENDER_EXAMPLES** | env | Set to `1` to save a sample of rendered images (and Q&A) per reward step under **STORAGE_PATH/rendered_images/examples/step_0/**, step_1/, … for progression tracing. |
| **SAVE_RENDER_EXAMPLES_N** | env | Max number of example images to save per step (default **10**). Only used if `SAVE_RENDER_EXAMPLES=1`. Questions and answers are saved in **info.json** in each step folder (caption, easy_question, easy_answer, hard_question, hard_answer per image). |

**Why render was slow:** Previously each snippet ran in a **new Python process** (start interpreter + import matplotlib/plotly every time). Now we use a **process pool** where each worker imports once and runs many snippets in-process, so render is much faster unless you set `RENDER_USE_SUBPROCESS=1`.

---

## 5. **Steps per model**

| Parameter | Where | Effect |
|-----------|--------|--------|
| **TRAIN_STEPS** | `main.sh` or env (default 20) | Steps for Proposer, then CodeGen, then Solver per iteration. **Smaller = switch to next model sooner** (e.g. `TRAIN_STEPS=5`). |

---

## 6. **Memory vs speed (80GB)**

On **80GB** you can often:

- Use **80GB configs** (`GPU_MEM=80`).
- Set **offload_params: false**, **offload_optimizer: false** in proposer/codegen/solver configs (if they’re true).
- Set **enable_gradient_checkpointing: false** if OOM is not an issue.
- Increase **rollout_batch_size** or **global_batch_size** slightly for better throughput.
- Increase CodeGen **chunk_size** (e.g. 32–64) and vLLM **gpu_memory_utilization** (e.g. 0.8).

---

## 7. **Override from the command line**

Training scripts pass overrides to `verl.trainer.main`. You can add overrides in the train scripts, e.g.:

```bash
# In proposer_train.sh, add e.g.:
data.rollout_batch_size=32 \
worker.rollout.n=4 \
```

Or set env and use them in the script (e.g. `ROLLOUT_BATCH_SIZE`, `ROLLOUT_N`) so you don’t edit YAML every time.
