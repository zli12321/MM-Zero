# Training speed tuning

Ways to speed up Self-Agent SVG training (at the cost of some quality or data volume).

## GPU layout (proposer vs codegen phase)

- **Proposer training**: 2 GPUs training (0–1), **2 CodeGen** vLLM (GPUs 2–3), **4 Solver** vLLM (GPUs 4–7). Solver (hard Q) is the main bottleneck, so we give it 4 instances.
- **CodeGen training**: 2 GPUs training (0–1), **6 Solver** vLLM (GPUs 2–7). No CodeGen service needed for reward (only render + Solver).

## 1. Reward pipeline (largest impact)

| Env var | Default | Faster value | Effect |
|--------|---------|--------------|--------|
| `CODEGEN_N_SAMPLES` | 8 | 4 | Half the code samples per proposal. Set **before** starting CodeGen vLLM services. |
| `SOLVER_N_ROLLOUTS` | 10 | 4 or 8 | Fewer solver answers per (image, question). Set **before** starting Solver vLLM services. |

## 2. main_svg.sh env vars

| Env var | Default | Faster | Effect |
|--------|---------|--------|--------|
| `TRAIN_STEPS` | 5 | 3 | Fewer gradient steps per model. |
| `NUM_PROPOSALS_PER_GPU` | 450 | 300 | Fewer proposals per proposer round. |
| `NUM_PROPOSALS_PER_GPU_SOLVER` | 1000 | 600 | Fewer samples per solver round. |
| `RENDER_MAX_WORKERS` | auto | e.g. 16 | More parallel SVG→PNG renders. |

## 3. Config YAML

- **max_steps**: 20 → 10 or 15 to stop earlier.
- **micro_batch_size_per_device_for_update/experience**: increase if GPU memory allows.
- **enable_gradient_checkpointing**: `false` only if you have plenty of VRAM (faster, more memory).

## 4. Fewer iterations

In `scripts/main_svg.sh`, change `for i in {2..6}` to e.g. `{2..4}`.

---

## Proposer OOM on 2×80GB (update_policy / backward)

If you see `torch.OutOfMemoryError` during **loss.backward()** in proposer training (GPUs 0–1), the cause is usually peak memory during the **experience** phase (log-prob computation) or **update** phase (backward over rollouts).

**What we did in config:**

- **imagefree_proposer_config_80gb.yaml**: `micro_batch_size_per_device_for_experience` reduced from **8 → 4** so each GPU processes 4 sequences at a time instead of 8.
- **proposer_train.sh**: sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce CUDA fragmentation.

**If OOM persists:**

1. Reduce **GLOBAL_BATCH_SIZE_PROPOSER** (e.g. 8 → 4) in `main_svg.sh` or env.
2. Reduce **PROPOSER_ROLLOUT_BATCH_SIZE** (e.g. 16 → 8).
3. In the 80GB proposer config, set **micro_batch_size_per_device_for_experience: 2**.
4. Keep **enable_gradient_checkpointing: true** in `worker.actor.model`.

## CodeGen training OOM (2×80GB)

Same idea as proposer: **codegen_train.sh** now sets **PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True**. The **imagefree_codegen_config_80gb.yaml** already uses **micro_batch_size_per_device_for_experience: 4**. If you still hit OOM during CodeGen training, set **micro_batch_size_per_device_for_experience: 2** in that config.
