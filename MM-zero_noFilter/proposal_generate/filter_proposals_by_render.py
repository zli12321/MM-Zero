#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filter proposals by render success rate before CodeGen GRPO training.

For each proposal in this GPU's shard, generates N code samples using the
current CodeGen model, renders each, and keeps only proposals whose render
success rate falls within [min_rate, max_rate].

Designed to run 8 copies in parallel (one per GPU), like proposal_generate.py.

Usage (single shard):
    CUDA_VISIBLE_DEVICES=0 python filter_proposals_by_render.py \
        --codegen_model <path> --save_name <exp> --suffix 0 \
        --n_samples 8 --min_render_rate 0.25 --max_render_rate 0.75
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

STORAGE_PATH = os.getenv("STORAGE_PATH")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codegen_model", type=str, required=True)
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="0",
                        help="Shard index (usually GPU index 0-7)")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--min_render_rate", type=float, default=0.25)
    parser.add_argument("--max_render_rate", type=float, default=0.75)
    parser.add_argument("--render_timeout", type=int, default=30)
    parser.add_argument("--render_workers", type=int, default=8)
    args = parser.parse_args()

    # Load this shard's proposals
    input_file = f"{STORAGE_PATH}/generated_proposals/{args.save_name}_{args.suffix}.json"
    print(f"[filter-{args.suffix}] Loading proposals from {input_file}")
    try:
        with open(input_file) as f:
            proposals = json.load(f)
    except FileNotFoundError:
        print(f"[filter-{args.suffix}] ERROR: {input_file} not found")
        return

    valid_proposals = [p for p in proposals
                       if p.get("caption") and p.get("easy_question") and p.get("easy_answer")]
    print(f"[filter-{args.suffix}] {len(valid_proposals)} valid proposals")

    if not valid_proposals:
        output_file = f"{STORAGE_PATH}/generated_proposals/{args.save_name}_{args.suffix}.json"
        with open(output_file, "w") as f:
            json.dump([], f)
        print(f"[filter-{args.suffix}] No valid proposals, wrote empty shard")
        return

    # Generate N code samples per proposal
    import vllm
    from transformers import AutoTokenizer

    jinja_path = os.path.join(os.path.dirname(__file__), '..', 'format_prompt', 'codegen.jinja')
    with open(jinja_path) as f:
        codegen_template = f.read()

    def build_prompt(proposal):
        hard_q = proposal.get('hard_question', '') or 'N/A'
        hard_a = proposal.get('hard_answer', '') or 'N/A'
        content = (
            f"Visual Type: svg\n\n"
            f"Chart Description:\n{proposal['caption']}\n\n"
            f"Easy Question: {proposal['easy_question']}\n"
            f"Easy Answer: {proposal['easy_answer']}\n\n"
            f"Hard Question: {hard_q}\n"
            f"Hard Answer: {hard_a}"
        )
        user_msg = codegen_template.replace('{{ content | trim }}', content.strip())
        return (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.codegen_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.85"))
    max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "32768"))
    model = vllm.LLM(
        model=args.codegen_model,
        tokenizer=args.codegen_model,
        seed=int(args.suffix),
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
    )

    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=0.7,
        top_p=0.95,
        n=args.n_samples,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    prompts = [build_prompt(p) for p in valid_proposals]
    total_gens = len(prompts) * args.n_samples
    print(f"[filter-{args.suffix}] Generating {args.n_samples} code samples × {len(prompts)} proposals = {total_gens} total...")
    completions = model.generate(prompts, sampling_params=sample_params)

    # Extract SVG code
    import regex as re

    def extract_svg(response_text):
        m = re.search(r"```svg\s*\n([\s\S]*?)```", response_text)
        if m:
            return m.group(1).strip()
        m = re.search(r"```xml\s*\n([\s\S]*?)```", response_text)
        if m and "<svg" in m.group(1).lower():
            return m.group(1).strip()
        m = re.search(r"(<svg[\s\S]*?</svg>)", response_text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return ""

    render_tasks = []
    task_map = []
    for pidx, completion in enumerate(completions):
        for sidx, output in enumerate(completion.outputs):
            code = extract_svg(output.text)
            render_tasks.append((code, "svg"))
            task_map.append(pidx)

    # Free GPU before CPU-only rendering
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Render
    from SelfAgent_svg.code_render.render_code import render_batch_codes

    print(f"[filter-{args.suffix}] Rendering {len(render_tasks)} code samples ({args.render_workers} workers)...")
    render_results = render_batch_codes(
        render_tasks,
        max_workers=args.render_workers,
        timeout=args.render_timeout,
    )

    # Compute render success rate per proposal
    success_counts = [0] * len(valid_proposals)
    total_counts = [0] * len(valid_proposals)
    for pidx, result in zip(task_map, render_results):
        total_counts[pidx] += 1
        if result is not None:
            success_counts[pidx] += 1

    # Filter
    filtered = []
    n_too_low = 0
    n_too_high = 0
    for pidx, proposal in enumerate(valid_proposals):
        total = total_counts[pidx]
        if total == 0:
            n_too_low += 1
            continue
        rate = success_counts[pidx] / total
        if rate < args.min_render_rate:
            n_too_low += 1
        elif rate > args.max_render_rate:
            n_too_high += 1
        else:
            proposal["render_success_rate"] = rate
            filtered.append(proposal)

    print(f"[filter-{args.suffix}] === Results ===")
    print(f"  Total: {len(valid_proposals)} | Kept: {len(filtered)} | "
          f"Too low: {n_too_low} | Too high: {n_too_high}")

    # Overwrite shard file with filtered proposals
    with open(input_file, "w") as f:
        json.dump(filtered, f, indent=2)
    print(f"[filter-{args.suffix}] Wrote {len(filtered)} filtered proposals to {input_file}")


if __name__ == "__main__":
    main()
