#!/usr/bin/env python3
"""
Run an LLM judge (via vLLM) on existing eval_responses to re-grade correctness
of model answers vs gold, then write llm_accuracy_summary.jsonl.

Single GPU:
  python llm_judge_eval.py --eval_responses_dir /path/to/eval_responses --judge_model /path/to/judge

Multi-GPU (8-way data parallel): run 8 processes, one per GPU, then merge.
  CUDA_VISIBLE_DEVICES=0 python llm_judge_eval.py ... --shard_id 0 --num_shards 8 &
  CUDA_VISIBLE_DEVICES=1 python llm_judge_eval.py ... --shard_id 1 --num_shards 8 &
  ... (shard_id 2..7)
  wait
  python llm_judge_eval.py --eval_responses_dir /path/to/eval_responses --merge_only
"""

import argparse
import json
import os
import re
import time
from typing import List, Tuple

import vllm


# ---------------------------------------------------------------------------
# Judge prompt and parsing
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = (
    "You are an answer correctness judge. Given a question, the gold (correct) answer, "
    "and the model's answer, determine if the model's answer is correct: equivalent to the "
    "gold answer or semantically the same. Consider numeric equality (e.g. 14 vs 14.0), "
    "option equivalence (A vs A.), and paraphrases. Answer with exactly one word: Yes or No."
)

def build_judge_prompt(question: str, gold: str, model_answer: str, max_q: int = 400, max_a: int = 600) -> str:
    q = (question or "").strip()[:max_q]
    if len((question or "").strip()) > max_q:
        q += "..."
    g = (gold or "").strip()[:max_a]
    m = (model_answer or "").strip()[:max_a]
    if len((model_answer or "").strip()) > max_a:
        m += "..."
    return (
        f"{JUDGE_SYSTEM}\n\n"
        f"Question: {q}\n\nGold answer: {g}\n\nModel answer: {m}\n\n"
        "Is the model answer correct? Answer with exactly one word: Yes or No."
    )


def parse_judge_response(text: str) -> bool:
    """Parse judge output to True (correct) or False (incorrect)."""
    if not text:
        return False
    t = text.strip().lower()
    # Take first word or first line
    first = t.split()[0] if t.split() else ""
    if not first:
        first = t.split("\n")[0].strip().lower() if t else ""
        first = first.split()[0] if first.split() else first
    return first.startswith("yes")


# ---------------------------------------------------------------------------
# Discovery and batching
# ---------------------------------------------------------------------------
def discover_eval_files(eval_dir: str) -> List[Tuple[str, str, str]]:
    """Return list of (model_name, dataset_name, jsonl_path). Skips .shard*.jsonl."""
    out = []
    eval_dir = os.path.abspath(eval_dir)
    if not os.path.isdir(eval_dir):
        return out
    for name in sorted(os.listdir(eval_dir)):
        sub = os.path.join(eval_dir, name)
        if not os.path.isdir(sub):
            continue
        for fname in sorted(os.listdir(sub)):
            if not fname.endswith(".jsonl") or ".shard" in fname:
                continue
            dataset = fname[:-6]  # strip .jsonl
            out.append((name, dataset, os.path.join(sub, fname)))
    return out


def load_records(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def get_model_answer(record: dict) -> str:
    """Prefer extracted_answer; else truncate model_response."""
    ext = record.get("extracted_answer")
    if ext is not None and str(ext).strip():
        return str(ext).strip()
    resp = record.get("model_response") or ""
    return resp.strip()[:600]


def merge_shard_files(eval_dir: str, output_file: str) -> None:
    """Combine llm_accuracy_summary.shard*.jsonl into a single llm_accuracy_summary.jsonl."""
    import glob
    base = os.path.join(eval_dir, "llm_accuracy_summary")
    pattern = base + ".shard*.jsonl"
    shard_files = sorted(glob.glob(pattern))
    if not shard_files:
        print(f"No shard files found matching {pattern}")
        return
    rows = []
    for path in shard_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    rows.sort(key=lambda r: (r.get("model", ""), r.get("dataset", "")))
    with open(output_file, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Merged {len(shard_files)} shards ({len(rows)} rows) -> {output_file}")
    for path in shard_files:
        try:
            os.remove(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# vLLM judge
# ---------------------------------------------------------------------------
def run_judge_batch(
    llm: vllm.LLM,
    prompts: List[str],
    sampling_params: vllm.SamplingParams,
    chunk_size: int = 64,
) -> List[str]:
    """Run judge in chunks to avoid OOM."""
    results = []
    for i in range(0, len(prompts), chunk_size):
        chunk = prompts[i : i + chunk_size]
        outputs = llm.generate(chunk, sampling_params)
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            results.append(text)
    return results


def main():
    parser = argparse.ArgumentParser(description="LLM judge on eval_responses, output llm_accuracy_summary.jsonl")
    parser.add_argument("--eval_responses_dir", type=str, required=True,
                        help="Path to eval_responses (contains model subdirs with dataset .jsonl files)")
    parser.add_argument("--judge_model", type=str, default=None,
                        help="vLLM model path for the judge (e.g. Qwen2.5-7B-Instruct). Not needed for --merge_only.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output jsonl path (default: <eval_responses_dir>/llm_accuracy_summary.jsonl)")
    parser.add_argument("--only_model", type=str, default=None,
                        help="If set, only run on this model subdir (e.g. solver_v1_step5)")
    parser.add_argument("--only_dataset", type=str, default=None,
                        help="If set, only run on this dataset (e.g. ChartQA)")
    parser.add_argument("--chunk_size", type=int, default=64,
                        help="Batch size for judge generation")
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--shard_id", type=int, default=0,
                        help="This worker's shard index (0-based). Use with num_shards for multi-GPU.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of data-parallel shards (e.g. 8 for 8 GPUs).")
    parser.add_argument("--merge_only", action="store_true",
                        help="Only merge existing llm_accuracy_summary.shard*.jsonl into llm_accuracy_summary.jsonl.")
    args = parser.parse_args()

    eval_dir = os.path.abspath(args.eval_responses_dir)
    output_file = args.output_file or os.path.join(eval_dir, "llm_accuracy_summary.jsonl")

    # Merge-only mode: combine shard files and exit
    if args.merge_only:
        merge_shard_files(eval_dir, output_file)
        return

    if not args.judge_model:
        parser.error("--judge_model is required unless --merge_only is set.")

    # Discover files and assign this shard's subset
    files = discover_eval_files(eval_dir)
    if args.only_model:
        files = [(m, d, p) for m, d, p in files if m == args.only_model]
    if args.only_dataset:
        files = [(m, d, p) for m, d, p in files if d == args.only_dataset]
    if args.num_shards > 1:
        files = [f for i, f in enumerate(files) if i % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(files)} (model, dataset) files.")
    if not files:
        print("No eval JSONL files found for this shard.")
        return

    # When sharding, write to shard-specific output so merge can combine later
    if args.num_shards > 1:
        base, ext = os.path.splitext(output_file)
        output_file = f"{base}.shard{args.shard_id}{ext}"

    print(f"Found {len(files)} (model, dataset) files. Loading all records and building judge prompts...")
    all_prompts = []
    meta = []  # (model_name, dataset, model_path, index_in_batch, count_for_this_file)

    for model_name, dataset_name, path in files:
        records = load_records(path)
        if not records:
            continue
        model_path = records[0].get("model_path", "")
        start_idx = len(all_prompts)
        for r in records:
            q = r.get("question", "")
            gold = r.get("ground_truth", "")
            model_ans = get_model_answer(r)
            prompt = build_judge_prompt(q, gold, model_ans)
            all_prompts.append(prompt)
        meta.append((model_name, dataset_name, model_path, start_idx, len(records)))

    if not all_prompts:
        print("No records to judge.")
        return

    # vLLM judge
    print(f"Running LLM judge on {len(all_prompts)} samples (model={args.judge_model})...")
    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["\n"],
    )
    llm = vllm.LLM(model=args.judge_model, trust_remote_code=True)
    t0 = time.time()
    judge_outputs = run_judge_batch(llm, all_prompts, sampling_params, chunk_size=args.chunk_size)
    elapsed = time.time() - t0
    print(f"Judge finished in {elapsed:.1f}s.")

    # Aggregate per (model, dataset)
    summary_rows = []
    for model_name, dataset_name, model_path, start_idx, count in meta:
        correct = 0
        for i in range(count):
            raw = judge_outputs[start_idx + i]
            if parse_judge_response(raw):
                correct += 1
        total = count
        acc = round(correct / total * 100, 2) if total > 0 else 0.0
        row = {
            "model": model_name,
            "model_path": model_path,
            "dataset": dataset_name,
            "accuracy": acc,
            "correct": correct,
            "total": total,
        }
        summary_rows.append(row)
        print(f"  {model_name} / {dataset_name}: {correct}/{total} = {acc:.2f}% (LLM judge)")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        for row in summary_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {output_file}")


if __name__ == "__main__":
    main()
