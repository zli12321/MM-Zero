#!/usr/bin/env python3
"""
Generate model responses on HuggingFace evaluation datasets using vLLM.

Supports data-parallel sharding: run N copies of this script (one per GPU),
each with a different --shard_id, to split the dataset N ways.

Usage:
    # Single GPU (no sharding):
    CUDA_VISIBLE_DEVICES=0 python eval_generate.py \
        --model_path /path/to/model --save_name base --datasets zli12321/MMSI

    # 8-way data parallel (run by run_eval_all.sh):
    CUDA_VISIBLE_DEVICES=3 python eval_generate.py \
        --model_path /path/to/model --save_name base --datasets zli12321/MMSI \
        --shard_id 3 --num_shards 8
"""

import argparse
import io
import json
import math
import os
import re
import time
from typing import Optional

import vllm
from datasets import load_dataset
from PIL import Image
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate evaluation responses with vLLM.")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--save_name", type=str, required=True,
                    help="Name for this model run (used in output directory).")
parser.add_argument("--datasets", type=str, nargs="+", required=True,
                    help="HuggingFace dataset IDs (e.g. zli12321/MMSI).")
parser.add_argument("--output_dir", type=str,
                    default="/workspace/selfAgent_Storage_qwen3vl_4b/eval_responses")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--n", type=int, default=1,
                    help="Number of response samples per question.")
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--gpu_mem_util", type=float, default=0.85)
parser.add_argument("--max_model_len", type=int, default=16384)
parser.add_argument("--chunk_size", type=int, default=256,
                    help="Batch chunk size for generation.")
parser.add_argument("--shard_id", type=int, default=0,
                    help="This worker's shard index (0-based).")
parser.add_argument("--num_shards", type=int, default=1,
                    help="Total number of data-parallel shards.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------
MAX_IMAGE_PIXELS = 1280 * 1280
MAX_ASPECT_RATIO = 199


def resize_for_vl(img: Image.Image) -> Image.Image:
    """Resize to satisfy Qwen VL constraints (aspect ratio < 200, bounded pixels)."""
    w, h = img.size
    if w == 0 or h == 0:
        return img
    ratio = max(w, h) / min(w, h)
    if ratio > MAX_ASPECT_RATIO:
        if w >= h:
            w = int(h * MAX_ASPECT_RATIO)
        else:
            h = int(w * MAX_ASPECT_RATIO)
        w, h = max(1, w), max(1, h)
    total = w * h
    if total > MAX_IMAGE_PIXELS:
        scale = math.sqrt(MAX_IMAGE_PIXELS / total)
        w, h = max(1, int(w * scale)), max(1, int(h * scale))
        ratio = max(w, h) / min(w, h)
        if ratio > MAX_ASPECT_RATIO:
            if w >= h:
                w = int(h * MAX_ASPECT_RATIO)
            else:
                h = int(w * MAX_ASPECT_RATIO)
            w, h = max(1, w), max(1, h)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    return img


def prepare_image(img) -> Optional[Image.Image]:
    """Convert a dataset image to a valid RGB PIL Image.
    Accepts PIL Image or HuggingFace image dict with 'bytes' or 'path'."""
    try:
        if isinstance(img, dict):
            if "bytes" in img:
                img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            elif "path" in img:
                img = Image.open(img["path"]).convert("RGB")
            else:
                return None
        elif not isinstance(img, Image.Image):
            return None
        else:
            img = img.convert("RGB")
        if img.width == 0 or img.height == 0:
            return None
        img = resize_for_vl(img)
        w, h = img.size
        if w == 0 or h == 0 or max(w, h) / min(w, h) >= 200:
            return None
        return img
    except Exception as e:
        print(f"  WARNING: Failed to process image: {e}")
        return None


# ---------------------------------------------------------------------------
# Prompt construction — mirrors solver.jinja
# ---------------------------------------------------------------------------
VISION_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"

SOLVER_INSTRUCTION = (
    "Please reason step by step based on the question and image. "
    "Put your final answer inside \\boxed{} as a single number, single word, "
    "or short phrase only (e.g. \\boxed{42}, \\boxed{blue}, \\boxed{Q1})"
    "\u2014no units, no full sentences, so the grader can match it."
)


def build_prompt(problem_text: str, num_images: int) -> str:
    content = problem_text.strip()
    if "<image>" in content:
        body = content.replace("<image>", VISION_TOKEN)
    else:
        body = VISION_TOKEN * num_images + content
    body = f"{body} {SOLVER_INSTRUCTION}"
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{body}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
def extract_boxed(text: str) -> Optional[str]:
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else None


def normalize_answer(ans: str) -> str:
    return ans.strip().lower().rstrip(".")


# ---------------------------------------------------------------------------
# Chunked generation with fallback
# ---------------------------------------------------------------------------
def _engine_is_dead(e: Exception) -> bool:
    msg = str(e).lower()
    return "died unexpectedly" in msg or "shutting down" in msg or "engine core" in msg


def generate_chunked(model, inputs, sampling_params, chunk_size=256):
    all_outputs = []
    total = len(inputs)
    n_chunks = (total + chunk_size - 1) // chunk_size

    for start in range(0, total, chunk_size):
        chunk_num = start // chunk_size + 1
        chunk = inputs[start:start + chunk_size]
        print(f"  [shard {args.shard_id}] chunk {chunk_num}/{n_chunks} "
              f"({start + len(chunk)}/{total})...", flush=True)
        try:
            outputs = model.generate(chunk, sampling_params=sampling_params)
            all_outputs.extend(outputs)
        except Exception as e:
            if _engine_is_dead(e):
                print(f"  FATAL: Engine died. Marking remaining as None.", flush=True)
                all_outputs.extend([None] * (total - len(all_outputs)))
                return all_outputs
            print(f"  WARNING: Chunk failed ({str(e)[:100]}), retrying one-by-one...", flush=True)
            for item in chunk:
                try:
                    out = model.generate([item], sampling_params=sampling_params)
                    all_outputs.extend(out)
                except Exception as e2:
                    if _engine_is_dead(e2):
                        print(f"  FATAL: Engine died. Marking remaining as None.", flush=True)
                        all_outputs.extend([None] * (total - len(all_outputs)))
                        return all_outputs
                    print(f"  WARNING: Skipping item ({str(e2)[:80]})", flush=True)
                    all_outputs.append(None)
    return all_outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"=== Eval Generate [shard {args.shard_id}/{args.num_shards}] ===")
    print(f"Model: {args.model_path}")
    print(f"Save name: {args.save_name}")
    print(f"Datasets: {args.datasets}")
    print()

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = vllm.LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        tensor_parallel_size=1,
        limit_mm_per_prompt={"image": 10},
        trust_remote_code=True,
    )
    sampling_params = vllm.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    print("Model loaded.\n", flush=True)

    for ds_id in args.datasets:
        short_name = ds_id.split("/")[-1]
        print(f"--- Dataset: {ds_id} ({short_name}) [shard {args.shard_id}] ---", flush=True)

        try:
            ds = load_dataset(ds_id, split=args.split)
        except Exception as e:
            print(f"  ERROR loading dataset {ds_id}: {e}", flush=True)
            print(f"  Skipping {short_name}.\n", flush=True)
            continue
        total_len = len(ds)

        # Shard the dataset
        shard_indices = list(range(args.shard_id, total_len, args.num_shards))
        print(f"  Total: {total_len}, this shard: {len(shard_indices)} samples.", flush=True)

        vllm_inputs = []
        valid_indices = []

        for idx in shard_indices:
            item = ds[idx]
            problem = item.get("problem", "")
            raw_images = item.get("images", [])
            if not isinstance(raw_images, list):
                raw_images = [raw_images]

            images = [prepare_image(img) for img in raw_images]
            images = [img for img in images if img is not None]

            if not images:
                continue

            n_tags = problem.count("<image>")
            if n_tags > 0 and n_tags != len(images):
                if n_tags < len(images):
                    images = images[:n_tags]
                else:
                    while problem.count("<image>") > len(images):
                        last = problem.rfind("<image>")
                        problem = problem[:last] + problem[last + 7:]

            prompt = build_prompt(problem, len(images))
            img_data = images if len(images) > 1 else images[0]

            vllm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": img_data},
            })
            valid_indices.append(idx)

        print(f"  {len(vllm_inputs)} valid samples. Generating...", flush=True)
        t0 = time.time()
        outputs = generate_chunked(llm, vllm_inputs, sampling_params, chunk_size=args.chunk_size)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s.\n", flush=True)

        # Save shard results
        save_dir = os.path.join(args.output_dir, args.save_name)
        os.makedirs(save_dir, exist_ok=True)
        if args.num_shards > 1:
            output_file = os.path.join(save_dir, f"{short_name}.shard{args.shard_id}.jsonl")
        else:
            output_file = os.path.join(save_dir, f"{short_name}.jsonl")

        correct = 0
        total_answered = 0

        with open(output_file, "w") as f:
            for out, orig_idx in zip(outputs, valid_indices):
                if out is None:
                    continue
                item = ds[orig_idx]
                ground_truth = item.get("answer", "")

                for resp in out.outputs:
                    predicted = extract_boxed(resp.text)
                    is_correct = False
                    if predicted and ground_truth:
                        is_correct = (normalize_answer(predicted)
                                      == normalize_answer(ground_truth))
                        total_answered += 1
                        if is_correct:
                            correct += 1

                    record = {
                        "dataset": short_name,
                        "index": orig_idx,
                        "question": item.get("problem", ""),
                        "ground_truth": ground_truth,
                        "model_response": resp.text,
                        "extracted_answer": predicted,
                        "correct": is_correct,
                        "model_path": args.model_path,
                        "save_name": args.save_name,
                    }
                    for key in ["question_type", "task_types", "task_id",
                                "id", "difficulty"]:
                        if key in item:
                            val = item[key]
                            if isinstance(val, (str, int, float, bool)):
                                record[key] = val
                    f.write(json.dumps(record) + "\n")

        accuracy = correct / total_answered * 100 if total_answered > 0 else 0.0
        print(f"  Saved: {output_file}")
        print(f"  Shard accuracy: {correct}/{total_answered} = {accuracy:.2f}%")
        print()

    print(f"=== Shard {args.shard_id} done ===")


if __name__ == "__main__":
    main()
