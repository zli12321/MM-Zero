#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for ImageFree Self-Play.

Similar to evaluate.py but works with rendered images from code_render/.
Input: rendered image shards (from code_render/render_code.py) with hard questions.
Process: Use the Solver to generate answers, then grade via majority voting.
Output: Scored results with silver labels for Solver training.

Setup:
    pip install stopit transformers torch vllm

Example Usage:
    CUDA_VISIBLE_DEVICES=0 python evaluate_imagefree.py --model "path/to/solver" --suffix 0 --save_name "exp_name"
"""

import json
import vllm
from transformers import AutoTokenizer
import argparse
import re
import os
import stopit
from mathruler.grader import extract_boxed_content, grade_answer
import base64
import math
from io import BytesIO
from PIL import Image

# Refuse to decode images that would exceed this pixel count (avoids OOM and decompression bomb warning).
# Oversized images are skipped in b64_to_image. We resize loaded images to ~1.6M pixels before vLLM anyway.
Image.MAX_IMAGE_PIXELS = int(256e6)  # 256M pixels; ~16000x16000 max. Larger images raise and are skipped.

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Evaluate rendered images with hard questions using vLLM Solver.")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="Path to the Solver model.")
parser.add_argument("--num_samples", type=int, default=9,
                    help="Number of candidate answers to generate per question (n).")
parser.add_argument("--suffix", type=str, default="0",
                    help="A unique suffix for file naming (usually GPU index).")
parser.add_argument("--save_name", type=str, required=True,
                    help="Experiment name for input/output files.")
args = parser.parse_args()

# --- Constants and Paths ---
STORAGE_PATH = os.getenv("STORAGE_PATH")
# Input: rendered image shards from code_render/render_code.py
INPUT_FILE = f"{STORAGE_PATH}/rendered_images/{args.save_name}_{args.suffix}.json"
# Output: scored results
OUTPUT_FILE = f"{STORAGE_PATH}/rendered_images/{args.save_name}_{args.suffix}_results.json"


# --- Timeout-Protected Grading Function ---
@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """Wraps grade_answer with a timeout. Returns 'TIMED_OUT' if it takes too long."""
    return grade_answer(res1, res2)


# --- Main Script Logic ---

# 1. Load and Prepare Data
print(f"[{args.suffix}] EVALUATE: loading data from {INPUT_FILE}", flush=True)
try:
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"[{args.suffix}] ERROR: Input file not found: {INPUT_FILE}. Exiting.")
    # Write empty results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump([], f)
    exit()

# Extract fields — using hard_question for Solver evaluation
questions = [item.get("hard_question", "") for item in data]
answers = [item.get("hard_answer", "") for item in data]
easy_questions = [item.get("easy_question", "") for item in data]
easy_answers = [item.get("easy_answer", "") for item in data]
captions = [item.get("caption", "") for item in data]
images_base64 = [item.get("image", "") for item in data]

# Filter out items with missing fields and validate base64 images
# Only keep items with valid base64 strings (at least 100 chars, looks like base64)
def is_valid_base64_image(b64_str):
    """Check if string looks like a valid base64-encoded image."""
    if not b64_str or not isinstance(b64_str, str):
        return False
    b64_str = b64_str.strip()
    # Base64 images should be reasonably long (at least 100 chars for a minimal image)
    # and either start with data:image or be a long base64 string
    return (
        len(b64_str) > 100 and
        (b64_str.startswith("data:image") or len(b64_str) > 200)
    )

filtered = [
    (q, a, eq, ea, cap, img)
    for q, a, eq, ea, cap, img in zip(questions, answers, easy_questions, easy_answers, captions, images_base64)
    if q and is_valid_base64_image(img)
]

if not filtered:
    print(f"[{args.suffix}] No valid items found. Exiting.")
    with open(OUTPUT_FILE, "w") as f:
        json.dump([], f)
    exit()

questions, answers, easy_questions, easy_answers, captions, images_base64 = zip(*filtered)
print(f"[{args.suffix}] Found {len(questions)} valid items to process.")

# 2. Initialize Model and Tokenizer
print(f"[{args.suffix}] Initializing vLLM for model: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = vllm.LLM(
    model=args.model,
    tokenizer=args.model,
    gpu_memory_utilization=0.85,
    seed=int(args.suffix),
)
sample_params = vllm.SamplingParams(
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=args.num_samples,
)

# 3. Generate Responses
print(f"[{args.suffix}] EVALUATE: Generating {args.num_samples} candidate answers per question.", flush=True)

placeholder = "<|image_pad|>"
# Aligned with solver.jinja: single user message, no system prompt,
# <think> tags for reasoning, \boxed{} for answer.
prompts = [
    (
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
        f"{q}\n\n"
        "Look at the image carefully and answer the question. "
        "First, think step by step inside <think> </think> tags. "
        "Then, give your final answer inside \\boxed{{}} as a single number, "
        "single word, or short phrase only "
        "(e.g. \\boxed{{42}}, \\boxed{{blue}}, \\boxed{{Q1}})"
        "—no units, no full sentences."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    for q in questions
]

# Qwen2.5-VL limits: aspect ratio must be *strictly* < 200; keep total pixels bounded
MAX_IMAGE_PIXELS = 1280 * 1280  # ~1.6M pixels
MAX_ASPECT_RATIO = 199  # processor requires "smaller than 200", so use 199 to be safe


def _resize_for_vl(img: Image.Image) -> Image.Image:
    """Resize image to satisfy max pixels and max aspect ratio for Qwen2.5-VL."""
    w, h = img.size
    if w == 0 or h == 0:
        return img
    # Cap aspect ratio first (Qwen2.5-VL requires strictly < 200)
    ratio = max(w, h) / min(w, h)
    if ratio > MAX_ASPECT_RATIO:
        if w >= h:
            w, h = int(h * MAX_ASPECT_RATIO), h
        else:
            w, h = w, int(w * MAX_ASPECT_RATIO)
        w, h = max(1, w), max(1, h)
    # Then cap total pixels (same scale preserves aspect ratio)
    total = w * h
    if total > MAX_IMAGE_PIXELS:
        scale = math.sqrt(MAX_IMAGE_PIXELS / total)
        w, h = max(1, int(w * scale)), max(1, int(h * scale))
        # Ensure we didn't push ratio over 199 due to rounding
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


# Decode base64 images to PIL Images
def b64_to_image(b64_str):
    """Decode base64 string to PIL Image. Returns None if decoding fails or image is too large."""
    if not b64_str:
        return None
    try:
        # Handle data:image/png;base64,<data> format
        b64_data = b64_str.split(",")[1] if "," in b64_str else b64_str
        img_bytes = base64.b64decode(b64_data)
        if len(img_bytes) == 0:
            return None
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        # Verify image is valid (has dimensions)
        if img.width == 0 or img.height == 0:
            return None
        # Resize to satisfy VL model limits (aspect ratio <= 200, bounded pixels)
        img = _resize_for_vl(img)
        # Final validation: ensure we never pass an invalid image to vLLM (avoids worker crash)
        w, h = img.size
        if w == 0 or h == 0:
            return None
        if max(w, h) / min(w, h) >= 200:
            print(f"[{args.suffix}] WARNING: Dropping image with aspect ratio >= 200 after resize ({w}x{h})")
            return None
        return img
    except Exception as e:
        err_msg = str(e)
        if "DecompressionBomb" in type(e).__name__ or "decompression" in err_msg.lower():
            print(f"[{args.suffix}] WARNING: Image too large (exceeds MAX_IMAGE_PIXELS); skipping.")
        else:
            print(f"[{args.suffix}] WARNING: Failed to decode base64 image (len={len(b64_str)}): {err_msg[:100]}")
        return None


def _generate_chunked(model, valid_chats, valid_indices, sampling_params, chunk_size=64, suffix="0", use_tqdm=False):
    """
    Generate in chunks. If a chunk fails (e.g. one image triggers processor error),
    retry that chunk one-by-one and skip failing items so the worker does not die.
    Returns (responses_list, valid_indices_list) with same length; failed items are excluded.
    use_tqdm: set True (e.g. via EVALUATE_IMAGEFREE_TQDM=1) only when running a single worker.
    """
    all_responses = []
    all_indices = []
    chunk_size = max(1, chunk_size)
    total_chunks = (len(valid_chats) + chunk_size - 1) // chunk_size

    for start in range(0, len(valid_chats), chunk_size):
        chunk_num = start // chunk_size + 1
        items_this_chunk = min(chunk_size, len(valid_chats) - start)
        items_so_far = start + items_this_chunk
        print(f"[{suffix}] EVALUATE chunk {chunk_num}/{total_chunks} ({items_this_chunk} items, {items_so_far}/{len(valid_chats)} total)...", flush=True)
        chunk_chats = valid_chats[start : start + chunk_size]
        chunk_indices = valid_indices[start : start + chunk_size]
        try:
            chunk_responses = model.generate(chunk_chats, sampling_params=sampling_params, use_tqdm=use_tqdm)
            all_responses.extend(chunk_responses)
            all_indices.extend(chunk_indices)
        except (ValueError, Exception) as e:
            # One or more items in the chunk caused an error (e.g. aspect ratio); process one-by-one
            print(f"[{args.suffix}] WARNING: Chunk failed ({str(e)[:80]}), retrying one-by-one...", flush=True)
            for i, (chat, idx) in enumerate(zip(chunk_chats, chunk_indices)):
                try:
                    single = model.generate([chat], sampling_params=sampling_params, use_tqdm=use_tqdm)
                    all_responses.extend(single)
                    all_indices.append(idx)
                except (ValueError, Exception) as e2:
                    print(f"[{args.suffix}] WARNING: Skipping item {idx} (processor error): {str(e2)[:80]}", flush=True)
                    continue
        print(f"[{suffix}] EVALUATE progress: chunk {chunk_num}/{total_chunks} done ({len(all_responses)}/{len(valid_chats)} items).", flush=True)
    return all_responses, all_indices


images_pil = [b64_to_image(b64) for b64 in images_base64]

# Prepare valid chats with prompts and images
# Only process items with successfully decoded images
valid_chats = []
valid_indices = []
for idx, (prompt, img) in enumerate(zip(prompts, images_pil)):
    if img is not None:
        valid_chat = {
            "prompt": prompt,
            "multi_modal_data": {"image": img}
        }
        valid_chats.append(valid_chat)
        valid_indices.append(idx)

print(f"[{args.suffix}] EVALUATE: {len(valid_chats)} items have valid images. Starting generation.", flush=True)

# Generate responses using vLLM (chunked + fallback so one bad image doesn't kill the worker)
# Progress: print per chunk (safe for multi-worker). Set EVALUATE_IMAGEFREE_TQDM=1 for vLLM tqdm (single-worker only).
CHUNK_SIZE = 64
use_tqdm = os.environ.get("EVALUATE_IMAGEFREE_TQDM", "").strip().lower() in ("1", "true", "yes")
responses, valid_indices = _generate_chunked(
    model, valid_chats, valid_indices, sample_params, chunk_size=CHUNK_SIZE, suffix=args.suffix, use_tqdm=use_tqdm
)
if len(responses) < len(valid_chats):
    print(f"[{args.suffix}] NOTE: {len(valid_chats) - len(responses)} items skipped due to processor errors.")
print(f"[{args.suffix}] EVALUATE: Generation complete ({len(responses)} responses).", flush=True)

# 4. Process and Grade Responses
results_all = []
print(f"[{args.suffix}] EVALUATE: Grading responses...", flush=True)
for resp_idx, (response, orig_idx) in enumerate(zip(responses, valid_indices)):
    golden_answer = answers[orig_idx]
    question = questions[orig_idx]
    image_b64 = images_base64[orig_idx]
    caption = captions[orig_idx]
    easy_q = easy_questions[orig_idx]
    easy_a = easy_answers[orig_idx]

    try:
        # Extract boxed content from all generated samples
        results = [extract_boxed_content(output.text) for output in response.outputs]
        results = [res for res in results if res]

        if not results:
            print(f"[{args.suffix}] WARNING: No valid boxed answers for: '{question[:50]}...'")
            continue

        # Majority voting with answer consolidation
        answer_counts = {}
        for result in results:
            matched = False
            for existing_answer in answer_counts:
                if result == existing_answer or ('no ' in result.lower() and 'no ' in existing_answer.lower()):
                    answer_counts[existing_answer] += 1
                    matched = True
                    break

                match_1 = grade_answer_with_timeout(result, existing_answer, timeout=10)
                if match_1 == 'TIMED_OUT':
                    continue
                if match_1:
                    answer_counts[existing_answer] += 1
                    matched = True
                    break

                match_2 = grade_answer_with_timeout(existing_answer, result, timeout=10)
                if match_2 == 'TIMED_OUT':
                    continue
                if match_2:
                    answer_counts[existing_answer] += 1
                    matched = True
                    break

            if not matched:
                answer_counts[result] = 1

        if not answer_counts:
            continue

        majority_answer = max(answer_counts, key=answer_counts.get)
        max_count = answer_counts[majority_answer]
        score = max_count / len(results)

        # Only save results with valid images (same "image" key as render shards for upload_imagefree)
        if image_b64 and is_valid_base64_image(image_b64):
            results_all.append({
                "question": question,
                "answer": majority_answer,
                "score": score,
                "image": image_b64 if isinstance(image_b64, str) else image_b64.decode("utf-8"),
                "caption": caption,
                "easy_question": easy_q,
                "easy_answer": easy_a,
                "hard_question": question,
                "hard_answer": golden_answer,
                "question_type": "numerical",
                "results": results,
            })
        else:
            print(f"[{args.suffix}] WARNING: Skipping result with invalid image for question: '{question[:50]}...'")

    except Exception as e:
        print(f"[{args.suffix}] CRITICAL ERROR: '{question[:50]}...': {e}")
        continue

# 5. Save Final Results
print(f"[{args.suffix}] Processed {len(results_all)} items. Saving to: {OUTPUT_FILE}")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(results_all, f, indent=4)

print(f"[{args.suffix}] EVALUATE: Script finished.", flush=True)
