#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CodeGen vLLM Flask Server for ImageFree Self-Play.

This server generates SVG code from full proposals (caption, easy_question,
easy_answer, hard_question, hard_answer).
Used as a service during Proposer training to compute the compile reward.

Interface (same file-based pattern as start_vllm_server.py):
  Input JSON:  [{caption, easy_question, easy_answer, hard_question, hard_answer}, ...]
  Output JSON: [{caption, easy_question, easy_answer, hard_question, hard_answer, generated_codes: [code1,...,code8]}, ...]

Setup:
    pip install stopit flask vllm transformers

Usage:
    CUDA_VISIBLE_DEVICES=4 python MM-zero_final/vllm_service_init/start_codegen_server.py --port 7000 --model_path <path>
"""

from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import re
import threading
import time
import torch
from transformers import AutoTokenizer

# ------------------------- Command-Line Arguments ------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='7000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct')
parser.add_argument('--gpu_mem_util', type=float, default=0.8,
                    help='The maximum GPU memory utilization fraction for vLLM. Use 0.5-0.6 if this GPU is shared with other processes.')
parser.add_argument('--max_model_len', type=int, default=None,
                    help='Max sequence length for KV cache. Default 32768 to avoid OOM. Use 16384 if still OOM.')
parser.add_argument('--chunk_size', type=int, default=32,
                    help='Prompts per generation chunk (default 32; 32×n=128 sequences). Lower if OOM.')
args = parser.parse_args()

# ------------------------- vLLM Initialization ------------------------ #
# Without max_model_len, vLLM uses model default (e.g. 128k) which needs ~7 GiB KV and can OOM.
max_model_len = args.max_model_len if args.max_model_len is not None else 32768
print(f'[codegen_server] Loading model (max_model_len={max_model_len})...')

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
llm_kwargs = dict(
    model=args.model_path,
    tokenizer=args.model_path,
    gpu_memory_utilization=args.gpu_mem_util,
    max_model_len=max_model_len,
)
model = vllm.LLM(**llm_kwargs)

sample_params = vllm.SamplingParams(
    max_tokens=4096,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=4,  # Generate 4 code samples per proposal (fewer = faster; 8 = more diversity)
)

# ---------------------- GPU Idle Utilization Thread ---------------------- #
stop_event = threading.Event()
pause_event = threading.Event()


def gpu_idle_worker():
    """Keep GPU busy when idle to prevent power state drops."""
    print('[codegen_idle_worker] Started.')
    running = True
    while not stop_event.is_set():
        if pause_event.is_set():
            if running:
                print('[codegen_idle_worker] Paused.')
                running = False
            time.sleep(0.1)
            continue
        else:
            if not running:
                print('[codegen_idle_worker] Resumed.')
                running = True
        try:
            a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            torch.matmul(a, b)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f'[codegen_idle_worker] RuntimeError: {e}. Sleeping 1s...')
            time.sleep(1)
    print('[codegen_idle_worker] Stopped.')


idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)
idle_thread.start()

# ----------------------------- Prompt (aligned with codegen.jinja) --------- #
# Load codegen.jinja so the service uses the exact same prompt the codegen
# model was RL-fine-tuned on.

_JINJA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'format_prompt', 'codegen.jinja'
)

def _load_codegen_template() -> str:
    with open(_JINJA_PATH, 'r') as f:
        return f.read()

_CODEGEN_TEMPLATE: str = _load_codegen_template()


def _build_prompt_text(item: dict) -> str:
    """Build prompt_text in the same format as codegen training parquet."""
    vt = 'svg'
    hard_q = item.get('hard_question', '') or 'N/A'
    hard_a = item.get('hard_answer', '') or 'N/A'
    return (
        f"Visual Type: {vt}\n\n"
        f"Chart Description:\n{item['caption']}\n\n"
        f"Easy Question: {item['easy_question']}\n"
        f"Easy Answer: {item['easy_answer']}\n\n"
        f"Hard Question: {hard_q}\n"
        f"Hard Answer: {hard_a}"
    )


def _render_codegen_jinja(content: str) -> str:
    """Render the codegen.jinja template with the given content."""
    return _CODEGEN_TEMPLATE.replace('{{ content | trim }}', content.strip())


def extract_code_block(response: str, visual_type: str = "svg") -> str:
    """Extract SVG markup from model response."""
    # Try ```svg ... ``` fenced block
    match = re.search(r"```svg\s*\n([\s\S]*?)```", response)
    if match:
        return match.group(1).strip()
    # Try ```xml ... ``` if it contains <svg
    match = re.search(r"```xml\s*\n([\s\S]*?)```", response)
    if match and "<svg" in match.group(1).lower():
        return match.group(1).strip()
    # Fallback: extract raw <svg ...>...</svg> from response
    match = re.search(r"(<svg[\s\S]*?</svg>)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


# ---------------------------- Flask Application --------------------------- #
app = Flask(__name__)


@app.route('/codegen', methods=['GET'])
def codegen():
    """
    Endpoint: reads a task file with captions+questions, generates code, writes results.
    Same file-based interface as start_vllm_server.py's /hello endpoint.
    """
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f'[codegen_server] Received request for task file: {name}')

    # ---------- Load Data ----------
    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    captions = [item.get('caption', '') for item in data]
    easy_questions = [item.get('easy_question', '') for item in data]
    easy_answers = [item.get('easy_answer', '') for item in data]
    hard_questions = [item.get('hard_question', '') for item in data]
    hard_answers = [item.get('hard_answer', '') for item in data]
    visual_types = ['svg'] * len(data)

    # ---------- Build Prompts (aligned with codegen.jinja training format) ----------
    prompts = []
    valid_indices = []
    for idx, (caption, easy_q, easy_a, hard_q, hard_a, vt) in enumerate(
        zip(captions, easy_questions, easy_answers, hard_questions, hard_answers, visual_types)
    ):
        if caption and easy_q:
            item = {
                'caption': caption,
                'easy_question': easy_q,
                'easy_answer': easy_a,
                'hard_question': hard_q,
                'hard_answer': hard_a,
                'visual_type': vt,
            }
            prompt_text = _build_prompt_text(item)
            user_msg = _render_codegen_jinja(prompt_text)
            prompt = (
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            prompts.append(prompt)
            valid_indices.append(idx)

    print(f'[codegen_server] {len(prompts)} valid prompts prepared.')

    # ---------- vLLM Generation (chunked to avoid OOM / EngineDeadError) ----------
    # Many prompts × n=8 can OOM; process in chunks (e.g. 16 prompts × 8 = 128 sequences per chunk).
    chunk_size = getattr(args, 'chunk_size', 16)
    responses = []
    if prompts:
        for start in range(0, len(prompts), chunk_size):
            chunk = prompts[start : start + chunk_size]
            chunk_resp = model.generate(chunk, sampling_params=sample_params, use_tqdm=True)
            responses.extend(chunk_resp)
        print(f'[codegen_server] Generation complete ({len(responses)} responses).')
    else:
        print('[codegen_server] No prompts, skipping generation.')

    # ---------- Extract Code (all n=8 samples per item) ----------
    results_all = []
    resp_idx = 0
    for idx in range(len(data)):
        vt = visual_types[idx] if idx < len(visual_types) else "svg"
        if idx in valid_indices and resp_idx < len(responses):
            response = responses[resp_idx]
            # Debug: print first raw model output and parsed code for first item
            if resp_idx == 0 and response.outputs:
                raw_first = response.outputs[0].text
                parsed_first = extract_code_block(raw_first, vt)
                raw_preview = raw_first[:600] + ("..." if len(raw_first) > 600 else "")
                print(f'[codegen_server] DEBUG: First item (visual_type={vt}) first output — raw:\n{raw_preview}')
                print(f'[codegen_server] DEBUG: First item parsed code length: {len(parsed_first)}, preview (200 chars): {repr(parsed_first[:200])}')
            resp_idx += 1
            # Extract code from all 8 outputs (use visual_type for svg vs python extraction)
            codes = [extract_code_block(out.text, vt) for out in response.outputs]
            results_all.append({
                'visual_type': vt,
                'caption': captions[idx],
                'easy_question': easy_questions[idx],
                'easy_answer': easy_answers[idx],
                'hard_question': hard_questions[idx],
                'hard_answer': hard_answers[idx],
                'generated_codes': codes,
            })
        else:
            results_all.append({
                'visual_type': vt,
                'caption': captions[idx],
                'easy_question': easy_questions[idx],
                'easy_answer': easy_answers[idx],
                'hard_question': hard_questions[idx],
                'hard_answer': hard_answers[idx],
                'generated_codes': [],
            })

    print(f'[codegen_server] All results processed.')

    # ---------- Write Results ----------
    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    pause_event.clear()
    print(f'[codegen_server] Results saved to {out_path}. Resuming idle worker.')
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})


# ------------------------- Main Application Entrypoint --------------------------- #
if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=int(args.port), threaded=True)
    finally:
        stop_event.set()
        idle_thread.join()
        print('[codegen_server] Shutdown complete.')
