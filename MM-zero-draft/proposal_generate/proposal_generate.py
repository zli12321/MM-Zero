#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Proposal Generator for ImageFree Self-Play.

This script uses a VLM (text-only mode) to generate chart/visualization proposals
consisting of a caption, an easy question, and a hard question.
No images are needed as input — the model imagines the visualization from text prompts.

Modeled on question_generate/question_generate.py but without image inputs.
"""

import vllm
import torch
from transformers import AutoTokenizer
import argparse
from typing import List
from vllm.outputs import RequestOutput
import sys
import os
import json
import regex as re
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

STORAGE_PATH = os.getenv("STORAGE_PATH")

# ----------------------------- Prompt (aligned with proposer.jinja) --------- #
# Load proposer.jinja so inference uses the exact same prompt the proposer
# model was RL-fine-tuned on.  The template expects {{ content | trim }}
# which is the seed topic (same format as the 'problem' column in training parquet).

_JINJA_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'format_prompt', 'proposer.jinja'
)

def _load_proposer_template() -> str:
    with open(_JINJA_PATH, 'r') as f:
        return f.read()

_PROPOSER_TEMPLATE: str = _load_proposer_template()


def _render_proposer_jinja(seed_topic: str) -> str:
    """Render the proposer.jinja template with the given seed topic,
    producing the full user-message the model was trained on."""
    return _PROPOSER_TEMPLATE.replace('{{ content | trim }}', seed_topic.strip())

# ----------------------------- Parsing ----------------------------- #

def extract_proposal(response: str):
    """Extract structured proposal fields from the model response.
    
    Tries multiple parsing strategies:
    1. XML tags with closing tags: <caption>...</caption>
    2. XML tags without closing tags: <caption>...
    3. Label format: caption: ...
    """
    def extract_field(field_name, response_text):
        """Try multiple patterns to extract a field."""
        patterns = [
            # XML with closing tag
            rf"<{field_name}>([\s\S]*?)</{field_name}>",
            # XML without closing tag (greedy until next tag or end)
            rf"<{field_name}>([\s\S]*?)(?=<[^/]|$)",
            # Label format: "caption: ..." or "caption:..."
            rf"(?:^|\n)\s*{field_name}\s*:\s*([^\n<]+)",
            # Label format with angle brackets: "caption: <...>"
            rf"(?:^|\n)\s*{field_name}\s*:\s*<([^>]+)>",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.MULTILINE | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content and len(content) > 0:
                    return content
        return None
    
    # Extract all fields
    caption = extract_field("caption", response)
    easy_question = extract_field("easy_question", response)
    easy_answer = extract_field("easy_answer", response)
    hard_question = extract_field("hard_question", response)
    hard_answer = extract_field("hard_answer", response)
    
    # Extract visual_type (default to matplotlib if missing)
    visual_type = extract_field("visual_type", response)
    if visual_type:
        visual_type = visual_type.strip().lower()
        if visual_type not in ("matplotlib", "plotly", "pillow", "svg"):
            visual_type = "matplotlib"
    else:
        visual_type = "matplotlib"

    # Require at least caption, easy_question, and easy_answer (needed for codegen training)
    if caption and easy_question and easy_answer:
        return {
            "visual_type": visual_type,
            "caption": caption,
            "easy_question": easy_question,
            "easy_answer": easy_answer,
            "hard_question": hard_question or "",
            "hard_answer": hard_answer or "",
        }
    return None


# ----------------------------- Seed Topics ----------------------------- #

# SEED_TOPICS = [
#     "population statistics", "GDP comparison across countries", "temperature trends over decades",
#     "company revenue quarterly report", "student exam score distribution", "stock price movements",
#     "website traffic analytics", "COVID-19 case trends", "election voting results",
#     "energy consumption by source", "carbon emissions by country", "rainfall monthly data",
#     "product sales by category", "employee satisfaction survey", "age distribution of a city",
#     "internet usage worldwide", "food production by region", "transportation mode usage",
#     "housing price trends", "literacy rates across nations", "Olympic medal counts",
#     "social media user growth", "budget allocation pie chart", "crop yield comparison",
#     "air quality index over time", "hospital admission rates", "university enrollment numbers",
#     "battery technology improvements", "mobile phone market share", "global trade volume",
#     "species population decline", "water usage by sector", "crime rate trends",
#     "vaccine distribution progress", "renewable energy adoption", "inflation rate comparison",
#     "unemployment trends", "birth and death rates", "forest coverage changes",
#     "satellite launch frequency", "programming language popularity", "movie box office earnings",
#     "airline passenger volume", "e-commerce growth rates", "smartphone screen size trends",
#     "coffee consumption per capita", "earthquake frequency by region", "patent filings by country",
#     "average commute time by city", "global internet speed rankings",
# ]

SEED_TOPICS = [
    # Data & Information Graphics (Matplotlib / Plotly)
    "chart and graph understanding",
    "statistical and trend analysis",
    
    # Mathematics & Geometry (Matplotlib / Pillow / SVG)
    "mathematical and quantitative reasoning",
    "spatial and geometric reasoning",
    
    # Logic & Abstraction (SVG / Pillow)
    "logical and abstract reasoning",
    "diagram and structural comprehension",
    
    # Basic Perception & Reading (Pillow)
    "visual text recognition and OCR",
    "basic shape and object perception"
]


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Use lower gpu_memory_util and max_model_len so this works when GPUs have limited free
    # memory (e.g. right after training: only ~40 GiB free on 80GB). Override with env if needed.
    gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.5"))
    max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "32768"))
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        seed=int(args.suffix),
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
    )

    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # Build prompts using the same format as proposer GRPO training:
    #   seed_topic → proposer.jinja → single user message
    # This matches the training distribution exactly.
    prompts = []
    for i in range(args.num_samples):
        topic = SEED_TOPICS[i % len(SEED_TOPICS)]
        user_msg = _render_proposer_jinja(topic)

        prompt = (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompts.append(prompt)

    print(f"[{args.suffix}] Generating {len(prompts)} proposals...")

    # Generate all at once (text-only, no multi_modal_data)
    completions: List[RequestOutput] = model.generate(prompts, sampling_params=sample_params)

    results = []
    for i, completion in enumerate(completions):
        response = completion.outputs[0].text
        try:
            proposal = extract_proposal(response)
            if proposal:
                results.append(proposal)
            else:
                print(f"[{args.suffix}] WARNING: Could not parse proposal {i}: {response[:100]}...")
                results.append({
                    "caption": response,
                    "easy_question": "",
                    "easy_answer": "",
                    "hard_question": "",
                    "hard_answer": "",
                })
        except Exception as e:
            print(f"[{args.suffix}] ERROR processing response {i}: {e}")
            results.append({
                "caption": response,
                "easy_question": "",
                "easy_answer": "",
                "hard_question": "",
                "hard_answer": "",
            })

    # Save results as JSON
    output_file = f"{STORAGE_PATH}/generated_proposals/{args.save_name}_{args.suffix}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[{args.suffix}] Generated {len(results)} proposals, saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chart/visualization proposals (text-only).")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model name or path")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of proposals to generate")
    parser.add_argument("--suffix", type=str, default="0", help="Suffix for output file (usually GPU index)")
    parser.add_argument("--save_name", type=str, default="proposals", help="Base name for output file")
    args = parser.parse_args()

    main(args)
