# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reward function for the Code Generator model in ImageFree Self-Play.

The CodeGen model generates matplotlib code given a full proposal (caption, easy_question,
easy_answer, hard_question, hard_answer).

Reward = renderability + solvability + difficulty - penalties  (range about [-0.2, 3])

  1. Renderability (0 or 1):
     Does the generated code execute and produce a valid image?
     If NOT renderable → base 0 and a small penalty is applied (default -0.1).
     Syntax errors in the extracted code also add a small penalty (default -0.05).

  2. Solvability (0.0 to 1.0):
     Rendered image + easy_question → Solver (8 rollouts).
     Silver label = PROPOSER's easy_answer.
     Score = fraction of rollouts that match the proposer's answer.

  3. Difficulty (0 or 1):
     Rendered image + hard_question → Solver (8 rollouts).
     Silver label = SOLVER's own majority vote (self-consistency).
     If the Solver can answer the hard question at least once (score > 0),
     the hard question is reasonable/solvable → reward = 1. Otherwise 0.

The Solver is queried via the vLLM service (same as in questioner training).
"""

import ast
import re
import os
import json
import time
import random
import subprocess
import tempfile
import base64
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from mathruler.grader import extract_boxed_content, grade_answer

# Import parallel rendering utilities
try:
    from SelfAgent.code_render.render_code import render_batch_codes
except ImportError:
    from code_render.render_code import render_batch_codes

STORAGE_PATH = os.getenv("STORAGE_PATH")
if STORAGE_PATH is None:
    STORAGE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

TEMP_RESULTS_DIR = os.path.join(STORAGE_PATH, "temp_results")
os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)

# Step counter for saving example renders (incremented each reward call so step_0, step_1, ...)
_codegen_render_example_step = [0]


def _save_codegen_render_examples(
    step: int,
    rendered_images: List[Optional[str]],
    easy_questions: List[str],
    easy_answers: List[str],
    hard_questions: List[str],
    hard_answers: List[str],
    questions: List[str],
    max_examples: int = 10,
) -> None:
    """Save up to max_examples random rendered images to STORAGE_PATH/rendered_images/examples/codegen/step_{step}/.
    If fewer than max_examples images were rendered, save all of them.
    Uses a 'codegen' subfolder so these do not overwrite proposer training examples (proposer uses examples/step_N/)."""
    storage = os.environ.get("STORAGE_PATH") or "."
    base = os.path.join(storage, "rendered_images", "examples", "codegen", f"step_{step}")
    os.makedirs(base, exist_ok=True)

    # Indices that have a successfully rendered image
    indices_with_images = [i for i in range(len(rendered_images)) if rendered_images[i] is not None]
    if not indices_with_images:
        return

    # Sample up to max_examples (or all if fewer)
    n_save = min(max_examples, len(indices_with_images))
    selected_indices = random.sample(indices_with_images, n_save)

    saved = []
    for idx in selected_indices:
        img_b64 = rendered_images[idx]
        if img_b64 is None:
            continue
        easy_q = easy_questions[idx] if idx < len(easy_questions) else ""
        easy_a = easy_answers[idx] if idx < len(easy_answers) else ""
        hard_q = hard_questions[idx] if idx < len(hard_questions) else ""
        hard_a = hard_answers[idx] if idx < len(hard_answers) else ""
        # Extract caption from prompt_text (Chart Description: ...)
        caption = ""
        if idx < len(questions):
            prompt_text = questions[idx] or ""
            m = re.search(r"Chart Description:\s*([\s\S]*?)(?=\n\nEasy Question:|\n\nHard Question:|$)", prompt_text)
            if m:
                caption = m.group(1).strip()[:500]

        try:
            raw = base64.b64decode(img_b64)
            fname = f"codegen_{idx}.png"
            path = os.path.join(base, fname)
            with open(path, "wb") as f:
                f.write(raw)
            saved.append({
                "file": fname,
                "index": idx,
                "caption": caption,
                "easy_question": easy_q[:500],
                "easy_answer": easy_a,
                "hard_question": hard_q[:500],
                "hard_answer": hard_a,
            })
        except Exception as e:
            print(f"[codegen_reward] save example {idx}: {e}", flush=True)

    if saved:
        info_path = os.path.join(base, "info.json")
        with open(info_path, "w") as f:
            json.dump({"step": step, "saved_count": len(saved), "entries": saved}, f, indent=2)
        print(f"[codegen_reward] Saved {len(saved)} example images + Q&A to {base}", flush=True)


# ----------------------------- Code Extraction ----------------------------- #

# Penalty for code that does not render or has syntax errors (tunable via env)
CODEGEN_PENALTY_NO_RENDER = float(os.environ.get("CODEGEN_PENALTY_NO_RENDER", "0.1"))
CODEGEN_PENALTY_SYNTAX_ERROR = float(os.environ.get("CODEGEN_PENALTY_SYNTAX_ERROR", "0.05"))


def code_has_syntax_error(code: str) -> bool:
    """Return True if the code has a Python syntax error."""
    if not code or not code.strip():
        return True
    try:
        ast.parse(code)
        return False
    except SyntaxError:
        return True


def extract_code_block(response: str, visual_type: str = "matplotlib") -> str:
    """Extract code from a fenced block in the model output.

    Supports ```python ... ``` for matplotlib/plotly/pillow,
    and ```svg ... ``` for raw SVG markup.
    """
    if visual_type == "svg":
        # Try ```svg ... ``` first
        match = re.search(r"```svg\s*\n([\s\S]*?)```", response)
        if match:
            return match.group(1).strip()
        # Fallback: raw SVG in the response
        match = re.search(r"(<svg[\s\S]*?</svg>)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    # Python code block (matplotlib, plotly, pillow)
    match = re.search(r"```(?:python)?\s*\n([\s\S]*?)```", response)
    if match:
        return match.group(1).strip()
    # Fallback: if the entire response looks like code (starts with import)
    lines = response.strip().split('\n')
    if lines and (lines[0].startswith('import ') or lines[0].startswith('from ')):
        return response.strip()
    return ""


# ----------------------------- Visual Type Compliance ----------------------------- #

def detect_actual_visual_type(code_str: str) -> str:
    """Infer the actual visual type from the generated code content."""
    s = code_str.strip()
    if not s:
        return "matplotlib"
    if s.lstrip().startswith("<") or "<?xml" in s[:200] or "<svg" in s[:200].lower():
        return "svg"
    lower = s.lower()
    if "import plotly" in lower or "plotly." in lower or "go." in lower:
        return "plotly"
    if "from pil" in lower or "import pil" in lower or "image.new" in lower or "image.draw" in lower or "image.open" in lower:
        return "pillow"
    return "matplotlib"

CODEGEN_PENALTY_WRONG_TYPE = float(os.environ.get("CODEGEN_PENALTY_WRONG_TYPE", "0.5"))

# Visual type diversity penalty (applied per-item based on batch-level dominance)
CODEGEN_VT_DIVERSITY_MAX = float(os.environ.get("CODEGEN_VT_DIVERSITY_MAX", "0.4"))


def _detect_matplotlib_subtype(code_str: str) -> str:
    """Detect the matplotlib chart subtype from code to track intra-matplotlib diversity."""
    lower = code_str.lower()
    if "stackplot" in lower or "fill_between" in lower or "stacked" in lower:
        return "mpl_stacked_area"
    if "bar(" in lower or "barh(" in lower:
        if "bottom=" in lower or "stacked" in lower:
            return "mpl_stacked_bar"
        return "mpl_bar"
    if "scatter(" in lower:
        return "mpl_scatter"
    if "pie(" in lower:
        return "mpl_pie"
    if "hist(" in lower:
        return "mpl_histogram"
    if "imshow(" in lower or "heatmap" in lower or "pcolormesh" in lower:
        return "mpl_heatmap"
    if "contour(" in lower or "contourf(" in lower:
        return "mpl_contour"
    if "polar" in lower:
        return "mpl_polar"
    if "plot(" in lower:
        return "mpl_line"
    return "mpl_other"


# ----------------------------- Code Rendering ----------------------------- #
# Note: We use render_batch_codes for parallel rendering with pre-loaded matplotlib
# (see Step 1 in compute_score below). The render_single function is kept for
# backward compatibility but is no longer used in the main pipeline.


# ----------------------------- vLLM Solver Query ----------------------------- #

def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000)
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"


def split_list(lst, n=4):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def fetch(index, filepath):
    """Send a task file to the vLLM Solver service."""
    response = requests.get(f"http://0.0.0.0:{6000 + index}/hello?name={filepath}")
    return True


def query_solver_with_images(items: List[Dict]) -> List[Dict]:
    """
    Send items (with rendered images) to the vLLM Solver service.
    Each item needs: 'question', 'answer', 'image' (base64), 'types'.
    Returns items with 'score' field added.
    """
    if not items:
        return []

    datas = split_list(items, 4)
    random_names = [generate_temp_filename(prefix=f"codegen_{i}", suffix=".json") for i in range(4)]

    for i in range(4):
        with open(random_names[i], 'w') as f:
            json.dump(datas[i], f, indent=4)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch, i, random_names[i]) for i in range(4)]
        for future in as_completed(futures):
            future.result()

    final_results = []
    for i in range(4):
        results_file = random_names[i].replace('.json', '_results.json')
        try:
            with open(results_file, 'r') as f:
                final_results.extend(json.load(f))
            os.remove(results_file)
        except json.JSONDecodeError as e:
            print(f"[codegen_reward] WARNING: results corrupted (service {i}): {e}")
            print(f"[codegen_reward] Filling {len(datas[i])} items with score=0.0")
            final_results.extend([{"question": "", "score": 0.0}] * len(datas[i]))
            try:
                os.remove(results_file)
            except OSError:
                pass
        except FileNotFoundError:
            print(f"[codegen_reward] WARNING: results file not found: {results_file}")
            final_results.extend([{"question": "", "score": 0.0}] * len(datas[i]))

    return final_results


# ----------------------------- Prompt Parsing ----------------------------- #

def extract_field_from_prompt(prompt_text: str, field_name: str) -> str:
    """
    Extract a named field from the full prompt_text column.
    The prompt_text format is:
        Visual Type: {visual_type}

        Chart Description:
        {caption}

        Easy Question: {easy_question}
        Easy Answer: {easy_answer}

        Hard Question: {hard_question}
        Hard Answer: {hard_answer}
    """
    match = re.search(rf"{field_name}:\s*(.+?)(?:\n|$)", prompt_text)
    if match:
        return match.group(1).strip()
    return ""


# ----------------------------- Main Reward ----------------------------- #

def compute_score(
    predicts: List[str],
    ground_truths: List[str],
    questions: List[str],
    description_answers: List[str],
    format_weight: float = 0.1,
    images: Optional[List] = None,
) -> List[Dict[str, float]]:
    """
    Compute reward for each CodeGen output.

    Pipeline per prediction:
      1. Extract code and render → renderability (0 or 1)
         If not renderable → total reward = 0 (skip solver)
      2. Solvability: render + easy_q → Solver (8 rollouts) → compare vs proposer's easy_answer
         solvability = fraction of rollouts matching the proposer's answer
      3. Difficulty: render + hard_q → Solver (8 rollouts) → self-consistency
         1 if Solver can answer at least once (score > 0), else 0
      4. reward = renderability + solvability + difficulty

    Args:
        predicts: List of model outputs (code wrapped in ```python ... ```).
        ground_truths: List of easy_answer strings.
        questions: List of prompt_text strings (full proposal; fields are extracted).
        description_answers: Not used (kept for API compatibility).
        format_weight: Not used.
        images: Not used (CodeGen is text-only; images are rendered from code).

    Returns:
        List of score dicts with keys: overall, renderability, solvability, difficulty.
    """
    # Extract easy and hard questions/answers from each prompt_text
    easy_questions = [extract_field_from_prompt(q, "Easy Question") for q in questions]
    easy_answers = [extract_field_from_prompt(q, "Easy Answer") or gt
                    for q, gt in zip(questions, ground_truths)]
    hard_questions = [extract_field_from_prompt(q, "Hard Question") for q in questions]
    hard_answers = [extract_field_from_prompt(q, "Hard Answer") for q in questions]

    # Extract visual_type from each prompt_text (default matplotlib for backward compat)
    visual_types = []
    for q in questions:
        vt = extract_field_from_prompt(q, "Visual Type").lower()
        if vt not in ("matplotlib", "plotly", "pillow", "svg"):
            vt = "matplotlib"
        visual_types.append(vt)
    vt_counts = {}
    for vt in visual_types:
        vt_counts[vt] = vt_counts.get(vt, 0) + 1
    print(f'[codegen_reward] Visual type distribution: {vt_counts}')

    # ==================================================================
    # Step 1: Extract code and render images (parallel with early stopping)
    # ==================================================================
    # Extract code blocks from all predictions; detect syntax errors for penalty
    codes_to_render = []
    syntax_errors = []  # True if code has syntax error (or is empty)
    type_compliant = []  # True if the generated code matches the requested visual_type
    for predict, vt in zip(predicts, visual_types):
        # Only strip whitespace around angle brackets for Python code (helps with
        # mangled tag output); skip for SVG since it would corrupt SVG markup.
        if vt == "svg":
            predict_clean = predict
        else:
            predict_clean = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)
        code = extract_code_block(predict_clean, visual_type=vt)
        codes_to_render.append(code)
        # SVG is not Python — skip syntax check for SVG
        if vt == "svg":
            syntax_errors.append(not code.strip())
        else:
            syntax_errors.append(code_has_syntax_error(code))
        # Check visual type compliance: does the code match the requested type?
        actual_vt = detect_actual_visual_type(code) if code.strip() else vt
        type_compliant.append(actual_vt == vt)

    # Detect actual visual types and matplotlib subtypes for diversity tracking
    actual_visual_types = []
    code_subtypes = []
    for code, vt in zip(codes_to_render, visual_types):
        avt = detect_actual_visual_type(code) if code.strip() else vt
        actual_visual_types.append(avt)
        if avt == "matplotlib":
            code_subtypes.append(_detect_matplotlib_subtype(code))
        else:
            code_subtypes.append(avt)

    wrong_type_count = sum(1 for c in type_compliant if not c)
    if wrong_type_count > 0:
        print(f'[codegen_reward] Visual type compliance: {sum(type_compliant)}/{len(type_compliant)} match '
              f'({wrong_type_count} wrong type, penalty={CODEGEN_PENALTY_WRONG_TYPE})')

    # Compute batch-level diversity statistics for actual generated types
    from collections import Counter
    actual_vt_counts = Counter(actual_visual_types)
    subtype_counts = Counter(code_subtypes)
    total_in_batch = len(actual_visual_types) or 1
    actual_vt_fracs = {vt: cnt / total_in_batch for vt, cnt in actual_vt_counts.items()}
    subtype_fracs = {st: cnt / total_in_batch for st, cnt in subtype_counts.items()}
    print(f'[codegen_reward] Actual VT distribution: {dict(actual_vt_counts)}')
    print(f'[codegen_reward] Code subtype distribution: {dict(subtype_counts)}')

    # Parallel render: long-lived workers (import matplotlib once per worker) to avoid per-snippet process startup.
    max_workers = int(os.environ.get("RENDER_MAX_WORKERS", "16"))
    total_to_render = len(codes_to_render)
    print(f'[codegen_reward] Rendering {total_to_render} images in parallel (max_workers={max_workers})...')

    # Progress callback: print to training terminal every ~5% or every 25 items, and at 100%
    last_reported_pct = -1
    report_every = max(1, total_to_render // 20)  # ~20 updates over the run

    def render_progress(done: int, total: int, success: int) -> None:
        nonlocal last_reported_pct
        pct = (100 * done) // total if total else 0
        if done == total or done % report_every == 0 or pct >= last_reported_pct + 5:
            last_reported_pct = pct
            if done == total:
                print(f'[codegen_reward] Render progress: {done}/{total} images (100%) — {success} OK — COMPLETE.')
            else:
                print(f'[codegen_reward] Render progress: {done}/{total} images ({pct}%) — {success} OK (parallel, up to {max_workers} at a time)')

    # Build tasks: (code_str, visual_type)
    render_tasks = [(code, vt) for code, vt in zip(codes_to_render, visual_types)]
    
    # Use process pool by default (faster - matplotlib pre-loaded per worker)
    # Set RENDER_USE_SUBPROCESS=1 to use one subprocess per snippet (slower but more isolated)
    use_pool = os.environ.get("RENDER_USE_SUBPROCESS", "").lower() not in ("1", "true", "yes")
    
    # Parallel render with early stopping: if 90%+ complete and tail takes >180s, cancel remaining
    rendered_images = render_batch_codes(
        render_tasks,
        max_workers=max_workers,
        timeout=30,
        use_process_pool=use_pool,
        progress_callback=render_progress,
    )
    
    # Compute render scores (1.0 if rendered, 0.0 if not)
    render_scores = [1.0 if img_b64 is not None else 0.0 for img_b64 in rendered_images]
    render_ok = sum(1 for s in render_scores if s > 0)
    syntax_err_count = sum(1 for e in syntax_errors if e)
    print(f'[codegen_reward] rendering: {render_ok}/{len(predicts)} succeeded, {syntax_err_count} syntax errors')

    # Optional: save a sample of rendered images (+ Q&A) per step (set SAVE_RENDER_EXAMPLES=1 before training)
    save_examples = os.environ.get("SAVE_RENDER_EXAMPLES", "").strip().lower() in ("1", "true", "yes")
    if save_examples and render_ok > 0:
        step = _codegen_render_example_step[0]
        max_ex = int(os.environ.get("SAVE_RENDER_EXAMPLES_N", "10"))
        _save_codegen_render_examples(
            step,
            rendered_images,
            easy_questions,
            easy_answers,
            hard_questions,
            hard_answers,
            questions,
            max_examples=max_ex,
        )
        _codegen_render_example_step[0] += 1

    # ==================================================================
    # Step 2: Solvability — easy_question on rendered images
    #   Silver label = PROPOSER's easy_answer
    #   Score = fraction of 8 rollouts that match the proposer's answer
    # ==================================================================
    easy_solver_items = []
    easy_solver_indices = []

    for idx, (img_b64, easy_q, easy_a) in enumerate(
        zip(rendered_images, easy_questions, easy_answers)
    ):
        if img_b64 is not None and easy_q:
            easy_solver_items.append({
                "question": easy_q,
                "answer": easy_a,  # proposer's answer (used as silver label below)
                "types": "numerical",
                "image": f"data:image/png;base64,{img_b64}",
            })
            easy_solver_indices.append(idx)

    print(f'[codegen_reward] solvability: sending {len(easy_solver_items)} items to Solver (easy_q)...')
    easy_solver_results = query_solver_with_images(easy_solver_items) if easy_solver_items else []

    # Solvability = fraction of Solver rollouts matching the Proposer's easy_answer
    solvability_scores = [0.0] * len(predicts)
    for i, result in enumerate(easy_solver_results):
        if i < len(easy_solver_indices):
            original_idx = easy_solver_indices[i]
            proposer_easy_answer = easy_answers[original_idx]

            rollout_answers = result.get("results", [])
            if rollout_answers and proposer_easy_answer:
                correct_count = 0
                for ans in rollout_answers:
                    if ans:
                        try:
                            if grade_answer(ans, proposer_easy_answer) or grade_answer(proposer_easy_answer, ans):
                                correct_count += 1
                        except Exception:
                            if ans.strip().lower() == proposer_easy_answer.strip().lower():
                                correct_count += 1
                solvability_scores[original_idx] = correct_count / len(rollout_answers)
            else:
                solvability_scores[original_idx] = result.get("score", 0.0)

    # ==================================================================
    # Step 3: Difficulty — hard_question on rendered images
    #   Silver label = SOLVER's own majority vote (self-consistency)
    #   Score = 1 if Solver can answer at least once (score > 0), else 0
    # ==================================================================
    hard_solver_items = []
    hard_solver_indices = []

    for idx, (img_b64, hard_q, hard_a) in enumerate(
        zip(rendered_images, hard_questions, hard_answers)
    ):
        if img_b64 is not None and hard_q:
            hard_solver_items.append({
                "question": hard_q,
                "answer": hard_a,  # passed but NOT used — Solver uses its own majority vote
                "types": "numerical",
                "image": f"data:image/png;base64,{img_b64}",
            })
            hard_solver_indices.append(idx)

    print(f'[codegen_reward] difficulty: sending {len(hard_solver_items)} items to Solver (hard_q)...')
    hard_solver_results = query_solver_with_images(hard_solver_items) if hard_solver_items else []

    difficulty_scores = [0.0] * len(predicts)
    for i, result in enumerate(hard_solver_results):
        if i < len(hard_solver_indices):
            original_idx = hard_solver_indices[i]
            raw_score = result.get("score", 0.0)  # self-consistency from Solver's majority vote
            # min(score, 1-score) peaks at 0.5: neither too easy nor too hard
            difficulty_scores[original_idx] = min(raw_score, 1.0 - raw_score)

    # ==================================================================
    # Step 4: Compute final scores = renderability + solvability + difficulty - penalties
    # ==================================================================
    scores = []
    for i in range(len(predicts)):
        r = render_scores[i]
        s = solvability_scores[i]
        d = difficulty_scores[i]
        has_syntax_err = syntax_errors[i] if i < len(syntax_errors) else False
        is_compliant = type_compliant[i] if i < len(type_compliant) else True

        # Visual type diversity penalty (based on actual generated type)
        # Uses both top-level type (matplotlib/plotly/pillow/svg) and subtype
        # (mpl_stacked_area, mpl_bar, etc.) to penalize intra-matplotlib convergence too.
        avt = actual_visual_types[i] if i < len(actual_visual_types) else "matplotlib"
        st = code_subtypes[i] if i < len(code_subtypes) else "mpl_other"

        # Top-level: penalize if one visual type dominates the batch
        avt_frac = actual_vt_fracs.get(avt, 0.25)
        if avt_frac > 0.8:
            vt_penalty = CODEGEN_VT_DIVERSITY_MAX * 2.0
        elif avt_frac > 0.6:
            vt_penalty = CODEGEN_VT_DIVERSITY_MAX * 1.5
        elif avt_frac > 0.4:
            vt_penalty = CODEGEN_VT_DIVERSITY_MAX * 0.5
        else:
            vt_penalty = 0.0

        # Subtype: penalize if one specific chart pattern dominates (e.g. mpl_stacked_area > target)
        # Scale penalty by how much this subtype exceeds CODEGEN_SUBTYPE_TARGET_FRAC.
        st_frac = subtype_fracs.get(st, 0.1)
        target = CODEGEN_SUBTYPE_TARGET_FRAC
        if st_frac <= target:
            st_penalty = 0.0
            # Small bonus for rare subtypes to encourage bar, scatter, heatmap, etc.
            st_bonus = CODEGEN_SUBTYPE_BONUS if st_frac < max(0.05, target * 0.5) else 0.0
        else:
            st_bonus = 0.0
            # Escalating penalty: e.g. at 60% with target 25%, excess = 0.35 → penalty up to 2*MAX
            excess = min(1.0 - target, st_frac - target)
            st_penalty = CODEGEN_VT_DIVERSITY_MAX * (1.0 + excess / (1.0 - target))
            st_penalty = min(st_penalty, CODEGEN_VT_DIVERSITY_MAX * 2.5)

        diversity_penalty = vt_penalty + st_penalty - st_bonus

        if r <= 0.0:
            base = 0.0
            penalty = CODEGEN_PENALTY_NO_RENDER
            if has_syntax_err:
                penalty += CODEGEN_PENALTY_SYNTAX_ERROR
            if not is_compliant:
                penalty += CODEGEN_PENALTY_WRONG_TYPE
            overall = base - penalty
        else:
            base = r + s + d
            penalty = 0.0
            if has_syntax_err:
                penalty += CODEGEN_PENALTY_SYNTAX_ERROR
            if not is_compliant:
                penalty += CODEGEN_PENALTY_WRONG_TYPE
            penalty += diversity_penalty
            overall = base - penalty

        overall = max(-0.5, overall)

        scores.append({
            "overall": overall,
            "renderability": r,
            "solvability": s,
            "difficulty": d,
            "type_compliant": 1.0 if is_compliant else 0.0,
            "vt_diversity_penalty": float(diversity_penalty),
        })

    # Log summary
    rendered = [s for s in scores if s["renderability"] > 0]
    compliant_count = sum(1 for s in scores if s["type_compliant"] > 0)
    print(f'[codegen_reward] === SCORING SUMMARY ===')
    print(f'  renderable:      {len(rendered)}/{len(scores)}')
    print(f'  type_compliant:  {compliant_count}/{len(scores)}')
    if rendered:
        print(f'  avg_solvability: {sum(s["solvability"] for s in rendered) / len(rendered):.3f}')
        print(f'  avg_difficulty:  {sum(s["difficulty"] for s in rendered) / len(rendered):.3f}')
        print(f'  avg_vt_div_pen:  {sum(s["vt_diversity_penalty"] for s in rendered) / len(rendered):.3f}')
    print(f'  avg_overall:     {sum(s["overall"] for s in scores) / len(scores):.3f}')

    return scores
