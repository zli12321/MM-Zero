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
Reward function for the Proposer model in ImageFree Self-Play.

The Proposer generates: <caption>, <easy_question>, <easy_answer>, <hard_question>, <hard_answer>.

Reward computation (per proposal):

  1. CodeGen produces N code samples (N=8).
  2. Each code sample is rendered:
     - If render FAILS  → that rollout contributes 0 to the reward.
     - If render SUCCEEDS → that rendered image is sent to the Solver twice:
       a) Solvability: image + easy_question → Solver (8 rollouts).
          Compare each rollout against the PROPOSER's easy_answer (silver label).
          solvability_score = fraction of rollouts matching the proposer's answer.
       b) Difficulty: image + hard_question → Solver (8 rollouts).
          Use the SOLVER's own majority vote as silver label (self-consistency).
          difficulty_score = min(consistency, 1 - consistency), peaks at 0.5.
       That rollout contributes (solvability_score + difficulty_score).
  3. Final reward = (sum of all N rollout contributions) / N

  If format is invalid (can't parse 5 tags): reward = -1.0

Services (launched by start_proposer_services.sh):
  - CodeGen: ports 7000-7001 (GPUs 4-5), n=8 code samples per item
  - Solver:  ports 6000-6001 (GPUs 6-7), n=8 rollouts per item
"""

import re
import os
import sys
import json
import time
import random
import subprocess
import tempfile
import base64
from typing import Dict, List, Optional


# Step counter for saving example renders (incremented each reward call so step_0, step_1, ...)
# Auto-detect: start from the next unused step_N directory so restarts don't overwrite.
def _init_render_step_counter():
    storage = os.environ.get("STORAGE_PATH") or "."
    base = os.path.join(storage, "rendered_images", "examples")
    if not os.path.isdir(base):
        return 0
    existing = [d for d in os.listdir(base) if d.startswith("step_") and d[5:].isdigit()]
    if not existing:
        return 0
    return max(int(d[5:]) for d in existing) + 1

_render_example_step = [_init_render_step_counter()]


def _term(msg: str, end: str = "\n") -> None:
    """Print to stderr so it appears on the training terminal even when stdout is logged to a file."""
    sys.stderr.write(msg + end)
    sys.stderr.flush()


def _save_render_examples(
    step: int,
    rendered_per_proposal: Dict[int, List[str]],
    all_fields: List[Optional[Dict]],
    codegen_indices: List[int],
    max_examples: int = 10,
) -> None:
    """Save a sample of rendered images to STORAGE_PATH/rendered_images/examples/step_{step}/.
    Path: STORAGE_PATH/rendered_images/examples/step_N/ (images + info.json with Q&A)."""
    storage = os.environ.get("STORAGE_PATH") or "."
    base = os.path.join(storage, "rendered_images", "examples", f"step_{step}")
    os.makedirs(base, exist_ok=True)
    saved = []
    count = 0
    for orig_idx in codegen_indices:
        if count >= max_examples:
            break
        images = rendered_per_proposal.get(orig_idx, [])
        if not images:
            continue
        fields = all_fields[orig_idx] if orig_idx < len(all_fields) and all_fields[orig_idx] else {}
        caption = (fields.get("caption") or "")[:500]
        easy_question = (fields.get("easy_question") or "")[:500]
        easy_answer = fields.get("easy_answer") or ""
        hard_question = (fields.get("hard_question") or "")[:500]
        hard_answer = fields.get("hard_answer") or ""
        # For diversity: save at most ONE image per proposal (first successful rollout)
        img_b64 = images[0]
        try:
            raw = base64.b64decode(img_b64)
            fname = f"proposal_{orig_idx}_rollout_0.png"
            path = os.path.join(base, fname)
            with open(path, "wb") as f:
                f.write(raw)
            saved.append({
                "file": fname,
                "proposal_idx": orig_idx,
                "rollout": 0,
                "caption": caption,
                "easy_question": easy_question,
                "easy_answer": easy_answer,
                "hard_question": hard_question,
                "hard_answer": hard_answer,
            })
            count += 1
        except Exception as e:
            print(f"[proposer_reward] save example {orig_idx}_0: {e}", flush=True)
    info_path = os.path.join(base, "info.json")
    with open(info_path, "w") as f:
        json.dump({"step": step, "saved_count": len(saved), "entries": saved}, f, indent=2)
    print(f"[proposer_reward] Saved {len(saved)} example images + Q&A to {base}", flush=True)


from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from mathruler.grader import grade_answer

from io import BytesIO

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    from sklearn.cluster import AgglomerativeClustering
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    AgglomerativeClustering = None

STORAGE_PATH = os.getenv("STORAGE_PATH")
if STORAGE_PATH is None:
    STORAGE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"

TEMP_RESULTS_DIR = os.path.join(STORAGE_PATH, "temp_results")
os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)

# Service port configuration: vLLM uses 7010-7011 (CodeGen) and 6010-6011 (Solver)
# so 7000-7001 and 6000-6001 stay free for GPU server. Set by start_proposer_services.sh.
def _get_ports_from_env_or_file(prefix: str, default_a: int, default_b: int) -> list:
    a = os.environ.get(f"{prefix}_PORT_0")
    b = os.environ.get(f"{prefix}_PORT_1")
    if a is not None and b is not None:
        return [int(a), int(b)]
    ports_file = os.path.join(TEMP_RESULTS_DIR, "proposer_service_ports.env")
    if os.path.isfile(ports_file):
        try:
            with open(ports_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"export {prefix}_PORT_0="):
                        a = line.split("=", 1)[1].strip().strip("'\"")
                    elif line.startswith(f"export {prefix}_PORT_1="):
                        b = line.split("=", 1)[1].strip().strip("'\"")
            if a is not None and b is not None:
                return [int(a), int(b)]
        except Exception:
            pass
    return [default_a, default_b]


CODEGEN_PORTS = _get_ports_from_env_or_file("CODEGEN", 7010, 7011)
SOLVER_PORTS = _get_ports_from_env_or_file("SOLVER", 6010, 6011)


# ========================== Tag Extraction ========================== #

def extract_fields(predict: str) -> Optional[Dict[str, str]]:
    """Extract all structured fields from the Proposer output (including visual_type)."""
    predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)

    visual_type = re.search(r"<visual_type>([\s\S]*?)</visual_type>", predict)
    caption = re.search(r"<caption>([\s\S]*?)</caption>", predict)
    easy_q = re.search(r"<easy_question>([\s\S]*?)</easy_question>", predict)
    easy_a = re.search(r"<easy_answer>([\s\S]*?)</easy_answer>", predict)
    hard_q = re.search(r"<hard_question>([\s\S]*?)</hard_question>", predict)
    hard_a = re.search(r"<hard_answer>([\s\S]*?)</hard_answer>", predict)

    if caption and easy_q and easy_a and hard_q and hard_a:
        vt = (visual_type.group(1).strip().lower() if visual_type else "matplotlib").strip()
        if vt not in ("matplotlib", "plotly", "pillow", "svg"):
            vt = "matplotlib"
        return {
            "visual_type": vt,
            "caption": caption.group(1).strip(),
            "easy_question": easy_q.group(1).strip(),
            "easy_answer": easy_a.group(1).strip(),
            "hard_question": hard_q.group(1).strip(),
            "hard_answer": hard_a.group(1).strip(),
        }
    return None


# ========================== Code Rendering ========================== #
try:
    from SelfAgent_svg.code_render.render_code import render_single, render_batch_codes
except ImportError:
    from code_render.render_code import render_single, render_batch_codes


# ========================== Diversity Penalty ========================== #

def _bleu_distance_matrix(sentences):
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    return dist


def cluster_share_per_problem(
        problems: List[str],
        distance_threshold: float = 0.5,
        linkage: str = "average") -> List[float]:
    if not problems:
        return []
    if not _HAS_SKLEARN or AgglomerativeClustering is None:
        # No scikit-learn: return uniform weights (pip install scikit-learn for diversity penalty)
        return [1.0 / len(problems)] * len(problems)
    print('[proposer_reward] clustering for diversity penalty...')
    start_time = time.time()
    dist_mat = _bleu_distance_matrix(problems)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    print(f'[proposer_reward] clustering done in {time.time() - start_time:.1f}s')
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}
    return [cluster_ratio[lab] for lab in labels]


# ========================== Image Diversity ========================== #

_IMG_RESIZE = 64  # resize all images to 64x64 before comparing

def _b64_to_vector(b64_str: str) -> Optional[np.ndarray]:
    """Decode a base64 PNG, resize to _IMG_RESIZE x _IMG_RESIZE, flatten to a 1-D float vector."""
    if not _HAS_PIL or not b64_str:
        return None
    try:
        raw = base64.b64decode(b64_str)
        img = PILImage.open(BytesIO(raw)).convert("RGB")
        img = img.resize((_IMG_RESIZE, _IMG_RESIZE), PILImage.LANCZOS)
        arr = np.asarray(img, dtype=np.float32).flatten()
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        return arr
    except Exception:
        return None


def _cosine_distance_matrix(vectors: List[np.ndarray]) -> np.ndarray:
    """Compute pairwise cosine distance matrix (1 - cosine_similarity)."""
    n = len(vectors)
    mat = np.stack(vectors)  # (n, d)
    sim = mat @ mat.T  # cosine similarity (vectors are already L2-normalized)
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def image_cluster_shares(
    b64_images: List[str],
    distance_threshold: float = 0.3,
    linkage: str = "average",
) -> List[float]:
    """Cluster rendered images by visual similarity; return per-image cluster share."""
    vectors = [_b64_to_vector(b) for b in b64_images]
    valid_mask = [v is not None for v in vectors]
    valid_vectors = [v for v in vectors if v is not None]

    if len(valid_vectors) < 2 or not _HAS_SKLEARN:
        return [1.0 / max(len(b64_images), 1)] * len(b64_images)

    print(f'[proposer_reward] Image diversity: clustering {len(valid_vectors)} images...')
    t0 = time.time()
    dist = _cosine_distance_matrix(valid_vectors)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage,
    )
    labels = clustering.fit_predict(dist)
    n_clusters = len(set(labels))
    print(f'[proposer_reward] Image diversity: {n_clusters} clusters in {time.time() - t0:.1f}s')

    total = len(valid_vectors)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}
    valid_shares = [cluster_ratio[lab] for lab in labels]

    shares = []
    vi = 0
    uniform = 1.0 / max(len(b64_images), 1)
    for m in valid_mask:
        if m:
            shares.append(valid_shares[vi])
            vi += 1
        else:
            shares.append(uniform)
    return shares


# ========================== Service Communication ========================== #

def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000)
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def fetch_codegen(index, filepath, timeout_sec=None):
    """Send a task file to the CodeGen vLLM service. Returns (success, status_code, error_msg).
    timeout_sec: None = wait indefinitely; else max seconds to wait for response."""
    port = CODEGEN_PORTS[index]
    url = f"http://0.0.0.0:{port}/codegen?name={filepath}"
    try:
        r = requests.get(url, timeout=timeout_sec)
        return (r.ok, r.status_code, r.text[:500] if not r.ok else None)
    except Exception as e:
        return (False, -1, str(e))


def _codegen_timeout_and_wait(datas: List[List]) -> tuple:
    """Compute (request_timeout_sec, file_wait_max_sec). Either can be None = wait until done.
    Env CODEGEN_HTTP_TIMEOUT:
      - "0", "none", "inf" → no timeout (wait until CodeGen finishes).
      - "auto" (default) → scale with batch: max(600, 600 + 2*items_per_service), capped at 7200 (2h).
      - positive integer → use that many seconds."""
    env = (os.environ.get("CODEGEN_HTTP_TIMEOUT", "auto") or "auto").strip().lower()
    if env in ("0", "none", "inf", "infinity"):
        return (None, None)
    if env == "auto":
        max_items = max(len(d) for d in datas) if datas else 0
        # ~2 s per item heuristic; min 10 min, cap 2 h
        t = min(7200, max(600, 600 + 2 * max_items))
        return (t, t)
    t = max(60, int(env))
    return (t, t)


def fetch_solver(index, filepath):
    """Send a task file to the Solver vLLM service."""
    port = SOLVER_PORTS[index]
    requests.get(f"http://0.0.0.0:{port}/hello?name={filepath}")
    return True


def query_codegen_service(items: List[Dict]) -> List[Dict]:
    """
    Send items to CodeGen vLLM service to generate code (8 samples per item).
    Items: [{caption, easy_question, easy_answer, hard_question, hard_answer}, ...]
    Returns: [{..., generated_codes: [code1, ..., code8]}, ...]
    """
    if not items:
        return []

    n_services = len(CODEGEN_PORTS)
    datas = split_list(items, n_services)
    random_names = [generate_temp_filename(prefix=f"codegen_{i}", suffix=".json")
                    for i in range(n_services)]

    for i in range(n_services):
        with open(random_names[i], 'w') as f:
            json.dump(datas[i], f, indent=4)

    # Debug: print input file paths and first 1–2 items we send to CodeGen
    print(f'[proposer_reward] sending {len(items)} items to {n_services} CodeGen services...')
    print(f'[proposer_reward] DEBUG: CodeGen input paths: {random_names}')
    for si, it in enumerate(items[:2]):
        cap = (it.get("caption", "") or "")[:100]
        eq = (it.get("easy_question", "") or "")[:80]
        hq = (it.get("hard_question", "") or "")[:80]
        print(f'  Input item {si}: caption="{cap}..." easy_q="{eq}..." hard_q="{hq}..."')

    timeout_sec, wait_max = _codegen_timeout_and_wait(datas)
    if timeout_sec is None:
        _term("[proposer_reward] CodeGen timeout: none (waiting until services finish or you kill the process).")
    else:
        _term(f"[proposer_reward] CodeGen timeout: {timeout_sec}s (request + file wait). Set CODEGEN_HTTP_TIMEOUT=0 for no limit.")

    with ThreadPoolExecutor(max_workers=n_services) as executor:
        future_to_idx = {executor.submit(fetch_codegen, i, random_names[i], timeout_sec): i for i in range(n_services)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            ok, status, err = future.result()
            if not ok:
                print(f"[proposer_reward] WARNING: CodeGen service {idx} (port {CODEGEN_PORTS[idx]}) failed: status={status}, error={err}")

    # Wait for result files (CodeGen can take 10+ min for 601 items with 7B).
    results_files = [random_names[i].replace('.json', '_results.json') for i in range(n_services)]
    log_dir = os.path.join(STORAGE_PATH or ".", "temp_results")
    print("[proposer_reward] >>> Phase: CodeGen — generating code from proposals.", flush=True)
    if wait_max is None:
        print("[proposer_reward] >>> Waiting for result files (no time limit).", flush=True)
    else:
        print(f"[proposer_reward] >>> Waiting up to {wait_max}s for CodeGen results.", flush=True)
    _term("")
    _term("[proposer_reward] >>> Phase: CodeGen — generating code from proposals (not hanging).")
    if wait_max is None:
        _term("[proposer_reward] >>> Waiting for result files until they appear (no time limit). Progress: 'Processed prompts' in logs.")
    else:
        _term(f"[proposer_reward] >>> For 7B + many items, expect 10–30+ min. Waiting up to {wait_max}s. Progress: 'Processed prompts' in logs.")
    _term(f"[proposer_reward] >>> For live progress, run: tail -f {log_dir}/codegen_*.log")
    _term(f'[proposer_reward] CodeGen: waiting for result files' + (f' (max {wait_max}s)...' if wait_max else ' (no limit)...'))
    waited = 0
    step = 10
    while True:
        if all(os.path.isfile(rf) for rf in results_files):
            if waited > 0:
                _term(f'[proposer_reward] CodeGen: results ready at {waited}s.')
            break
        if wait_max is not None and waited >= wait_max:
            missing = [rf for rf in results_files if not os.path.isfile(rf)]
            print(f"[proposer_reward] WARNING: After {wait_max}s wait, still missing: {missing}")
            break
        if waited > 0:
            n_ready = sum(1 for rf in results_files if os.path.isfile(rf))
            _term(f'[proposer_reward] CodeGen: waiting... {waited}s elapsed ({n_ready}/{n_services} files)')
        for _ in range(step):
            time.sleep(1)
            waited += 1
            if all(os.path.isfile(rf) for rf in results_files):
                break
            if wait_max is not None and waited >= wait_max:
                break
        if wait_max is not None and waited >= wait_max:
            missing = [rf for rf in results_files if not os.path.isfile(rf)]
            print(f"[proposer_reward] WARNING: After {wait_max}s wait, still missing: {missing}")
            break

    final_results = []
    for i in range(n_services):
        results_file = results_files[i]
        try:
            with open(results_file, 'r') as f:
                raw = json.load(f)
            final_results.extend(raw)
            for ri, res in enumerate(raw[:2]):
                keys = list(res.keys())
                codes = res.get("generated_codes", [])
                print(f'[proposer_reward] DEBUG: CodeGen result service {i} item {ri}: keys={keys}, len(generated_codes)={len(codes)}')
                for ci, code in enumerate(codes[:2]):
                    preview = (code[:800] + "\n...") if len(code) > 800 else code
                    print(f'  Parsed code [{ci}]:\n{preview}\n')
                if not codes:
                    print(f'  (no parsed code; generated_codes empty or missing)')
            os.remove(results_file)
        except json.JSONDecodeError as e:
            print(f"[proposer_reward] WARNING: CodeGen results corrupted (service {i}): {e}")
            print(f"[proposer_reward] Filling {len(datas[i])} items with empty codes (score=0)")
            final_results.extend([{"generated_codes": []}] * len(datas[i]))
            try:
                os.remove(results_file)
            except OSError:
                pass
        except FileNotFoundError:
            print(f"[proposer_reward] WARNING: CodeGen results not found: {results_file}")
            final_results.extend([{"generated_codes": []}] * len(datas[i]))

    return final_results


def query_solver_service(items: List[Dict]) -> List[Dict]:
    """
    Send items to Solver vLLM service to evaluate easy questions on rendered images.
    Items: [{question, answer, types, image}, ...]
    Returns: [{question, answer, score, ...}, ...]
    """
    if not items:
        return []

    n_services = len(SOLVER_PORTS)
    datas = split_list(items, n_services)
    random_names = [generate_temp_filename(prefix=f"solver_{i}", suffix=".json")
                    for i in range(n_services)]

    for i in range(n_services):
        with open(random_names[i], 'w') as f:
            json.dump(datas[i], f, indent=4)

    print(f'[proposer_reward] sending {len(items)} items to {n_services} Solver services...')
    with ThreadPoolExecutor(max_workers=n_services) as executor:
        futures = [executor.submit(fetch_solver, i, random_names[i])
                   for i in range(n_services)]
        for future in as_completed(futures):
            future.result()

    results_files = [random_names[i].replace('.json', '_results.json') for i in range(n_services)]
    wait_max = 180
    for waited in range(0, wait_max, 10):
        if all(os.path.isfile(rf) for rf in results_files):
            if waited > 0:
                _term(f'[proposer_reward] Solver: results ready at {waited}s.')
            break
        if waited > 0:
            n_ready = sum(1 for rf in results_files if os.path.isfile(rf))
            _term(f'[proposer_reward] Solver: waiting... {waited}s elapsed ({n_ready}/{n_services} files)')
        for _ in range(10):
            time.sleep(1)
            if all(os.path.isfile(rf) for rf in results_files):
                break

    final_results = []
    for i in range(n_services):
        results_file = results_files[i]
        try:
            with open(results_file, 'r') as f:
                final_results.extend(json.load(f))
            os.remove(results_file)
        except json.JSONDecodeError as e:
            _term(f"[proposer_reward] WARNING: Solver results corrupted (service {i}): {e}")
            _term(f"[proposer_reward] Filling {len(datas[i])} items with score=0.0")
            final_results.extend([{"score": 0.0}] * len(datas[i]))
            try:
                os.remove(results_file)
            except OSError:
                pass
        except FileNotFoundError:
            _term(f"[proposer_reward] WARNING: Solver results not found: {results_file}")
            final_results.extend([{"score": 0.0}] * len(datas[i]))

    return final_results


# ========================== Main Reward ========================== #

def compute_score(
    predicts: List[str],
    ground_truths: List[str],
    questions: List[str],
    description_answers: List[str],
    format_weight: float = 0.1,
    images: Optional[List] = None
) -> List[Dict[str, float]]:
    """
    Compute reward for each Proposer output.

    Pipeline per prediction:
      1. Parse → extract caption, easy_q, easy_a, hard_q, hard_a
      2. Proposal → CodeGen service → N=8 code samples
      3. Render each code sample → some succeed, some fail
      4a. Solvability: render + easy_q → Solver (8 rollouts) → compare vs proposer's easy_answer
      4b. Difficulty: render + hard_q → Solver (8 rollouts) → self-consistency → min(score,1-score)
      5. reward = (sum of: 0 for failed renders + (solvability + difficulty) for successes) / N

    If format is wrong (can't parse 5 tags): score = -1.0
    """
    step = _render_example_step[0]
    _render_example_step[0] += 1

    n = len(predicts)

    # ==================================================================
    # DEBUG: Log example Proposer generations
    # ==================================================================
    _debug_max_len = 1200
    print("[proposer_reward] === DEBUG: Example Proposer generations ===")
    for i, pred in enumerate(predicts[:2]):
        snippet = pred[: _debug_max_len] + ("..." if len(pred) > _debug_max_len else "")
        print(f"  [Proposer sample {i}] (len={len(pred)}):\n{snippet}\n")
    if n > 2:
        print(f"  ... and {n - 2} more predictions.\n")

    # ==================================================================
    # Step 1: Parse all predictions
    # ==================================================================
    all_fields = []  # None if parsing failed, dict otherwise
    for predict in predicts:
        predict_clean = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)
        fields = extract_fields(predict_clean)
        all_fields.append(fields)

    # ==================================================================
    # Step 2: CodeGen service → N code samples per proposal
    # ==================================================================
    codegen_items = []
    codegen_indices = []  # codegen item index → original predict index
    for idx, fields in enumerate(all_fields):
        if fields is not None:
            codegen_items.append({
                "visual_type": fields.get("visual_type", "matplotlib"),
                "caption": fields["caption"],
                "easy_question": fields["easy_question"],
                "easy_answer": fields["easy_answer"],
                "hard_question": fields["hard_question"],
                "hard_answer": fields["hard_answer"],
            })
            codegen_indices.append(idx)

    print(f'[proposer_reward] {len(codegen_items)}/{n} valid format, sending to CodeGen...')
    codegen_results = query_codegen_service(codegen_items)

    n_with_codes = sum(1 for ex in codegen_results if ex.get("generated_codes"))
    total_codes = sum(len(ex.get("generated_codes", [])) for ex in codegen_results)
    print(f"[proposer_reward] >>> CodeGen finished. {len(codegen_results)} proposals, {n_with_codes} with ≥1 code, {total_codes} total code samples.", flush=True)
    _term("")
    _term(f"[proposer_reward] >>> CodeGen finished. Summary: {len(codegen_results)} proposals, {n_with_codes} with ≥1 code, {total_codes} total code samples.")
    _term("")
    # ==================================================================
    # DEBUG: CodeGen outputs — always show a few samples to verify service is working
    # ==================================================================
    _term(f"[proposer_reward] === CodeGen outputs: {len(codegen_results)} items, {n_with_codes} with generated code ===")
    for idx, ex in enumerate(codegen_results[:3]):
        codes = ex.get("generated_codes", [])
        caption = (ex.get("caption", "") or "")[:100]
        easy_q = (ex.get("easy_question", "") or "")[:70]
        _term(f"  [CodeGen sample {idx}] len(generated_codes)={len(codes)} | caption: {caption}...")
        _term(f"    easy_q: {easy_q}...")
        if codes:
            code_preview = (codes[0][:500] + "\n...") if len(codes[0]) > 500 else codes[0]
            _term(f"    first code (first 500 chars):\n{code_preview}")
            if len(codes) > 1:
                _term(f"    second code (first 200 chars): {(codes[1][:200] + '...') if len(codes[1]) > 200 else codes[1]}")
        else:
            _term("    (no parsed code — CodeGen may have failed or returned empty)")
        _term("")
    if not codegen_results:
        _term("  (no CodeGen results at all)\n")

    # ==================================================================
    # Step 3: Render all N code samples per proposal (parallel to avoid long sequential runs)
    # ==================================================================
    rendered_per_proposal = {}  # original_idx → list of base64 images
    num_codegen_rollouts = {}   # original_idx → N (total code samples for this proposal)

    # Build flat list of (orig_idx, code, visual_type) for parallel rendering
    render_tasks = []
    for i, result in enumerate(codegen_results):
        orig_idx = codegen_indices[i]
        codes = result.get("generated_codes", [])
        num_codegen_rollouts[orig_idx] = max(len(codes), 1)
        visual_type = (all_fields[orig_idx] or {}).get("visual_type", "matplotlib")
        for code in codes:
            render_tasks.append((orig_idx, code, visual_type))

    # Parallel render: long-lived workers (import matplotlib/plotly once per worker) to avoid per-snippet process startup.
    max_workers = int(os.environ.get("RENDER_MAX_WORKERS", "16"))
    total_to_render = len(render_tasks)
    print(f"[proposer_reward] Rendering {total_to_render} images in parallel (max_workers={max_workers}); progress = each image as it completes.", flush=True)
    _term(f"[proposer_reward] Rendering {total_to_render} images in parallel (max_workers={max_workers}); progress = each image as it completes.")

    # Progress callback: print to training terminal every ~5% or every 25 items, and at 100%
    last_reported_pct = -1
    report_every = max(1, total_to_render // 20)  # ~20 updates over the run

    def render_progress(done: int, total: int, success: int) -> None:
        nonlocal last_reported_pct
        pct = (100 * done) // total if total else 0
        if done == total or done % report_every == 0 or pct >= last_reported_pct + 5:
            last_reported_pct = pct
            if done == total:
                msg = f"[proposer_reward] Render progress: {done}/{total} images (100%) — {success} OK — COMPLETE."
            else:
                msg = f"[proposer_reward] Render progress: {done}/{total} images ({pct}%) — {success} OK (parallel, up to {max_workers} at a time)"
            print(msg, flush=True)
            _term(msg)

    tasks_only = [(code, vt) for _orig, code, vt in render_tasks]
    use_pool = os.environ.get("RENDER_USE_SUBPROCESS", "").lower() not in ("1", "true", "yes")
    b64_results = render_batch_codes(
        tasks_only,
        max_workers=max_workers,
        timeout=30,
        use_process_pool=use_pool,
        progress_callback=render_progress,
    )
    print(f"[proposer_reward] Render complete: {len(b64_results)}/{total_to_render} images done.", flush=True)
    _term(f"[proposer_reward] Render complete: {len(b64_results)}/{total_to_render} images done.")
    for k, (orig_idx, _code, _vt) in enumerate(render_tasks):
        if orig_idx not in rendered_per_proposal:
            rendered_per_proposal[orig_idx] = []
        if k < len(b64_results) and b64_results[k] is not None:
            rendered_per_proposal[orig_idx].append(b64_results[k])

    # Optional: save a sample of rendered images (+ Q&A) per step under STORAGE_PATH/rendered_images (set SAVE_RENDER_EXAMPLES=1 before training)
    save_examples = os.environ.get("SAVE_RENDER_EXAMPLES", "").strip().lower() in ("1", "true", "yes")
    if save_examples:
        max_ex = int(os.environ.get("SAVE_RENDER_EXAMPLES_N", "10"))
        _save_render_examples(step, rendered_per_proposal, all_fields, codegen_indices, max_examples=max_ex)

    total_renders = sum(len(imgs) for imgs in rendered_per_proposal.values())
    total_items = len(codegen_results)
    render_ok = sum(1 for imgs in rendered_per_proposal.values() if len(imgs) > 0)
    total_rollouts = sum(num_codegen_rollouts.get(orig_idx, 0) for orig_idx in codegen_indices)
    print(f"[proposer_reward] >>> Render finished. Summary: {total_renders} successful images from {total_rollouts} code samples ({render_ok}/{total_items} proposals have ≥1 success).", flush=True)
    _term("")
    _term(f"[proposer_reward] >>> Render finished. Summary: {total_renders} successful images from {total_rollouts} code samples ({render_ok}/{total_items} proposals have ≥1 success).")
    print(f'[proposer_reward] rendering done: {total_renders} successful renders across {total_items} proposals '
          f'({render_ok} proposals have ≥1 success)')
    # DEBUG: Out of CodeGen rollouts, how many successfully produced an image?
    print("[proposer_reward] === DEBUG: CodeGen rollout → image success ===")
    print(f"  Total CodeGen rollouts: {total_rollouts}  |  Successfully generated images: {total_renders}  "
          f"|  Success rate: {100.0 * total_renders / total_rollouts if total_rollouts else 0:.1f}%")
    for i, orig_idx in enumerate(codegen_indices[:5]):
        N = num_codegen_rollouts.get(orig_idx, 0)
        success = len(rendered_per_proposal.get(orig_idx, []))
        print(f"  Proposal {i} (orig_idx={orig_idx}): {success}/{N} rollouts produced an image")
    if len(codegen_indices) > 5:
        print(f"  ... and {len(codegen_indices) - 5} more proposals.\n")

    # ==================================================================
    # Step 4a: Solvability — send all rendered images + easy_question to Solver
    #   Silver label = PROPOSER's easy_answer
    #   Score = fraction of 8 rollouts that match the proposer's answer
    # ==================================================================
    easy_solver_items = []
    easy_solver_map = []  # solver_item_index → original predict index

    for orig_idx, images in rendered_per_proposal.items():
        fields = all_fields[orig_idx]
        for img_b64 in images:
            easy_solver_items.append({
                "question": fields["easy_question"],
                "answer": fields["easy_answer"],  # proposer's answer (used as silver label below)
                "types": "numerical",
                "image": f"data:image/png;base64,{img_b64}",
            })
            easy_solver_map.append(orig_idx)

    log_dir = os.path.join(STORAGE_PATH or ".", "temp_results")
    print(f"[proposer_reward] >>> Phase: Solver (solvability) — evaluating easy questions on rendered images.", flush=True)
    print(f'[proposer_reward] solvability: sending {len(easy_solver_items)} rendered images to Solver (easy_q)...', flush=True)
    _term("")
    _term("[proposer_reward] >>> Phase: Solver (solvability) — evaluating easy questions on rendered images.")
    _term(f"[proposer_reward] >>> For live progress, run: tail -f {log_dir}/solver_*.log")
    easy_solver_results = query_solver_service(easy_solver_items)

    easy_scores_flat = []
    for r in easy_solver_results:
        sc = r.get("score")
        if isinstance(sc, (int, float)):
            easy_scores_flat.append(sc)
    avg_easy = sum(easy_scores_flat) / len(easy_scores_flat) if easy_scores_flat else 0.0
    print(f"[proposer_reward] >>> Solver (easy) finished. Summary: {len(easy_solver_results)} items, avg score {avg_easy:.3f}.", flush=True)
    _term("")
    _term(f"[proposer_reward] >>> Solver (easy) finished. Summary: {len(easy_solver_results)} items, avg self-consistency score {avg_easy:.3f}.")
    _term("")
    # ==================================================================
    # Solver (easy Q) outputs — show samples to verify Solver is working
    # ==================================================================
    _term(f"[proposer_reward] === Solver (solvability / easy Q): {len(easy_solver_results)} results ===")
    for si, (ex, item) in enumerate(zip(easy_solver_results[:2], easy_solver_items[:2])):
        q = (item.get("question") or "")[:90]
        expected = (item.get("answer") or "")[:50]
        rollout_answers = ex.get("results", [])
        score = ex.get("score", "N/A")
        _term(f"  [Solver easy sample {si}] question: {q}...")
        _term(f"    expected (proposer): {expected} | score: {score}")
        _term(f"    rollout answers ({len(rollout_answers)}): {rollout_answers[:6]}")
        if len(rollout_answers) > 6:
            _term(f"      ... and {len(rollout_answers) - 6} more")
        _term("")
    if not easy_solver_results:
        _term("  (no easy_solver results to show)\n")

    # Solvability = fraction of Solver rollouts matching the Proposer's easy_answer
    easy_scores_per_proposal = {}  # original_idx → list of solvability scores
    for i, result in enumerate(easy_solver_results):
        orig_idx = easy_solver_map[i]
        fields = all_fields[orig_idx]
        proposer_easy_answer = fields["easy_answer"]

        rollout_answers = result.get("results", [])
        if rollout_answers:
            correct_count = 0
            for ans in rollout_answers:
                if ans and proposer_easy_answer:
                    try:
                        if grade_answer(ans, proposer_easy_answer) or grade_answer(proposer_easy_answer, ans):
                            correct_count += 1
                    except Exception:
                        if ans.strip().lower() == proposer_easy_answer.strip().lower():
                            correct_count += 1
            solvability_score = correct_count / len(rollout_answers)
        else:
            solvability_score = result.get("score", 0.0)

        if orig_idx not in easy_scores_per_proposal:
            easy_scores_per_proposal[orig_idx] = []
        easy_scores_per_proposal[orig_idx].append(solvability_score)

    # ==================================================================
    # Step 4b: Difficulty — send all rendered images + hard_question to Solver
    #   Silver label = SOLVER's own majority vote (self-consistency)
    #   Score = min(consistency, 1 - consistency), peaks at 0.5
    # ==================================================================
    hard_solver_items = []
    hard_solver_map = []  # solver_item_index → original predict index

    for orig_idx, images in rendered_per_proposal.items():
        fields = all_fields[orig_idx]
        for img_b64 in images:
            hard_solver_items.append({
                "question": fields["hard_question"],
                "answer": fields["hard_answer"],  # passed but NOT used — Solver uses its own majority vote
                "types": "numerical",
                "image": f"data:image/png;base64,{img_b64}",
            })
            hard_solver_map.append(orig_idx)

    print(f"[proposer_reward] >>> Phase: Solver (difficulty) — evaluating hard questions on rendered images.", flush=True)
    print(f'[proposer_reward] difficulty: sending {len(hard_solver_items)} rendered images to Solver (hard_q)...', flush=True)
    _term("")
    _term("[proposer_reward] >>> Phase: Solver (difficulty) — evaluating hard questions on rendered images.")
    hard_solver_results = query_solver_service(hard_solver_items)

    hard_scores_flat = [r.get("score") for r in hard_solver_results if isinstance(r.get("score"), (int, float))]
    avg_hard = sum(hard_scores_flat) / len(hard_scores_flat) if hard_scores_flat else 0.0
    print(f"[proposer_reward] >>> Solver (hard) finished. Summary: {len(hard_solver_results)} items, avg score {avg_hard:.3f}.", flush=True)
    _term("")
    _term(f"[proposer_reward] >>> Solver (hard) finished. Summary: {len(hard_solver_results)} items, avg self-consistency score {avg_hard:.3f}.")
    _term("")
    # Solver (hard Q) outputs — show first 2 samples only; avg score is over all items (some may be None, others valid)
    _term(f"[proposer_reward] === Solver (difficulty / hard Q): {len(hard_solver_results)} results ===")
    _term("[proposer_reward] (Showing first 2 samples below; overall avg uses all items. 'None' answers are excluded from scoring.)")
    for si, (ex, item) in enumerate(zip(hard_solver_results[:2], hard_solver_items[:2])):
        hard_q = (item.get("question") or "")[:90]
        rollout_answers = ex.get("results", [])
        score = ex.get("score", "N/A")
        _term(f"  [Solver hard sample {si}] question: {hard_q}...")
        _term(f"    score (self-consistency): {score} | rollout answers ({len(rollout_answers)}): {rollout_answers[:6]}")
        if len(rollout_answers) > 6:
            _term(f"      ... and {len(rollout_answers) - 6} more")
        _term("")
    if not hard_solver_results:
        _term("  (no hard_solver results to show)\n")

    # Difficulty uses the Solver's self-consistency score (majority vote / total)
    # Apply min(score, 1-score) so it peaks at 0.5 (neither too easy nor too hard)
    hard_scores_per_proposal = {}  # original_idx → list of difficulty scores
    for i, result in enumerate(hard_solver_results):
        orig_idx = hard_solver_map[i]
        raw_score = result.get("score", 0.0)  # self-consistency from Solver's majority vote
        diff_score = min(raw_score, 1.0 - raw_score)  # peaks at 0.5
        if orig_idx not in hard_scores_per_proposal:
            hard_scores_per_proposal[orig_idx] = []
        hard_scores_per_proposal[orig_idx].append(diff_score)

    # ==================================================================
    # Step 5: Compute per-proposal reward (with diversity bonuses)
    # ==================================================================

    # --- 5a: Visual type diversity ---
    VT_DIVERSITY_MAX_BONUS = float(os.environ.get("VT_DIVERSITY_MAX_BONUS", "0.3"))
    type_counts = Counter()
    for idx in range(n):
        fields = all_fields[idx]
        if fields is not None:
            vt = fields.get("visual_type", "matplotlib")
            type_counts[vt] += 1
    total_valid = sum(type_counts.values()) or 1
    type_fractions = {vt: cnt / total_valid for vt, cnt in type_counts.items()}
    print(f'[proposer_reward] Visual type distribution in batch: {dict(type_counts)} '
          f'(fractions: { {vt: f"{f:.2f}" for vt, f in type_fractions.items()} })')

    # --- 5b: Per-field diversity (BLEU clustering on caption, easy_q, hard_q separately) ---
    # Each field is clustered independently. Proposals in large clusters are penalized.
    # Weights: caption=0.45, easy_question=0.20, hard_question=0.35 (sum=1.0)
    # The total diversity budget is CAPTION_DIVERSITY_WEIGHT (default 0.3), split across fields.
    CAPTION_DIVERSITY_WEIGHT = float(os.environ.get("CAPTION_DIVERSITY_WEIGHT", "0.3"))
    W_CAPTION = float(os.environ.get("DIV_W_CAPTION", "0.45"))
    W_EASY_Q = float(os.environ.get("DIV_W_EASY_Q", "0.20"))
    W_HARD_Q = float(os.environ.get("DIV_W_HARD_Q", "0.35"))

    valid_indices = [idx for idx in range(n) if all_fields[idx] is not None]
    n_valid = len(valid_indices)

    if n_valid > 1:
        caption_texts = [all_fields[idx]['caption'] for idx in valid_indices]
        easy_q_texts = [all_fields[idx]['easy_question'] for idx in valid_indices]
        hard_q_texts = [all_fields[idx]['hard_question'] for idx in valid_indices]

        caption_shares = cluster_share_per_problem(caption_texts)
        easy_q_shares = cluster_share_per_problem(easy_q_texts)
        hard_q_shares = cluster_share_per_problem(hard_q_texts)

        uniform_share = 1.0 / n_valid
        diversity_penalties = {}
        for i, idx in enumerate(valid_indices):
            weighted_excess = (
                W_CAPTION * (caption_shares[i] - uniform_share)
                + W_EASY_Q * (easy_q_shares[i] - uniform_share)
                + W_HARD_Q * (hard_q_shares[i] - uniform_share)
            )
            raw_pen = weighted_excess * n_valid * CAPTION_DIVERSITY_WEIGHT
            diversity_penalties[idx] = max(-CAPTION_DIVERSITY_WEIGHT, min(CAPTION_DIVERSITY_WEIGHT, raw_pen))

        n_cap_clusters = len(set(caption_shares))
        n_eq_clusters = len(set(easy_q_shares))
        n_hq_clusters = len(set(hard_q_shares))
        print(f'[proposer_reward] Per-field diversity from {n_valid} proposals: '
              f'caption={n_cap_clusters} clusters (w={W_CAPTION}), '
              f'easy_q={n_eq_clusters} clusters (w={W_EASY_Q}), '
              f'hard_q={n_hq_clusters} clusters (w={W_HARD_Q}), '
              f'budget={CAPTION_DIVERSITY_WEIGHT}')
    else:
        diversity_penalties = {idx: 0.0 for idx in valid_indices}

    # --- 5c: Image visual diversity (resize + cosine distance clustering) ---
    IMG_DIVERSITY_WEIGHT = float(os.environ.get("IMG_DIVERSITY_WEIGHT", "0.15"))
    # Collect one representative image per proposal (first successful render)
    img_div_indices = []  # proposal indices that have at least one image
    img_div_b64 = []
    for idx in valid_indices:
        imgs = rendered_per_proposal.get(idx, [])
        if imgs:
            img_div_indices.append(idx)
            img_div_b64.append(imgs[0])

    if len(img_div_b64) > 1 and _HAS_PIL:
        img_shares = image_cluster_shares(img_div_b64, distance_threshold=0.3)
        uniform_img = 1.0 / len(img_div_b64)
        img_diversity_penalties = {}
        for i, idx in enumerate(img_div_indices):
            raw_pen = (img_shares[i] - uniform_img) * len(img_div_b64) * IMG_DIVERSITY_WEIGHT
            img_diversity_penalties[idx] = max(-IMG_DIVERSITY_WEIGHT, min(IMG_DIVERSITY_WEIGHT, raw_pen))
        n_img_clusters = len(set(img_shares))
        print(f'[proposer_reward] Image diversity: {n_img_clusters} visual clusters from '
              f'{len(img_div_b64)} images (weight={IMG_DIVERSITY_WEIGHT})')
    else:
        img_diversity_penalties = {}
        print(f'[proposer_reward] Image diversity: skipped (need ≥2 images and PIL)')

    # --- 5d: Compute final scores ---
    scores = []
    for idx in range(n):
        fields = all_fields[idx]

        if fields is None:
            scores.append({
                "overall": -1.0,
                "format": 0.0,
                "num_rendered": 0,
                "num_rollouts": 0,
                "avg_solvability": 0.0,
                "avg_difficulty": 0.0,
                "vt_diversity_bonus": 0.0,
                "caption_diversity_bonus": 0.0,
                "img_diversity_bonus": 0.0,
            })
        else:
            N = num_codegen_rollouts.get(idx, 8)
            successful_images = rendered_per_proposal.get(idx, [])
            easy_scores = easy_scores_per_proposal.get(idx, [])
            hard_scores = hard_scores_per_proposal.get(idx, [])
            num_rendered = len(successful_images)

            contribution_sum = 0.0
            for solv, diff in zip(easy_scores, hard_scores):
                contribution_sum += solv + diff
            final_score = contribution_sum / N

            # Visual type diversity bonus
            vt = fields.get("visual_type", "matplotlib")
            vt_frac = type_fractions.get(vt, 0.25)
            raw_bonus = (0.25 / max(vt_frac, 0.01)) - 1.0
            vt_bonus = max(-VT_DIVERSITY_MAX_BONUS, min(VT_DIVERSITY_MAX_BONUS, raw_bonus))
            final_score += vt_bonus

            # Caption/question diversity bonus (negative penalty = bonus for unique proposals)
            cap_div = -diversity_penalties.get(idx, 0.0)
            final_score += cap_div

            # Image visual diversity bonus
            img_div = -img_diversity_penalties.get(idx, 0.0)
            final_score += img_div

            avg_solv = np.mean(easy_scores) if easy_scores else 0.0
            avg_diff = np.mean(hard_scores) if hard_scores else 0.0

            scores.append({
                "overall": final_score,
                "format": 1.0,
                "num_rendered": num_rendered,
                "num_rollouts": N,
                "avg_solvability": float(avg_solv),
                "avg_difficulty": float(avg_diff),
                "vt_diversity_bonus": float(vt_bonus),
                "caption_diversity_bonus": float(cap_div),
                "img_diversity_bonus": float(img_div),
            })

    # Log summary
    valid = [s for s in scores if s["format"] > 0]
    print(f'[proposer_reward] === SCORING SUMMARY ===')
    print(f'  format_ok:        {len(valid)}/{n}')
    if valid:
        print(f'  avg_rendered:     {np.mean([s["num_rendered"] for s in valid]):.2f} / {valid[0]["num_rollouts"]}')
        print(f'  avg_solvability:  {np.mean([s["avg_solvability"] for s in valid]):.3f}')
        print(f'  avg_difficulty:   {np.mean([s["avg_difficulty"] for s in valid]):.3f}')
        print(f'  avg_vt_bonus:     {np.mean([s["vt_diversity_bonus"] for s in valid]):.3f}')
        print(f'  avg_caption_div:  {np.mean([s["caption_diversity_bonus"] for s in valid]):.3f}')
        print(f'  avg_img_div:      {np.mean([s["img_diversity_bonus"] for s in valid]):.3f}')
    print(f'  avg_overall:      {np.mean([s["overall"] for s in scores]):.3f}')

    return scores
