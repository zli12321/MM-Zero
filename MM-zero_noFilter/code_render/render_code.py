#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SVG-Only Code Renderer for ImageFree Self-Play.

All snippets are raw SVG markup, converted to PNG via cairosvg.
Can also be imported for its `render_single()` function (used by proposer_reward.py, codegen_reward.py).
"""

import subprocess
import tempfile
import base64
import json
import os
import argparse
import signal
import time
import warnings
from typing import Optional, List, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

STORAGE_PATH = os.getenv("STORAGE_PATH")

VALID_VISUAL_TYPES = ("svg",)


class _RenderTimeout(BaseException):
    """Raised by SIGALRM to forcefully escape ``except Exception`` blocks in generated code."""
    pass


def _render_timeout_handler(signum, frame):
    raise _RenderTimeout("render execution timed out")


# Per-worker state (set in child process by _render_worker_init)
_worker_cairosvg = None


def _render_worker_init() -> None:
    """Run once per worker process: import cairosvg for SVG-to-PNG conversion."""
    global _worker_cairosvg
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        import cairosvg
        _worker_cairosvg = cairosvg
    except Exception:
        _worker_cairosvg = None


def _render_worker_one(args: Tuple[str, str, int]) -> Optional[str]:
    """Run in a long-lived worker: convert one SVG snippet to PNG.
    args: (code_str, visual_type, timeout_sec). Returns base64 PNG or None."""
    code_str, visual_type, timeout_sec = args
    if not code_str or not code_str.strip():
        return None

    prev_handler = signal.signal(signal.SIGALRM, _render_timeout_handler)
    signal.alarm(timeout_sec)
    try:
        return _render_worker_exec(code_str)
    except _RenderTimeout:
        return None
    except MemoryError:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


# Qwen2.5-VL rejects images with aspect ratio >= 200.
_MAX_ASPECT_RATIO = 100
_MAX_IMAGE_DIM = 16384


def _validate_and_encode_png(png_path: str) -> Optional[str]:
    """Read a PNG, check its dimensions, and return base64 — or None if invalid."""
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(png_path) as img:
            w, h = img.size
            if w == 0 or h == 0:
                return None
            ratio = max(w, h) / max(min(w, h), 1)
            if ratio > _MAX_ASPECT_RATIO:
                print(f"  [render] Rejecting image: extreme aspect ratio {ratio:.1f} ({w}x{h})")
                return None
            if w > _MAX_IMAGE_DIM or h > _MAX_IMAGE_DIM:
                print(f"  [render] Rejecting image: dimension too large ({w}x{h})")
                return None
    except Exception:
        pass

    with open(png_path, "rb") as f:
        data = f.read()
    if not data:
        return None
    return base64.b64encode(data).decode("utf-8")


def _render_worker_exec(code_str: str) -> Optional[str]:
    """Core SVG rendering logic, called by _render_worker_one under alarm protection."""
    global _worker_cairosvg
    if _worker_cairosvg is None:
        try:
            import cairosvg
            _worker_cairosvg = cairosvg
        except Exception:
            return None

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "output.png")
        svg_path = os.path.join(tmpdir, "input.svg")
        code_strip = code_str.strip()
        if code_strip.lstrip().startswith("<") or "<?xml" in code_strip[:100]:
            with open(svg_path, "w") as f:
                f.write(code_str)
        else:
            return None
        try:
            _worker_cairosvg.svg2png(url=svg_path, write_to=output_path)
        except Exception:
            return None
        if os.path.exists(output_path):
            return _validate_and_encode_png(output_path)
    return None


def render_single(
    code_str: str,
    timeout: int = 30,
    visual_type: Optional[str] = None,
) -> Optional[str]:
    """Convert raw SVG to PNG. Returns base64-encoded PNG or None.

    Args:
        code_str: Raw SVG string.
        timeout: Maximum execution time in seconds.
        visual_type: Ignored (always SVG).

    Returns:
        Base64-encoded PNG string, or None if conversion fails.
    """
    if not code_str or not code_str.strip():
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "output.png")
        svg_path = os.path.join(tmpdir, "input.svg")
        code_strip = code_str.strip()
        if code_strip.lstrip().startswith("<") or "<?xml" in code_strip[:100]:
            with open(svg_path, 'w') as f:
                f.write(code_str)
        else:
            return None
        try:
            import cairosvg
            cairosvg.svg2png(url=svg_path, write_to=output_path)
            if os.path.exists(output_path):
                return _validate_and_encode_png(output_path)
        except Exception as e:
            print(f"  [render] cairosvg error: {e}")

    return None


def render_batch_codes(
    tasks: List[Tuple[str, Optional[str]]],
    max_workers: int = 8,
    timeout: int = 30,
    use_process_pool: bool = True,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> List[Optional[str]]:
    """Render many (code_str, visual_type) SVG snippets in parallel.

    Args:
        tasks: List of (code_str, visual_type) with visual_type ignored (always SVG).
        max_workers: Number of parallel workers.
        timeout: Per-item timeout in seconds.
        use_process_pool: If True (default), use ProcessPoolExecutor.
        progress_callback: Optional callable(done, total, success_count).

    Returns:
        List of base64 PNG strings or None, same length as tasks.
    """
    if not tasks:
        return []

    total = len(tasks)
    payloads = [(code, "svg", timeout) for code, vt in tasks]
    results = [None] * len(payloads)
    done_count = 0
    success_count = 0
    MIN_SUCCESS_FOR_CAP = 2000
    success_target = max(1, int(total * 0.7))
    done_target = max(1, int(total * 0.9))
    tail_timeout_seconds = 180
    tail_start_time = None
    done_tail_start_time = None
    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_render_worker_init,
        max_tasks_per_child=500,
    )
    try:
        future_to_idx = {executor.submit(_render_worker_one, p): i for i, p in enumerate(payloads)}
        pending_futures = set(future_to_idx.keys())
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            pending_futures.discard(future)
            try:
                results[idx] = future.result(timeout=timeout + 5)
            except Exception:
                results[idx] = None
            done_count += 1
            if results[idx] is not None:
                success_count += 1
            if progress_callback is not None:
                progress_callback(done_count, total, success_count)
            if tail_start_time is None and (
                success_count >= success_target or success_count >= MIN_SUCCESS_FOR_CAP
            ):
                tail_start_time = time.time()
            if done_count >= done_target and done_tail_start_time is None:
                done_tail_start_time = time.time()
            now = time.time()
            success_tail_expired = (
                tail_start_time is not None and (now - tail_start_time) >= tail_timeout_seconds
            )
            done_tail_expired = (
                done_tail_start_time is not None and (now - done_tail_start_time) >= tail_timeout_seconds
            )
            if pending_futures and (success_tail_expired or done_tail_expired):
                for pf in pending_futures:
                    pf_idx = future_to_idx[pf]
                    results[pf_idx] = None
                    pf.cancel()
                if progress_callback is not None and done_count < total:
                    progress_callback(total, total, success_count)
                break
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return results


def render_batch(
    items: list,
    max_workers: int = 8,
    timeout: int = 30,
    use_process_pool: bool = True,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> list:
    """Render a batch of SVG snippets in parallel.

    Args:
        items: List of dicts, each with a 'generated_code' key.
        max_workers: Number of parallel rendering workers.
        timeout: Per-item timeout in seconds.
        use_process_pool: If True, use ProcessPoolExecutor.
        progress_callback: Optional (done, total, success) callback.

    Returns:
        The same list with an 'image_base64' field added to each item.
    """
    if not items:
        return items
    tasks = [(item.get("generated_code", ""), "svg") for item in items]
    b64_list = render_batch_codes(
        tasks, max_workers=max_workers, timeout=timeout, use_process_pool=use_process_pool,
        progress_callback=progress_callback,
    )
    for i, item in enumerate(items):
        item["image_base64"] = b64_list[i] if i < len(b64_list) else None
    return items


def main(args):
    """Main CLI: load generated SVG JSONs, render to PNG, save combined output."""
    input_dir = f"{STORAGE_PATH}/generated_code"
    output_dir = f"{STORAGE_PATH}/rendered_images"
    os.makedirs(output_dir, exist_ok=True)

    all_items = []
    for i in range(8):
        input_file = os.path.join(input_dir, f"{args.experiment_name}_{i}.json")
        if os.path.exists(input_file):
            with open(input_file, 'r') as f:
                data = json.load(f)
                all_items.extend(data)
            print(f"Loaded {len(data)} items from {input_file}")
        else:
            print(f"File not found (skipping): {input_file}")

    if not all_items:
        print("ERROR: No code generation results found. Exiting.")
        return

    frac_str = os.environ.get("RENDER_FRACTION", "").strip()
    if frac_str:
        try:
            frac = float(frac_str)
            if 0 < frac < 1:
                cap = max(1, int(len(all_items) * frac))
                all_items = all_items[:cap]
                print(f"RENDER_FRACTION={frac}: rendering first {len(all_items)} items.")
        except ValueError:
            pass

    print(f"Total items to render: {len(all_items)}")
    start_time = time.time()

    report_every = max(1, len(all_items) // 20)
    last_pct = -1

    def cli_progress(done: int, total: int, success: int) -> None:
        nonlocal last_pct
        pct = (100 * done) // total if total else 0
        if done == total or done % report_every == 0 or pct >= last_pct + 5:
            last_pct = pct
            print(f"[render] progress: {done}/{total} ({pct}%) — {success} OK", flush=True)

    all_items = render_batch(
        all_items,
        max_workers=args.workers,
        timeout=args.timeout,
        progress_callback=cli_progress,
    )

    elapsed = time.time() - start_time
    success_count = sum(1 for item in all_items if item.get("image_base64"))
    fail_count = len(all_items) - success_count
    print(f"Rendering complete in {elapsed:.1f}s: {success_count} success, {fail_count} failed")

    rendered_items = []
    for item in all_items:
        b64 = item.get("image_base64")
        if b64 and item.get("hard_question") and item.get("hard_answer"):
            image_str = b64 if isinstance(b64, str) else b64.decode("utf-8")
            rendered_items.append({
                "caption": item.get("caption", ""),
                "easy_question": item.get("easy_question", ""),
                "easy_answer": item.get("easy_answer", ""),
                "hard_question": item.get("hard_question", ""),
                "hard_answer": item.get("hard_answer", ""),
                "image": image_str,
                "generated_code": item.get("generated_code", ""),
            })

    print(f"Successfully rendered items with valid questions: {len(rendered_items)}")

    output_file = os.path.join(output_dir, f"{args.experiment_name}_rendered.json")
    with open(output_file, 'w') as f:
        json.dump(rendered_items, f, indent=2)
    print(f"Saved rendered results to {output_file}")

    shard_size = len(rendered_items) // 8 + 1
    for i in range(8):
        shard = rendered_items[i * shard_size: (i + 1) * shard_size]
        shard_file = os.path.join(output_dir, f"{args.experiment_name}_{i}.json")
        with open(shard_file, 'w') as f:
            json.dump(shard, f, indent=2)
        print(f"  Shard {i}: {len(shard)} items -> {shard_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render generated SVG to PNG images.")
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Experiment name (matches code_generate output naming)")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("RENDER_MAX_WORKERS", "16")),
                        help="Number of parallel rendering workers")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Per-item rendering timeout in seconds")
    args = parser.parse_args()

    main(args)

    # Force-exit to avoid hanging in atexit handlers
    os._exit(0)
