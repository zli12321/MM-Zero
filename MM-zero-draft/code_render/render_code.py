#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code Renderer for ImageFree Self-Play.

Supports multiple visual types: matplotlib, plotly, pillow, svg.
Each snippet is executed (or converted) in a sandboxed way to produce a PNG.

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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

STORAGE_PATH = os.getenv("STORAGE_PATH")

VALID_VISUAL_TYPES = ("matplotlib", "plotly", "pillow", "svg")


class _RenderTimeout(BaseException):
    """Raised by SIGALRM to forcefully escape ``except Exception`` blocks in generated code.

    Inherits from BaseException (not Exception) so that catch-all ``except Exception``
    inside exec'd user code cannot swallow it.
    """
    pass


def _render_timeout_handler(signum, frame):
    """SIGALRM handler: raises _RenderTimeout to abort a stuck render worker."""
    raise _RenderTimeout("render execution timed out")


# Per-worker state (set in child process by _render_worker_init)
_worker_plt = None
_worker_go = None
_worker_px = None
_worker_plotly = None
_worker_Image = None
_worker_ImageDraw = None
_worker_ImageFont = None
_worker_np = None
_worker_pd = None
_worker_cairosvg = None


def _detect_visual_type(code_str: str) -> str:
    """Infer visual type from code content when not provided."""
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


def _render_matplotlib(code_str: str, output_path: str, tmpdir: str, timeout: int, env: dict) -> bool:
    full_code = f'output_path = "{output_path}"\n' + code_str
    script_path = os.path.join(tmpdir, "render.py")
    with open(script_path, 'w') as f:
        f.write(full_code)
    env = {**env, 'MPLBACKEND': 'Agg'}
    result = subprocess.run(
        ['python', script_path],
        timeout=timeout, capture_output=True, text=True, env=env, cwd=tmpdir,
    )
    if result.returncode != 0 and result.stderr:
        print(f"  [render] matplotlib stderr: {(result.stderr[:400])}")
    return result.returncode == 0 and os.path.exists(output_path)


def _render_plotly(code_str: str, output_path: str, tmpdir: str, timeout: int, env: dict) -> bool:
    # Ensure fig.write_image(output_path) exists; append if no save
    if "write_image" not in code_str and "output_path" not in code_str:
        code_str = code_str.rstrip() + f'\nfig.write_image(output_path, engine="kaleido")\n'
    full_code = f'output_path = "{output_path}"\n' + code_str
    script_path = os.path.join(tmpdir, "render.py")
    with open(script_path, 'w') as f:
        f.write(full_code)
    result = subprocess.run(
        ['python', script_path],
        timeout=timeout, capture_output=True, text=True, env=env, cwd=tmpdir,
    )
    if result.returncode != 0 and result.stderr:
        print(f"  [render] plotly stderr: {(result.stderr[:400])}")
    return result.returncode == 0 and os.path.exists(output_path)


def _render_pillow(code_str: str, output_path: str, tmpdir: str, timeout: int, env: dict) -> bool:
    if "output_path" not in code_str and ".save(" not in code_str:
        code_str = code_str.rstrip() + f'\nimg.save(output_path)\n'
    full_code = f'output_path = "{output_path}"\n' + code_str
    script_path = os.path.join(tmpdir, "render.py")
    with open(script_path, 'w') as f:
        f.write(full_code)
    result = subprocess.run(
        ['python', script_path],
        timeout=timeout, capture_output=True, text=True, env=env, cwd=tmpdir,
    )
    if result.returncode != 0 and result.stderr:
        print(f"  [render] pillow stderr: {(result.stderr[:400])}")
    return result.returncode == 0 and os.path.exists(output_path)


def _render_worker_init() -> None:
    """Run once per worker process: import all rendering libs so we don't restart Python per snippet."""
    global _worker_plt, _worker_go, _worker_px, _worker_plotly
    global _worker_Image, _worker_ImageDraw, _worker_ImageFont
    global _worker_np, _worker_pd, _worker_cairosvg

    import logging
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*Locator attempting to generate.*")
    warnings.filterwarnings("ignore", message=".*More than 20 figures have been opened.*", category=RuntimeWarning)

    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["figure.max_open_warning"] = 0
    import matplotlib.pyplot as plt
    import numpy as np
    _worker_plt = plt
    _worker_np = np
    try:
        import pandas as pd
        _worker_pd = pd
    except Exception:
        _worker_pd = None
    try:
        import plotly
        import plotly.graph_objects as go
        import plotly.express as px
        _worker_plotly = plotly
        _worker_go = go
        _worker_px = px
    except Exception:
        _worker_plotly = None
        _worker_go = None
        _worker_px = None
    try:
        from PIL import Image, ImageDraw, ImageFont
        _worker_Image = Image
        _worker_ImageDraw = ImageDraw
        _worker_ImageFont = ImageFont
    except Exception:
        _worker_Image = None
        _worker_ImageDraw = None
        _worker_ImageFont = None
    try:
        import cairosvg
        _worker_cairosvg = cairosvg
    except Exception:
        _worker_cairosvg = None


def _render_worker_one(args: Tuple[str, str, int]) -> Optional[str]:
    """
    Run in a long-lived worker: execute one snippet in-process (no subprocess).
    args: (code_str, visual_type, timeout_sec). Returns base64 PNG or None.

    Enforces a hard per-task SIGALRM timeout so that infinite loops or runaway
    memory allocation in generated code cannot block the entire render pool.
    """
    global _worker_plt
    code_str, visual_type, timeout_sec = args
    if not code_str or not code_str.strip():
        return None
    vt = (visual_type or "matplotlib").strip().lower()
    if vt not in VALID_VISUAL_TYPES:
        vt = _detect_visual_type(code_str)

    # Hard per-task timeout via SIGALRM.  ProcessPoolExecutor runs each task
    # in the worker's main thread, so signal.alarm() is safe here.
    prev_handler = signal.signal(signal.SIGALRM, _render_timeout_handler)
    signal.alarm(timeout_sec)
    try:
        return _render_worker_exec(code_str, vt)
    except _RenderTimeout:
        # Alarm fired — clean up rendering state so the worker stays usable
        if _worker_plt is not None:
            try:
                _worker_plt.close("all")
            except Exception:
                pass
        return None
    except MemoryError:
        if _worker_plt is not None:
            try:
                _worker_plt.close("all")
            except Exception:
                pass
        return None
    finally:
        signal.alarm(0)  # cancel pending alarm
        signal.signal(signal.SIGALRM, prev_handler)


# Qwen2.5-VL rejects images with aspect ratio >= 200.  Use a conservative
# limit so we never feed an unusable image to the Solver.
_MAX_ASPECT_RATIO = 100
_MAX_IMAGE_DIM = 16384  # pixels; larger images are almost certainly wrong


def _validate_and_encode_png(png_path: str) -> Optional[str]:
    """Read a PNG, check its dimensions, and return base64 — or None if invalid.

    Rejects images with extreme aspect ratios (>_MAX_ASPECT_RATIO) or any
    dimension larger than _MAX_IMAGE_DIM.  These are almost certainly rendering
    artefacts from buggy generated code and would crash the Qwen2.5-VL
    image processor (aspect ratio must be < 200).
    """
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
        pass  # if PIL check fails, fall through and just encode

    with open(png_path, "rb") as f:
        data = f.read()
    if not data:
        return None
    return base64.b64encode(data).decode("utf-8")


def _render_worker_exec(code_str: str, vt: str) -> Optional[str]:
    """Core rendering logic, called by _render_worker_one under alarm protection."""
    global _worker_plt, _worker_go, _worker_px, _worker_plotly
    global _worker_Image, _worker_ImageDraw, _worker_ImageFont
    global _worker_np, _worker_pd, _worker_cairosvg

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "output.png")
        old_cwd = os.getcwd()
        try:
            try:
                os.chdir(tmpdir)
            except Exception:
                pass
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    if vt == "matplotlib" and _worker_plt is not None:
                        ns = {
                            "output_path": output_path,
                            "plt": _worker_plt,
                            "np": _worker_np,
                            "__builtins__": __builtins__,
                        }
                        if _worker_pd is not None:
                            ns["pd"] = _worker_pd
                        full = f'output_path = {repr(output_path)}\n' + code_str
                        if "savefig" not in code_str and "save_fig" not in code_str:
                            full += "\nplt.savefig(output_path, bbox_inches='tight')"
                        try:
                            exec(compile(full, "<render>", "exec"), ns)
                        finally:
                            _worker_plt.close("all")
                        return _find_and_encode_png(output_path, tmpdir)

                    elif vt == "plotly" and _worker_go is not None:
                        ns = {
                            "output_path": output_path,
                            "go": _worker_go,
                            "np": _worker_np,
                            "__builtins__": __builtins__,
                        }
                        if _worker_px is not None:
                            ns["px"] = _worker_px
                        if _worker_plotly is not None:
                            ns["plotly"] = _worker_plotly
                        if _worker_pd is not None:
                            ns["pd"] = _worker_pd
                        code = code_str.rstrip()
                        if "write_image" not in code and "output_path" not in code:
                            code += f'\nfig.write_image(output_path, engine="kaleido")'
                        full = f'output_path = {repr(output_path)}\n' + code
                        try:
                            exec(compile(full, "<render>", "exec"), ns)
                        finally:
                            if _worker_plt is not None:
                                _worker_plt.close("all")
                        return _find_and_encode_png(output_path, tmpdir)

                    elif vt == "pillow" and _worker_Image is not None:
                        ns = {
                            "output_path": output_path,
                            "Image": _worker_Image,
                            "ImageDraw": _worker_ImageDraw,
                            "ImageFont": _worker_ImageFont,
                            "np": _worker_np,
                            "__builtins__": __builtins__,
                        }
                        if _worker_pd is not None:
                            ns["pd"] = _worker_pd
                        code = code_str.rstrip()
                        if "output_path" not in code and ".save(" not in code:
                            code += "\nimg.save(output_path)"
                        full = f'output_path = {repr(output_path)}\n' + code
                        exec(compile(full, "<render>", "exec"), ns)
                        return _find_and_encode_png(output_path, tmpdir)

                    elif vt == "svg":
                        return _render_svg_in_worker(code_str, output_path, tmpdir)
                except Exception:
                    pass
        finally:
            try:
                os.chdir(old_cwd)
            except Exception:
                pass
    return None


def _find_and_encode_png(output_path: str, tmpdir: str) -> Optional[str]:
    """Find the output PNG (exact path or any .png in tmpdir) and encode it."""
    png_path = output_path
    if not os.path.exists(png_path):
        for f in os.listdir(tmpdir):
            if f.endswith(".png"):
                png_path = os.path.join(tmpdir, f)
                break
    if os.path.exists(png_path):
        return _validate_and_encode_png(png_path)
    return None


def _render_svg_in_worker(code_str: str, output_path: str, tmpdir: str) -> Optional[str]:
    """Helper for SVG inside worker — uses pre-imported cairosvg for speed."""
    global _worker_cairosvg
    if _worker_cairosvg is None:
        try:
            import cairosvg
            _worker_cairosvg = cairosvg
        except Exception:
            return None
    svg_path = os.path.join(tmpdir, "input.svg")
    code_strip = code_str.strip()
    if code_strip.lstrip().startswith("<") or "<?xml" in code_strip[:100]:
        with open(svg_path, "w") as f:
            f.write(code_str)
    else:
        out_svg = os.path.join(tmpdir, "out.svg")
        full_code = f'output_svg = "{out_svg}"\n' + code_str
        script_path = os.path.join(tmpdir, "render_svg.py")
        with open(script_path, "w") as f:
            f.write(full_code)
        r = subprocess.run(
            ["python", script_path],
            timeout=30, capture_output=True, text=True, cwd=tmpdir,
        )
        if r.returncode != 0 or not os.path.exists(out_svg):
            return None
        svg_path = out_svg
    _worker_cairosvg.svg2png(url=svg_path, write_to=output_path)
    if os.path.exists(output_path):
        return _validate_and_encode_png(output_path)
    return None


def _render_svg(code_str: str, output_path: str, tmpdir: str) -> bool:
    """Render raw SVG string to PNG using cairosvg (if available) or subprocess."""
    svg_path = os.path.join(tmpdir, "input.svg")
    code_strip = code_str.strip()
    # If it looks like raw SVG, write directly
    if code_strip.lstrip().startswith("<") or "<?xml" in code_strip[:100]:
        with open(svg_path, 'w') as f:
            f.write(code_str)
    else:
        # Assume Python that writes SVG to a file; run it and then convert
        out_svg = os.path.join(tmpdir, "out.svg")
        full_code = f'output_svg = "{out_svg}"\n' + code_str
        if "output_svg" not in code_str:
            full_code += '\n# Expect variable svg_content or file written to output_svg'
        script_path = os.path.join(tmpdir, "render_svg.py")
        with open(script_path, 'w') as f:
            f.write(full_code)
        result = subprocess.run(
            ['python', script_path],
            timeout=30, capture_output=True, text=True, cwd=tmpdir,
        )
        if result.returncode != 0 or not os.path.exists(out_svg):
            return False
        svg_path = out_svg
    try:
        import cairosvg
        cairosvg.svg2png(url=svg_path, write_to=output_path)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"  [render] cairosvg error: {e}")
        return False


def render_single(
    code_str: str,
    timeout: int = 30,
    visual_type: Optional[str] = None,
) -> Optional[str]:
    """
    Execute code (or render SVG) to produce a PNG. Returns base64-encoded PNG or None.

    Supports: matplotlib, plotly, pillow, svg.
    If visual_type is None, it is inferred from code content.

    Args:
        code_str: Python code (matplotlib/plotly/pillow) or raw SVG string.
        timeout: Maximum execution time in seconds.
        visual_type: One of matplotlib, plotly, pillow, svg. Inferred if None.

    Returns:
        Base64-encoded PNG string, or None if execution fails.
    """
    if not code_str or not code_str.strip():
        return None

    vt = (visual_type or "").strip().lower()
    if vt not in VALID_VISUAL_TYPES:
        vt = _detect_visual_type(code_str)

    env = dict(os.environ)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "output.png")
        ok = False

        try:
            if vt == "matplotlib":
                ok = _render_matplotlib(code_str, output_path, tmpdir, timeout, env)
            elif vt == "plotly":
                ok = _render_plotly(code_str, output_path, tmpdir, timeout, env)
            elif vt == "pillow":
                ok = _render_pillow(code_str, output_path, tmpdir, timeout, env)
            elif vt == "svg":
                ok = _render_svg(code_str, output_path, tmpdir)
            else:
                ok = _render_matplotlib(code_str, output_path, tmpdir, timeout, env)
        except subprocess.TimeoutExpired:
            print(f"  [render] Execution timed out after {timeout}s (type={vt})")
        except Exception as e:
            print(f"  [render] Unexpected error (type={vt}): {e}")

        if ok and os.path.exists(output_path):
            return _validate_and_encode_png(output_path)

    return None


def render_batch_codes(
    tasks: List[Tuple[str, Optional[str]]],
    max_workers: int = 8,
    timeout: int = 30,
    use_process_pool: bool = True,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> List[Optional[str]]:
    """
    Render many (code_str, visual_type) snippets in parallel.
    Uses long-lived worker processes (import matplotlib/plotly/PIL once per worker)
    to avoid the per-snippet process startup cost.

    Args:
        tasks: List of (code_str, visual_type) with visual_type optional.
        max_workers: Number of parallel workers.
        timeout: Per-item timeout in seconds (enforced via SIGALRM in each worker).
        use_process_pool: If True (default), use ProcessPoolExecutor with pre-imported libs (fast).
                          If False, use one subprocess per snippet (slow but isolated).
        progress_callback: Optional callable(done, total, success_count) called as each task completes.
                           Use to display progress in the training terminal.

    Returns:
        List of base64 PNG strings or None, same length as tasks.
    """
    if not tasks:
        return []

    total = len(tasks)

    if use_process_pool and max_workers >= 1:
        # Long-lived workers: each worker imports matplotlib/plotly/PIL once, then runs many snippets.
        payloads = [(code, (vt or "matplotlib").strip().lower() if vt else "matplotlib", timeout) for code, vt in tasks]
        results = [None] * len(payloads)
        done_count = 0
        success_count = 0
        # (1) Start 3-minute tail cap when: >=70% success OR >= MIN_SUCCESS_FOR_CAP images (e.g. 2000) so we have enough to train.
        # (2) Fallback: once 90% of tasks have completed, give the rest 3 minutes so we don't hang on stuck tasks.
        MIN_SUCCESS_FOR_CAP = 2000  # if we have this many OK images (even below 70%), start the time cap
        success_target_ratio = 0.7  # 70%
        success_target = max(1, int(total * success_target_ratio))
        done_target = max(1, int(total * 0.9))
        tail_timeout_seconds = 180  # 3 minutes
        tail_start_time = None       # set when success_count >= 70% or >= MIN_SUCCESS_FOR_CAP
        done_tail_start_time = None  # set when done_count >= 90% (fallback)
        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_render_worker_init,
            max_tasks_per_child=500,
        )
        tail_cancelled = False
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
                # Start 3-minute tail timer when >=70% success or we have enough images (e.g. 2000) to train
                if tail_start_time is None and (
                    success_count >= success_target or success_count >= MIN_SUCCESS_FOR_CAP
                ):
                    tail_start_time = time.time()
                # Fallback: start 3-minute timer when 90% of tasks have completed
                if done_count >= done_target and done_tail_start_time is None:
                    done_tail_start_time = time.time()
                # Cancel rest if either tail timer has expired
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
                    tail_cancelled = True
                    break
        finally:
            # Always use wait=False: all futures are already collected via
            # as_completed, so there is nothing useful to wait for.  Waiting
            # with wait=True can hang indefinitely if a worker process is stuck
            # in cleanup (matplotlib backend, free() corruption, etc.).
            executor.shutdown(wait=False, cancel_futures=True)
        return results

    # Fallback: one subprocess per snippet (original behavior)
    def _one(idx_task):
        idx, (code, vt) = idx_task
        return idx, render_single(code, timeout=timeout, visual_type=vt)

    MIN_SUCCESS_FOR_CAP = 2000
    success_target = max(1, int(total * 0.7))
    done_target = max(1, int(total * 0.9))
    tail_timeout_seconds = 180
    results = [None] * len(tasks)
    done_count = 0
    success_count = 0
    tail_start_time = None
    done_tail_start_time = None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_one, (i, t)): i for i, t in enumerate(tasks)}
        pending_futures = set(futures.keys())
        for future in as_completed(futures):
            idx, img_b64 = future.result()
            pending_futures.discard(future)
            results[idx] = img_b64
            done_count += 1
            if img_b64 is not None:
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
                    results[futures[pf]] = None
                    pf.cancel()
                if progress_callback is not None and done_count < total:
                    progress_callback(total, total, success_count)
                break
    return results


def render_batch(
    items: list,
    max_workers: int = 8,
    timeout: int = 30,
    use_process_pool: bool = True,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> list:
    """
    Render a batch of code snippets in parallel.

    Args:
        items: List of dicts, each with a 'generated_code' key (and optionally 'visual_type').
        max_workers: Number of parallel rendering workers.
        timeout: Per-item timeout in seconds.
        use_process_pool: If True, use long-lived process workers (faster). Set RENDER_USE_SUBPROCESS=1 to disable.
        progress_callback: Optional (done, total, success) callback for progress display.

    Returns:
        The same list with an 'image_base64' field added to each item.
    """
    if not items:
        return items
    tasks = [(item.get("generated_code", ""), item.get("visual_type")) for item in items]
    use_pool = use_process_pool and os.environ.get("RENDER_USE_SUBPROCESS", "").lower() not in ("1", "true", "yes")
    b64_list = render_batch_codes(
        tasks, max_workers=max_workers, timeout=timeout, use_process_pool=use_pool, progress_callback=progress_callback
    )
    for i, item in enumerate(items):
        item["image_base64"] = b64_list[i] if i < len(b64_list) else None
    return items


def main(args):
    """
    Main CLI: load generated code JSONs, render images, save combined output.
    """
    input_dir = f"{STORAGE_PATH}/generated_code"
    output_dir = f"{STORAGE_PATH}/rendered_images"
    os.makedirs(output_dir, exist_ok=True)

    # Collect all code generation result files for this experiment
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

    # Optional: render only a fraction of items (e.g. RENDER_FRACTION=0.3 for quick testing)
    frac_str = os.environ.get("RENDER_FRACTION", "").strip()
    if frac_str:
        try:
            frac = float(frac_str)
            if 0 < frac < 1:
                cap = max(1, int(len(all_items) * frac))
                all_items = all_items[:cap]
                print(f"RENDER_FRACTION={frac}: rendering first {len(all_items)} items (of original total).")
        except ValueError:
            pass

    print(f"Total items to render: {len(all_items)}")
    start_time = time.time()

    # Progress to terminal (throttle to ~20 updates)
    report_every = max(1, len(all_items) // 20)
    last_pct = -1

    def cli_progress(done: int, total: int, success: int) -> None:
        nonlocal last_pct
        pct = (100 * done) // total if total else 0
        if done == total or done % report_every == 0 or pct >= last_pct + 5:
            last_pct = pct
            print(f"[render] progress: {done}/{total} ({pct}%) — {success} OK", flush=True)

    # Render all code snippets
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

    # Filter to successfully rendered items only.
    # Downstream: evaluate_imagefree.py expects key "image" (one base64 PNG string per item).
    rendered_items = []
    for item in all_items:
        b64 = item.get("image_base64")
        if b64 and item.get("hard_question") and item.get("hard_answer"):
            # Ensure string for JSON; must be full base64 so parquet/dataset can decode
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

    # Save as JSON (for evaluate_imagefree.sh to consume); shards use same "image" key
    output_file = os.path.join(output_dir, f"{args.experiment_name}_rendered.json")
    with open(output_file, 'w') as f:
        json.dump(rendered_items, f, indent=2)
    print(f"Saved rendered results to {output_file}")

    # Also split into per-GPU files for parallel evaluation (8 shards)
    shard_size = len(rendered_items) // 8 + 1
    for i in range(8):
        shard = rendered_items[i * shard_size: (i + 1) * shard_size]
        shard_file = os.path.join(output_dir, f"{args.experiment_name}_{i}.json")
        with open(shard_file, 'w') as f:
            json.dump(shard, f, indent=2)
        print(f"  Shard {i}: {len(shard)} items -> {shard_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render generated matplotlib code to images.")
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Experiment name (matches code_generate output naming)")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("RENDER_MAX_WORKERS", "16")),
                        help="Number of parallel rendering workers (default: RENDER_MAX_WORKERS env var or 16)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Per-item rendering timeout in seconds")
    args = parser.parse_args()

    main(args)

    # Force-exit to avoid hanging in atexit handlers that try to join
    # ProcessPoolExecutor worker processes stuck in matplotlib cleanup.
    os._exit(0)
