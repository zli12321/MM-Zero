#!/usr/bin/env python3
"""
Plot comprehensive co-evolution metrics for the SVG self-play pipeline.

Usage:
  python3 plot_coevolution.py \
      --storage_dirs /workspace/selfAgent_Storage_svg_long_round2 \
      --accuracy_file /workspace/selfAgent_Storage_svg_long_round2/eval_responses/accuracy_summary.jsonl \
      --model_name "Qwen3-VL-8B-Instruct-ImageFree-SVG"

  python3 plot_coevolution.py \
      --storage_dirs /workspace/selfAgent_Storage_svg_long_round3_filter \
      --model_name "Qwen3-VL-8B-Instruct-ImageFree-SVG (filtered)"

  python3 plot_coevolution.py \
      --storage_dirs /workspace/selfAgent_Storage_svg_long_round3_filter \
      --model_name "Qwen3-VL-8B-Instruct-ImageFree-SVG (filtered)"

  python3 plot_coevolution.py \
      --storage_dirs /workspace/selfAgent_Storage_4b_download_test \
      --model_name "Qwen3-VL-4B-Instruct-ImageFree-SVG (filtered)"
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

def _normalized_entropy(counter):
    """Compute normalized Shannon entropy from a Counter. Returns 0-1."""
    total = sum(counter.values())
    if total == 0:
        return None
    n_classes = len(counter)
    if n_classes <= 1:
        return 0.0
    probs = np.array([v / total for v in counter.values()], dtype=float)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(n_classes)
    return float(entropy / max_entropy)


CONTENT_TYPE_COLORS = {
    'data_chart': '#4C72B0',
    'diagram':    '#55A868',
    'geometry':   '#C44E52',
    'other':      '#8172B2',
}


def load_metrics_from_storage(storage_dirs):
    """Extract all metrics from saved pipeline data.

    Returns list of dicts per solver version with keys:
        label, round, storage, avg_difficulty, avg_solvability,
        render_rate, n_rendered, n_proposals, easy_hard_gap,
        hard_consistency_values, question_type_dist, content_type_dist,
        avg_code_length, n_training_samples, n_codegen_filtered
    """
    results = []

    for storage in storage_dirs:
        basename = os.path.basename(storage.rstrip('/'))
        ri_dir = os.path.join(storage, 'rendered_images')
        gc_dir = os.path.join(storage, 'generated_code')
        gp_dir = os.path.join(storage, 'generated_proposals')

        if not os.path.isdir(ri_dir):
            print(f"  WARNING: {ri_dir} not found, skipping", file=sys.stderr)
            continue

        results_files = sorted(glob.glob(os.path.join(ri_dir, '*_results.json')))
        versions = {}
        for rf in results_files:
            m = re.search(r'solver_(v\d+)_(\d+)_results', os.path.basename(rf))
            if m:
                ver = m.group(1)
                versions.setdefault(ver, []).append(rf)

        for ver in sorted(versions.keys(), key=lambda x: int(x[1:])):
            all_hard_cons = []
            all_easy_cons = []
            question_types = []

            for rf in versions[ver]:
                with open(rf) as f:
                    data = json.load(f)
                for item in data:
                    hard_c = item.get('hard_consistency')
                    easy_c = item.get('easy_consistency')
                    if hard_c is None and 'score' in item:
                        hard_c = item['score']
                    if hard_c is not None:
                        all_hard_cons.append(float(hard_c))
                    if easy_c is not None:
                        all_easy_cons.append(float(easy_c))
                    question_types.append(item.get('question_type', 'unknown'))

            # --- Render success rate ---
            prefix = os.path.basename(versions[ver][0]).rsplit('_0_results.json', 1)[0]
            rendered_file = os.path.join(ri_dir, f'{prefix}_rendered.json')
            n_rendered = 0
            if os.path.exists(rendered_file):
                with open(rendered_file) as f:
                    n_rendered = len(json.load(f))

            n_proposals = 0
            code_lengths = []
            if os.path.isdir(gc_dir):
                for i in range(8):
                    gc_file = os.path.join(gc_dir, f'{prefix}_{i}.json')
                    if os.path.exists(gc_file):
                        with open(gc_file) as f:
                            items = json.load(f)
                        n_proposals += len(items)
                        for item in items:
                            code_lengths.append(len(item.get('generated_code', '')))

            # --- Content type distribution from solver proposals ---
            content_types = []
            if os.path.isdir(gp_dir):
                for i in range(8):
                    pf = os.path.join(gp_dir, f'{prefix}_{i}.json')
                    if os.path.exists(pf):
                        with open(pf) as f:
                            items = json.load(f)
                        for item in items:
                            content_types.append(item.get('content_type', 'unknown'))

            # --- CodeGen filtered training data (parquet) ---
            ver_num = int(ver[1:])
            codegen_prefix = prefix.replace('solver_', 'codegen_')
            parquet_path = os.path.join(gp_dir, f'{codegen_prefix}_proposals.parquet')
            n_codegen_filtered = None
            if os.path.exists(parquet_path):
                try:
                    import pandas as pd
                    df = pd.read_parquet(parquet_path)
                    n_codegen_filtered = len(df)
                except Exception:
                    pass

            # --- Solver filtered training data (parquet) ---
            solver_parquet = os.path.join(gp_dir, f'{prefix}_proposals.parquet')
            n_solver_filtered = None
            if os.path.exists(solver_parquet):
                try:
                    import pandas as pd
                    df = pd.read_parquet(solver_parquet)
                    n_solver_filtered = len(df)
                except Exception:
                    pass

            # --- Compute aggregates ---
            hard_arr = np.array(all_hard_cons) if all_hard_cons else np.array([])
            easy_arr = np.array(all_easy_cons) if all_easy_cons else np.array([])

            difficulty = float(np.mean(1.0 - np.abs(2.0 * hard_arr - 1.0))) if len(hard_arr) > 0 else None
            solvability = float(np.mean(easy_arr)) if len(easy_arr) > 0 else None
            render_rate = n_rendered / n_proposals if n_proposals > 0 else None
            avg_code_len = float(np.mean(code_lengths)) if code_lengths else None

            easy_hard_gap = None
            if len(easy_arr) > 0 and len(hard_arr) > 0:
                easy_hard_gap = float(np.mean(easy_arr) - np.mean(hard_arr))

            # --- Content type diversity (normalized Shannon entropy) ---
            ct_dist = Counter(content_types)
            content_diversity = _normalized_entropy(ct_dist)

            # --- Question type diversity ---
            qt_dist = Counter(question_types)
            question_diversity = _normalized_entropy(qt_dist)

            entry = {
                'label': ver,
                'round': basename,
                'storage': storage,
                'avg_difficulty': difficulty,
                'avg_solvability': solvability,
                'render_rate': render_rate,
                'n_rendered': n_rendered,
                'n_proposals': n_proposals,
                'n_evaluated': len(all_hard_cons),
                'easy_hard_gap': easy_hard_gap,
                'hard_consistency_values': all_hard_cons,
                'question_type_dist': dict(qt_dist),
                'content_type_dist': dict(ct_dist),
                'content_diversity': content_diversity,
                'question_diversity': question_diversity,
                'avg_code_length': avg_code_len,
                'n_codegen_filtered': n_codegen_filtered,
                'n_solver_filtered': n_solver_filtered,
            }
            results.append(entry)

    return results


def load_accuracy(jsonl_path, datasets=None):
    """Load accuracy_summary.jsonl -> list of (model_label, avg_accuracy)."""
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by_model = defaultdict(dict)
    for r in rows:
        by_model[r["model"]][r["dataset"]] = float(r["accuracy"])

    if datasets is None:
        all_ds = set()
        for ds_dict in by_model.values():
            all_ds.update(ds_dict.keys())
        datasets = sorted(all_ds)

    results = []
    for model, ds_dict in by_model.items():
        accs = [ds_dict[ds] for ds in datasets if ds in ds_dict]
        if accs:
            results.append((model, sum(accs) / len(accs)))

    def sort_key(item):
        m = item[0]
        if m == "base":
            return (-1, 0)
        nums = re.findall(r'\d+', m)
        if nums:
            return (int(nums[0]), int(nums[-1]) if len(nums) > 1 else 0)
        return (999, 0)

    results.sort(key=sort_key)
    return results


def plot_all_metrics(metrics, accuracy_data, model_name, output_path):
    """Create a comprehensive multi-panel figure with all metrics."""
    n_iters = len(metrics)
    x = np.arange(n_iters)
    x_labels = [m['label'] for m in metrics]

    has_accuracy = len(accuracy_data) > 0
    has_gap = any(m['easy_hard_gap'] is not None for m in metrics)
    has_code_len = any(m['avg_code_length'] is not None for m in metrics)

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))

    # ── Panel 1: Difficulty + Solvability + Accuracy (dual y-axis) ──
    ax = axes[0, 0]
    diff_vals = [m['avg_difficulty'] if m['avg_difficulty'] is not None else np.nan for m in metrics]
    lines = []
    ln = ax.plot(x, diff_vals, 'o-', color='#E07020', linewidth=2, markersize=6, label='Difficulty')
    lines.extend(ln)
    for i, dv in enumerate(diff_vals):
        if not np.isnan(dv):
            ax.annotate(f'{dv:.2f}', (i, dv), textcoords="offset points",
                        xytext=(0, -12), ha='center', fontsize=7, color='#E07020')

    solv_vals = [m['avg_solvability'] if m['avg_solvability'] is not None else np.nan for m in metrics]
    if not all(np.isnan(v) for v in solv_vals):
        ln = ax.plot(x, solv_vals, 's--', color='#30A060', linewidth=2, markersize=5, label='Solvability')
        lines.extend(ln)
        for i, sv in enumerate(solv_vals):
            if not np.isnan(sv):
                ax.annotate(f'{sv:.2f}', (i, sv), textcoords="offset points",
                            xytext=(0, 8), ha='center', fontsize=7, color='#30A060')

    ax.set_ylabel('Score (0-1)', fontsize=10)
    ax.set_ylim(0, 1.05)

    if has_accuracy:
        ax2 = ax.twinx()
        acc_x, acc_y = [], []
        for label, acc in accuracy_data:
            m_ver = re.search(r'solver_v(\d+)', label)
            if m_ver:
                ver_num = int(m_ver.group(1))
                for i, m in enumerate(metrics):
                    if m['label'] == f'v{ver_num}':
                        acc_x.append(i)
                        acc_y.append(acc)
                        break
            elif label == 'base':
                acc_x.append(-0.3)
                acc_y.append(acc)
        if acc_x:
            ln = ax2.plot(acc_x, acc_y, 'D-', color='#3070C0', linewidth=2, markersize=5,
                          label='Accuracy (%)')
            lines.extend(ln)
            ax2.set_ylabel('Accuracy (%)', fontsize=10, color='#3070C0')
            ax2.tick_params(axis='y', labelcolor='#3070C0')

    ax.legend(lines, [l.get_label() for l in lines], loc='best', fontsize=8, framealpha=0.9)
    ax.set_title('Difficulty & Solvability', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel 2: Render Success Rate ──
    ax = axes[0, 1]
    render_vals = [m['render_rate'] if m['render_rate'] is not None else 0 for m in metrics]
    ax.plot(x, render_vals, 's-', color='#30A060', linewidth=2, markersize=6)
    for i, rv in enumerate(render_vals):
        ax.annotate(f'{rv:.1%}', (i, rv), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=7.5)
    ax.set_title('Render Success Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('Rate', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel 3: Easy-Hard Consistency Gap ──
    ax = axes[0, 2]
    if has_gap:
        gap_vals = [m['easy_hard_gap'] if m['easy_hard_gap'] is not None else np.nan for m in metrics]
        ax.plot(x, gap_vals, '^-', color='#9467BD', linewidth=2, markersize=6)
        for i, gv in enumerate(gap_vals):
            if not np.isnan(gv):
                ax.annotate(f'{gv:.3f}', (i, gv), textcoords="offset points",
                            xytext=(0, 8), ha='center', fontsize=7.5)
    ax.set_title('Easy−Hard Consistency Gap', fontsize=11, fontweight='bold')
    ax.set_ylabel('Easy − Hard Consistency', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.grid(True, alpha=0.2)
    if not has_gap:
        ax.text(0.5, 0.5, 'N/A\n(no easy_consistency)', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    # ── Panel 4: Hard Consistency Distribution (boxplot) ──
    ax = axes[0, 3]
    box_data = [m['hard_consistency_values'] for m in metrics if m['hard_consistency_values']]
    if box_data:
        bp = ax.boxplot(box_data, positions=x[:len(box_data)], widths=0.5,
                        patch_artist=True, showfliers=False, medianprops=dict(color='black'))
        for patch in bp['boxes']:
            patch.set_facecolor('#FFD580')
            patch.set_alpha(0.7)
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Ideal=0.5')
    ax.set_title('Hard Consistency Distribution', fontsize=11, fontweight='bold')
    ax.set_ylabel('Hard Consistency', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # ── Panel 5: Content Type Distribution + Diversity Score ──
    ax = axes[1, 0]
    all_ct = set()
    for m in metrics:
        all_ct.update(m['content_type_dist'].keys())
    content_types_sorted = sorted(all_ct)

    ct_data = {}
    for ct in content_types_sorted:
        ct_data[ct] = []
        for m in metrics:
            total = sum(m['content_type_dist'].values()) or 1
            ct_data[ct].append(m['content_type_dist'].get(ct, 0) / total)

    bottom = np.zeros(n_iters)
    for ct in content_types_sorted:
        color = CONTENT_TYPE_COLORS.get(ct, '#999999')
        ax.bar(x, ct_data[ct], bottom=bottom, label=ct, color=color, alpha=0.85, width=0.6)
        bottom += np.array(ct_data[ct])

    # Overlay diversity score as a line on right y-axis
    ax2_div = ax.twinx()
    div_vals = [m['content_diversity'] if m['content_diversity'] is not None else np.nan for m in metrics]
    ax2_div.plot(x, div_vals, 'D-', color='black', linewidth=2, markersize=5,
                 label='Diversity', zorder=5)
    for i, dv in enumerate(div_vals):
        if not np.isnan(dv):
            ax2_div.annotate(f'{dv:.2f}', (i, dv), textcoords="offset points",
                             xytext=(5, 5), ha='left', fontsize=7.5, fontweight='bold')
    ax2_div.set_ylabel('Diversity (norm. entropy)', fontsize=9)
    ax2_div.set_ylim(0, 1.15)

    ax.set_title('Content Type & Diversity', fontsize=11, fontweight='bold')
    ax.set_ylabel('Fraction', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2_div.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, fontsize=7, loc='upper right')

    # ── Panel 6: Filter Pass Rate Funnel ──
    ax = axes[1, 1]
    proposals_vals = [m['n_proposals'] for m in metrics]
    rendered_vals_n = [m['n_rendered'] for m in metrics]
    evaluated_vals = [m['n_evaluated'] for m in metrics]

    w = 0.25
    ax.bar(x - w, proposals_vals, width=w, label='Proposals', color='#A0C4E8', edgecolor='#5080B0')
    ax.bar(x, rendered_vals_n, width=w, label='Rendered', color='#80D080', edgecolor='#409040')
    ax.bar(x + w, evaluated_vals, width=w, label='Evaluated', color='#FFD080', edgecolor='#C09030')
    ax.set_title('Pipeline Funnel', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # ── Panel 7: Average Code Length ──
    ax = axes[1, 2]
    if has_code_len:
        cl_vals = [m['avg_code_length'] if m['avg_code_length'] is not None else 0 for m in metrics]
        ax.plot(x, cl_vals, 'o-', color='#D65F5F', linewidth=2, markersize=6)
        for i, cl in enumerate(cl_vals):
            if cl > 0:
                ax.annotate(f'{cl:.0f}', (i, cl), textcoords="offset points",
                            xytext=(0, 8), ha='center', fontsize=7.5)
    ax.set_title('Avg Code Length (chars)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Characters', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.grid(True, alpha=0.2)
    if not has_code_len:
        ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    # ── Panel 8: Training Data Volume ──
    ax = axes[1, 3]
    has_codegen_data = any(m['n_codegen_filtered'] is not None for m in metrics)
    has_solver_data = any(m['n_solver_filtered'] is not None for m in metrics)

    if has_codegen_data or has_solver_data:
        w = 0.3
        if has_codegen_data:
            cg_vals = [m['n_codegen_filtered'] if m['n_codegen_filtered'] is not None else 0 for m in metrics]
            ax.bar(x - w / 2, cg_vals, width=w, label='CodeGen Training', color='#7EB4D8', edgecolor='#4080B0')
            for i, v in enumerate(cg_vals):
                if v > 0:
                    ax.text(i - w / 2, v + 20, str(v), ha='center', fontsize=7.5)
        if has_solver_data:
            sv_vals = [m['n_solver_filtered'] if m['n_solver_filtered'] is not None else 0 for m in metrics]
            offset = w / 2 if has_codegen_data else 0
            ax.bar(x + offset, sv_vals, width=w, label='Solver Training', color='#FFB870', edgecolor='#C08030')
            for i, v in enumerate(sv_vals):
                if v > 0:
                    ax.text(i + offset, v + 20, str(v), ha='center', fontsize=7.5)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'N/A\n(no parquet files)', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')

    ax.set_title('Training Data Volume', fontsize=11, fontweight='bold')
    ax.set_ylabel('Filtered Samples', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle(model_name, fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def print_summary_table(metrics):
    """Print a comprehensive summary table to stdout."""
    header = (f"{'Iter':<6} {'Difficulty':>10} {'Solvability':>11} {'EH Gap':>8} "
              f"{'Render%':>8} {'Diversity':>9} {'Rendered':>8} {'Proposals':>9} "
              f"{'AvgCodeLen':>10} {'CG Train':>8} {'SV Train':>8}")
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'=' * len(header)}")
    for m in metrics:
        diff_s = f"{m['avg_difficulty']:.3f}" if m['avg_difficulty'] is not None else 'N/A'
        solv_s = f"{m['avg_solvability']:.3f}" if m['avg_solvability'] is not None else 'N/A'
        gap_s = f"{m['easy_hard_gap']:.3f}" if m['easy_hard_gap'] is not None else 'N/A'
        div_s = f"{m['content_diversity']:.3f}" if m['content_diversity'] is not None else 'N/A'
        rr_s = f"{m['render_rate']:.1%}" if m['render_rate'] is not None else 'N/A'
        cl_s = f"{m['avg_code_length']:.0f}" if m['avg_code_length'] is not None else 'N/A'
        cg_s = str(m['n_codegen_filtered']) if m['n_codegen_filtered'] is not None else 'N/A'
        sv_s = str(m['n_solver_filtered']) if m['n_solver_filtered'] is not None else 'N/A'
        print(f"{m['label']:<6} {diff_s:>10} {solv_s:>11} {gap_s:>8} "
              f"{rr_s:>8} {div_s:>9} {m['n_rendered']:>8} {m['n_proposals']:>9} "
              f"{cl_s:>10} {cg_s:>8} {sv_s:>8}")

        if m['content_type_dist']:
            total = sum(m['content_type_dist'].values()) or 1
            ct_str = ', '.join(f"{k}={v / total:.0%}" for k, v in
                               sorted(m['content_type_dist'].items(), key=lambda x: -x[1]))
            print(f"{'':>6}   content: {ct_str}")
        if m['question_type_dist']:
            qt_str = ', '.join(f"{k}={v}" for k, v in
                               sorted(m['question_type_dist'].items(), key=lambda x: -x[1]))
            print(f"{'':>6}   q_types: {qt_str}")
    print(f"{'=' * len(header)}\n")


def main():
    parser = argparse.ArgumentParser(description="Plot comprehensive SVG pipeline metrics")

    parser.add_argument("--storage_dirs", nargs='+', type=str,
                        help="Storage directories to scan for saved results (no GPU needed)")
    parser.add_argument("--accuracy_file", type=str,
                        help="Path to accuracy_summary.jsonl")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names to average (default: all)")
    parser.add_argument("--model_name", type=str, default="Model",
                        help="Model name for plot title")
    parser.add_argument("--output", type=str, default="coevolution_plot.png",
                        help="Output file path (default: <model_name>.png)")

    args = parser.parse_args()

    if not args.storage_dirs:
        print("ERROR: --storage_dirs is required.")
        sys.exit(1)

    # Default output to model_name (sanitized for filename) if not overridden
    if args.output == "coevolution_plot.png":
        safe_name = re.sub(r'[^\w\-.]', '_', args.model_name).strip('_')
        args.output = f"{safe_name or 'coevolution_plot'}.png"

    accuracy_data = []
    if args.accuracy_file:
        datasets = args.datasets.split(",") if args.datasets else None
        print(f"Loading accuracy from {args.accuracy_file}...")
        accuracy_data = load_accuracy(args.accuracy_file, datasets=datasets)
        for label, acc in accuracy_data:
            print(f"  {label}: {acc:.2f}%")

    print(f"\nScanning {len(args.storage_dirs)} storage dir(s)...")
    metrics = load_metrics_from_storage(args.storage_dirs)

    if not metrics:
        print("ERROR: No metrics found in the provided storage directories.")
        sys.exit(1)

    print_summary_table(metrics)
    plot_all_metrics(metrics, accuracy_data, args.model_name, args.output)


if __name__ == "__main__":
    main()
