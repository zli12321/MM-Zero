#!/usr/bin/env python3
"""
Compare model accuracies vs base across datasets from an accuracy_summary.jsonl file.
Usage: python eval_accuracy_comparison.py <path_to_accuracy_summary.jsonl> [--exclude-datasets DS1,DS2,...]

Prints per-dataset and per-model deltas vs base, and whether each is improving or decreasing.
"""

# python eval_accuracy_comparison.py /path/to/llm_accuracy_summary.jsonl --exclude-datasets ChartQA,DocVQA,MathVista

import argparse
import json
import sys
from collections import defaultdict


def load_summary(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare model accuracies vs base across datasets from an accuracy_summary.jsonl file."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to accuracy_summary.jsonl (or llm_accuracy_summary.jsonl)",
    )
    parser.add_argument(
        "--exclude-datasets",
        type=str,
        default="",
        help="Comma-separated dataset names to exclude from comparison (default: none). E.g. ChartQA,DocVQA",
    )
    args = parser.parse_args()
    path = args.path
    exclude = {d.strip() for d in args.exclude_datasets.split(",") if d.strip()}
    if exclude:
        print(f"Excluding datasets: {sorted(exclude)}", file=sys.stderr)
        print(file=sys.stderr)

    rows = load_summary(path)

    def is_mmmu_pro_series(dataset_name: str) -> bool:
        d = (dataset_name or "").lower()
        return "mmmu" in d and "pro" in d

    # Index by (model, dataset) -> accuracy; also keep correct/total for micro avg
    by_model_dataset = defaultdict(dict)
    by_model_correct_total = defaultdict(lambda: [0, 0])  # model -> [correct, total]
    models_set = set()
    datasets_set = set()
    mmmu_pro_accs = defaultdict(list)  # model -> [acc1, acc2, ...] for MMMU-pro series

    for r in rows:
        dataset = r.get("dataset", "")
        if dataset in exclude:
            continue
        model = r.get("model", "")
        acc = float(r.get("accuracy", 0))
        correct = int(r.get("correct", 0))
        total = int(r.get("total", 0))
        if is_mmmu_pro_series(dataset):
            by_model_correct_total[model][0] += correct
            by_model_correct_total[model][1] += total
            mmmu_pro_accs[model].append(acc)
            models_set.add(model)
            continue
        by_model_dataset[model][dataset] = acc
        by_model_correct_total[model][0] += correct
        by_model_correct_total[model][1] += total
        models_set.add(model)
        datasets_set.add(dataset)

    # Replace MMMU-pro series with one "MMMU-pro (macro)" row per model
    if mmmu_pro_accs:
        datasets_set.add("MMMU-pro (macro)")
        for model, accs in mmmu_pro_accs.items():
            by_model_dataset[model]["MMMU-pro (macro)"] = sum(accs) / len(accs)
        print(f"Aggregated MMMU-pro series -> \"MMMU-pro (macro)\" (macro accuracy)", file=sys.stderr)
        print(file=sys.stderr)

    # Base is the one named "base"; others are trained steps/versions
    if "base" not in models_set:
        print("No 'base' model found in summary. Models present:", sorted(models_set), file=sys.stderr)
        sys.exit(1)

    base_accs = by_model_dataset["base"]
    other_models = sorted(m for m in models_set if m != "base")
    datasets = sorted(datasets_set)

    # ---- OVERALL (micro = total correct/total; macro = avg of per-dataset accuracies) ----
    print("=" * 80)
    print("OVERALL ACCURACY (vs base)")
    print("=" * 80)
    base_correct, base_total = by_model_correct_total["base"]
    base_micro = 100.0 * base_correct / base_total if base_total else 0
    base_macro = sum(base_accs.get(ds, 0) for ds in datasets) / len(datasets) if datasets else 0
    print(f"  Base:  micro = {base_micro:.2f}%  ({base_correct}/{base_total})   macro = {base_macro:.2f}%")
    print()
    for m in other_models:
        correct, total = by_model_correct_total[m]
        micro = 100.0 * correct / total if total else 0
        macro = sum(by_model_dataset[m].get(ds, 0) for ds in datasets) / len(datasets) if datasets else 0
        micro_delta = micro - base_micro
        macro_delta = macro - base_macro
        micro_sign = "+" if micro_delta >= 0 else ""
        macro_sign = "+" if macro_delta >= 0 else ""
        # Overall verdict from micro (total correct/total across all datasets)
        if micro_delta > 0.01:
            overall_verdict = "IMPROVING"
        elif micro_delta < -0.01:
            overall_verdict = "DECREASING"
        else:
            overall_verdict = "SAME"
        print(f"  {m}:  micro = {micro:.2f}% ({micro_sign}{micro_delta:.2f})   macro = {macro:.2f}% ({macro_sign}{macro_delta:.2f})   ->  OVERALL {overall_verdict}")
    print()

    # Per-dataset comparison: base vs each model
    print("=" * 80)
    print("ACCURACY COMPARISON vs BASE (by dataset)")
    print("=" * 80)
    print(f"{'Dataset':<25} {'Base %':>8}", end="")
    for m in other_models:
        print(f" {m:>10}", end="")
    print()
    print("-" * 80)

    for ds in datasets:
        base_a = base_accs.get(ds)
        if base_a is None:
            continue
        parts = [f"{ds:<25}", f"{base_a:>7.2f}%"]
        for m in other_models:
            a = by_model_dataset[m].get(ds)
            if a is None:
                parts.append(f"{'—':>10}")
            else:
                delta = a - base_a
                sign = "+" if delta >= 0 else ""
                parts.append(f"{a:>6.2f}% ({sign}{delta:.2f})")
        print(" ".join(parts))

    # Average row (macro over datasets)
    base_avg = sum(base_accs.get(ds, 0) for ds in datasets) / len(datasets) if datasets else 0
    parts = [f"{'Average':<25}", f"{base_avg:>7.2f}%"]
    for m in other_models:
        avg = sum(by_model_dataset[m].get(ds, 0) for ds in datasets) / len(datasets) if datasets else 0
        delta = avg - base_avg
        sign = "+" if delta >= 0 else ""
        parts.append(f"{avg:>6.2f}% ({sign}{delta:.2f})")
    print("-" * 80)
    print(" ".join(parts))

    # Per-model summary: improving vs decreasing
    print()
    print("=" * 80)
    print("TREND vs BASE (improving / decreasing / same)")
    print("=" * 80)

    for m in other_models:
        better = 0
        worse = 0
        same = 0
        deltas = []
        for ds in datasets:
            base_a = base_accs.get(ds)
            a = by_model_dataset[m].get(ds)
            if base_a is None or a is None:
                continue
            delta = a - base_a
            deltas.append(delta)
            if delta > 0.01:
                better += 1
            elif delta < -0.01:
                worse += 1
            else:
                same += 1
        avg_delta = sum(deltas) / len(deltas) if deltas else 0
        print(f"  {m}:  improving={better}  decreasing={worse}  same={same}  |  avg delta vs base = {avg_delta:+.2f}%")

    # One-line verdict per model
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    for m in other_models:
        better = sum(1 for ds in datasets if (by_model_dataset[m].get(ds) or 0) - (base_accs.get(ds) or 0) > 0.01)
        worse = sum(1 for ds in datasets if (base_accs.get(ds) or 0) - (by_model_dataset[m].get(ds) or 0) > 0.01)
        if better > worse:
            verdict = "IMPROVING"
        elif worse > better:
            verdict = "DECREASING"
        else:
            verdict = "MIXED/SAME"
        print(f"  {m}: {verdict} (better on {better} datasets, worse on {worse})")
    print()


if __name__ == "__main__":
    main()
