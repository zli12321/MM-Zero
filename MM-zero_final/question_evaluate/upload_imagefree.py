#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Upload / Filter script for ImageFree Self-Play.

Collects evaluated results from evaluate_imagefree.py, filters by difficulty score,
and saves as a parquet file for Solver GRPO training.

Similar to upload.py but reads from rendered_images/ instead of generated_question/.
"""

import json
import pandas as pd
import argparse
import os
import glob

STORAGE_PATH = os.getenv("STORAGE_PATH")
print(f"STORAGE_PATH: {STORAGE_PATH}")

parser = argparse.ArgumentParser(description="Filter evaluated results and save as parquet for Solver training.")
parser.add_argument("--output_dir", type=str, default="", help="Output directory for parquet files")
parser.add_argument("--max_score", type=float, default=0.8, help="(Legacy) Maximum difficulty score")
parser.add_argument("--min_score", type=float, default=0.3, help="(Legacy) Minimum difficulty score")
parser.add_argument("--min_easy_consistency", type=float, default=0.5, help="Minimum easy question consistency (keep items above this)")
parser.add_argument("--min_hard_consistency", type=float, default=0.25, help="Minimum hard question consistency")
parser.add_argument("--max_hard_consistency", type=float, default=0.75, help="Maximum hard question consistency")
parser.add_argument("--save_name", type=str, required=True, help="Experiment name for input/output files")
args = parser.parse_args()

# Collect all result files from evaluate_imagefree.py
datas = []
result_files = glob.glob(f'{STORAGE_PATH}/rendered_images/{args.save_name}_*_results.json')
print(f"Found {len(result_files)} result files: {result_files}")

for file_path in result_files:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            datas.extend(data)
        print(f"Loaded {len(data)} samples from {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        continue

if not datas:
    print("ERROR: No data loaded. Exiting.")
    exit(1)

# Score statistics
scores = [d['score'] for d in datas]
easy_cons = [d.get('easy_consistency', -1) for d in datas]
hard_cons = [d.get('hard_consistency', -1) for d in datas]
has_consistency_fields = any(c >= 0 for c in easy_cons)
print(f"\nScore statistics:")
print(f"  Total items: {len(scores)}")
print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")
print(f"  Average score: {sum(scores)/len(scores):.4f}")
if has_consistency_fields:
    valid_easy = [c for c in easy_cons if c >= 0]
    valid_hard = [c for c in hard_cons if c >= 0]
    if valid_easy:
        print(f"  Easy consistency: avg={sum(valid_easy)/len(valid_easy):.4f}, range=[{min(valid_easy):.4f}, {max(valid_easy):.4f}]")
    if valid_hard:
        print(f"  Hard consistency: avg={sum(valid_hard)/len(valid_hard):.4f}, range=[{min(valid_hard):.4f}, {max(valid_hard):.4f}]")

# Save score distribution plot
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, edgecolor='black')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title(f'Score Distribution: {args.save_name}')
    plt.savefig(f'{STORAGE_PATH}/rendered_images/{args.save_name}_scores_distribution.png',
                dpi=100, bbox_inches='tight')
    plt.close()
    print("Score distribution plot saved.")
except Exception as e:
    print(f"Could not save score distribution plot: {e}")

# Filter for Solver training.
# Consistency = fraction of rollouts agreeing with majority (no ground-truth comparison).
# If easy_consistency / hard_consistency fields are present:
#   - easy_consistency > min_easy_consistency (model agrees on easy Q → image renders correctly)
#   - min_hard_consistency <= hard_consistency <= max_hard_consistency (challenging but not impossible)
# Fallback to legacy score-based filtering if consistency fields are missing.
filtered_datas = []
n_easy_fail = 0
n_hard_low = 0
n_hard_high = 0
for d in datas:
    raw = d.get('image') or ''
    image_str = raw.strip() if isinstance(raw, str) else ''
    is_valid_image = (
        len(image_str) > 100 and
        (image_str.startswith('data:image') or len(image_str) > 200)
    )

    if d.get('answer', '') in ['', 'None'] or not is_valid_image:
        continue

    if has_consistency_fields:
        easy_con = d.get('easy_consistency', 0.0)
        hard_con = d.get('hard_consistency', 0.0)
        if easy_con < args.min_easy_consistency:
            n_easy_fail += 1
            continue
        if hard_con < args.min_hard_consistency:
            n_hard_low += 1
            continue
        if hard_con > args.max_hard_consistency:
            n_hard_high += 1
            continue
    else:
        if d['score'] < args.min_score or d['score'] > args.max_score:
            continue

    filtered_datas.append({
        'problem': d['question'],
        'answer': d['answer'],
        'score': d['score'],
        'hard_consistency': d.get('hard_consistency', -1),
        'easy_consistency': d.get('easy_consistency', -1),
        'images': image_str,
        'problem_type': d.get('question_type', 'numerical'),
        'caption': d.get('caption', ''),
        'easy_question': d.get('easy_question', ''),
        'easy_answer': d.get('easy_answer', ''),
    })

if has_consistency_fields:
    print(f"\nFiltering: easy_consistency > {args.min_easy_consistency}, hard_consistency in [{args.min_hard_consistency}, {args.max_hard_consistency}]")
    print(f"  Rejected: {n_easy_fail} (easy too low) + {n_hard_low} (hard too low) + {n_hard_high} (hard too high)")
    print(f"  Kept: {len(filtered_datas)} / {len(datas)}")
else:
    print(f"\n(Legacy mode: no consistency fields found, filtering by score in [{args.min_score}, {args.max_score}])")
    print(f"  Kept: {len(filtered_datas)} / {len(datas)}")

if filtered_datas:
    # Determine output path
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{args.save_name}_train.parquet")
    else:
        output_dir = f"{STORAGE_PATH}/local_parquet"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{args.save_name}_train.parquet")

    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(filtered_datas)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(filtered_datas)} samples to {output_path}")

    # Save summary
    summary_path = output_path.replace('.parquet', '_summary.json')
    summary = {
        "total_raw_samples": len(datas),
        "total_filtered_samples": len(filtered_datas),
        "score_range": [args.min_score, args.max_score],
        "experiment_name": args.save_name,
        "output_file": output_path,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Saved summary to {summary_path}")
else:
    print("WARNING: No data to save after filtering!")
