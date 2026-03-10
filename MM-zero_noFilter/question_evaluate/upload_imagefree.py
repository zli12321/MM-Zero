#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Upload script for ImageFree Self-Play (no filtering).

Collects evaluated results from evaluate_imagefree.py and saves all valid items
as a parquet file for Solver GRPO training. No difficulty/consistency filtering.

Similar to upload.py but reads from rendered_images/ instead of generated_question/.
"""

import json
import pandas as pd
import argparse
import os
import glob

STORAGE_PATH = os.getenv("STORAGE_PATH")
print(f"STORAGE_PATH: {STORAGE_PATH}")

parser = argparse.ArgumentParser(description="Collect evaluated results and save as parquet for Solver training (no filtering).")
parser.add_argument("--output_dir", type=str, default="", help="Output directory for parquet files")
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

# Keep all valid items (no difficulty/consistency filtering).
filtered_datas = []
n_invalid = 0
for d in datas:
    raw = d.get('image') or ''
    image_str = raw.strip() if isinstance(raw, str) else ''
    is_valid_image = (
        len(image_str) > 100 and
        (image_str.startswith('data:image') or len(image_str) > 200)
    )

    if d.get('answer', '') in ['', 'None'] or not is_valid_image:
        n_invalid += 1
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

print(f"\nNo filtering applied (keeping all valid items).")
print(f"  Valid: {len(filtered_datas)} / {len(datas)}  (skipped {n_invalid} with missing answer or invalid image)")

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
        "total_valid_samples": len(filtered_datas),
        "experiment_name": args.save_name,
        "output_file": output_path,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Saved summary to {summary_path}")
else:
    print("WARNING: No valid data to save!")
