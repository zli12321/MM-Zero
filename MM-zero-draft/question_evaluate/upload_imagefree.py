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
parser.add_argument("--max_score", type=float, default=0.8, help="Maximum difficulty score (filter out too-easy questions)")
parser.add_argument("--min_score", type=float, default=0.3, help="Minimum difficulty score (filter out too-hard questions)")
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
print(f"\nScore statistics:")
print(f"  Total items: {len(scores)}")
print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")
print(f"  Average score: {sum(scores)/len(scores):.4f}")

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

# Filter by difficulty score range.
# For Solver training, we want questions that are neither too easy nor too hard.
# Parquet column "images" = one base64 string per row (raw or data:image/png;base64,...); dataset reads it as image_key "images".
filtered_datas = []
for d in datas:
    raw = d.get('image') or ''
    image_str = raw.strip() if isinstance(raw, str) else ''
    # Validate image: must be non-empty and look like valid base64 (at least 100 chars for a minimal image)
    is_valid_image = (
        len(image_str) > 100 and
        (image_str.startswith('data:image') or len(image_str) > 200)
    )

    if (d['score'] >= args.min_score and
        d['score'] <= args.max_score and
        d.get('answer', '') not in ['', 'None'] and
        is_valid_image):
        filtered_datas.append({
            'problem': d['question'],          # hard_question
            'answer': d['answer'],             # majority-voted silver label
            'score': d['score'],
            'images': image_str,               # single base64 string per row (dataset expects this)
            'problem_type': d.get('question_type', 'numerical'),
            'caption': d.get('caption', ''),
            'easy_question': d.get('easy_question', ''),
            'easy_answer': d.get('easy_answer', ''),
        })

print(f"\nFiltered {len(filtered_datas)} samples with score in [{args.min_score}, {args.max_score}]")

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
