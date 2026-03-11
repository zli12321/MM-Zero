#!/usr/bin/env python
"""
Generate a seed prompts parquet file for Proposer GRPO training.

Each row contains a short, general topic string in the 'problem' column.
The Proposer model uses these as starting points to imagine charts/visualizations
and generate structured proposals (caption, questions, answers).

Topics are intentionally broad — no chart types, no specific years, no data values.
The model is free to choose visualization type, invent data, and create diverse Q&A.

Usage:
    python MM-zero_final/data/generate_seed_prompts.py
    # Output: MM-zero_final/data/text_seed_prompts.parquet
"""

import pandas as pd
import os

# SEED_TOPICS = [
#     "mathematics",
#     "science",
#     "economics",
#     "finance",
#     "healthcare",
#     "technology",
#     "environment",
#     "education",
#     "geography",
#     "sports",
#     "arts and culture",
#     "society",
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

TARGET_SIZE = 1008  # ~1000 rows, evenly divisible by len(SEED_TOPICS)


def main():
    repeats = TARGET_SIZE // len(SEED_TOPICS)
    topics = SEED_TOPICS * repeats
    df = pd.DataFrame({
        "problem": topics,
        "answer": [""] * len(topics),
    })

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "text_seed_prompts.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} seed topics to {out_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDomain coverage:")
    print(f"  Total topics: {len(SEED_TOPICS)}")
    print(f"\nSample topics:")
    for topic in SEED_TOPICS:
        print(f"  - {topic}")


if __name__ == "__main__":
    main()
