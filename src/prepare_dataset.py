"""Prepare a balanced XSum subset with inferred BBC-style genres.

Usage example:
python src/prepare_dataset.py \
  --output_csv results/sampled_xsum_with_genre.csv \
  --samples_per_genre 20 \
  --candidate_labels business entertainment politics sport tech
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import List

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline


DEFAULT_LABELS = ["business", "entertainment", "politics", "sport", "tech"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="EdinburghNLP/xsum")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--candidate_pool_size", type=int, default=300)
    parser.add_argument("--samples_per_genre", type=int, default=20)
    parser.add_argument("--classifier_model", type=str, default="facebook/bart-large-mnli")
    parser.add_argument("--candidate_labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--article_chars_for_classification", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hypothesis_template", type=str, default="This text is about {}.")
    parser.add_argument("--output_csv", type=str, required=True)
    return parser.parse_args()


def get_device_index() -> int:
    return 0 if torch.cuda.is_available() else -1


def main() -> None:
    args = parse_args()

    print(f"Loading dataset: {args.dataset_name} [{args.split}]")
    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = dataset.select(range(min(args.candidate_pool_size, len(dataset))))

    print(f"Loaded candidate pool with {len(dataset)} articles")
    print(f"Using classifier model: {args.classifier_model}")

    classifier = pipeline(
        task="zero-shot-classification",
        model=args.classifier_model,
        device=get_device_index(),
    )

    rows = []
    genre_counts = defaultdict(int)
    target_total = len(args.candidate_labels) * args.samples_per_genre

    for start in tqdm(range(0, len(dataset), args.batch_size), desc="Classifying genres"):
        batch = dataset[start : start + args.batch_size]
        texts = [doc[: args.article_chars_for_classification] for doc in batch["document"]]

        outputs = classifier(
            texts,
            candidate_labels=args.candidate_labels,
            hypothesis_template=args.hypothesis_template,
            multi_label=False,
        )

        if isinstance(outputs, dict):
            outputs = [outputs]

        for idx, pred in enumerate(outputs):
            top_label = pred["labels"][0]
            top_score = float(pred["scores"][0])

            if genre_counts[top_label] >= args.samples_per_genre:
                continue

            rows.append(
                {
                    "id": batch["id"][idx],
                    "genre": top_label,
                    "genre_score": top_score,
                    "document": batch["document"][idx],
                    "reference_summary": batch["summary"][idx],
                }
            )
            genre_counts[top_label] += 1

            if len(rows) >= target_total:
                break
        if len(rows) >= target_total:
            break

    df = pd.DataFrame(rows)

    print("Final genre counts:")
    if not df.empty:
        print(df["genre"].value_counts().sort_index())
    else:
        print("No rows collected. Increase --candidate_pool_size or reduce --samples_per_genre.")

    if len(df) < target_total:
        print(
            f"Warning: target size was {target_total}, but only collected {len(df)} rows. "
            "Increase candidate_pool_size if you want a perfectly balanced set."
        )

    df.to_csv(args.output_csv, index=False)
    print(f"Saved prepared dataset to: {args.output_csv}")


if __name__ == "__main__":
    main()
