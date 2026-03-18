"""Evaluate generated summaries with ROUGE and BERTScore.

Usage example:
python src/evaluate_results.py \
  --input_csv results/generated_summaries.csv \
  --metrics_csv results/automatic_metrics.csv \
  --grouped_csv results/metrics_by_prompt_and_genre.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--metrics_csv", type=str, required=True)
    parser.add_argument("--grouped_csv", type=str, required=True)
    parser.add_argument("--lang", type=str, default="en")
    return parser.parse_args()


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["generated_length_words"] = pd.to_numeric(
        df["generated_length_words"], errors="coerce"
    ).fillna(0)

    df["reference_length_words"] = pd.to_numeric(
        df["reference_length_words"], errors="coerce"
    ).fillna(0)

    df["compression_ratio"] = (
        df["generated_length_words"]
        / df["reference_length_words"].replace(0, np.nan)
    )

    summary_text = df["generated_summary"].fillna("").astype(str)

    df["is_bullet_format"] = summary_text.str.contains(
        r"(?m)^\s*(?:-|\u2022)",
        regex=True,
        na=False,
    ).astype(int)

    return df


def compute_row_metrics(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    predictions = (
        df["generated_summary"]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )
    references = (
        df["reference_summary"]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    # 避免 BERTScore 在空字符串上报错
    predictions = [p if p else "[EMPTY]" for p in predictions]
    references = [r if r else "[EMPTY]" for r in references]

    rouge_row_scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_aggregator=False,
    )

    try:
        bert_row_scores = bertscore.compute(
            predictions=predictions,
            references=references,
            lang=lang,
            use_fast_tokenizer=True,
        )
        bert_p = bert_row_scores["precision"]
        bert_r = bert_row_scores["recall"]
        bert_f1 = bert_row_scores["f1"]
    except Exception as e:
        print(f"Warning: BERTScore failed, filling with NaN. Error: {e}")
        bert_p = [np.nan] * len(df)
        bert_r = [np.nan] * len(df)
        bert_f1 = [np.nan] * len(df)

    out = df.copy()
    out["rouge1"] = rouge_row_scores["rouge1"]
    out["rouge2"] = rouge_row_scores["rouge2"]
    out["rougeL"] = rouge_row_scores["rougeL"]
    out["bertscore_precision"] = bert_p
    out["bertscore_recall"] = bert_r
    out["bertscore_f1"] = bert_f1
    return out


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    metrics_path = Path(args.metrics_csv)
    grouped_path = Path(args.grouped_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, keep_default_na=False)
    df = add_basic_features(df)
    df = compute_row_metrics(df, lang=args.lang)

    grouped = (
        df.groupby(["prompt_name", "genre"], as_index=False)
        .agg(
            rouge1=("rouge1", "mean"),
            rouge2=("rouge2", "mean"),
            rougeL=("rougeL", "mean"),
            bertscore_f1=("bertscore_f1", "mean"),
            generated_length_words=("generated_length_words", "mean"),
            compression_ratio=("compression_ratio", "mean"),
            bullet_format_rate=("is_bullet_format", "mean"),
            n=("id", "count"),
        )
        .sort_values(["genre", "rougeL"], ascending=[True, False])
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_path, index=False)
    grouped.to_csv(grouped_path, index=False)

    print(f"Saved row-level metrics to: {metrics_path}")
    print(f"Saved grouped metrics to: {grouped_path}")

    print("\nTop prompt per genre by ROUGE-L:")
    best = (
        grouped.sort_values(["genre", "rougeL"], ascending=[True, False])
        .groupby("genre")
        .head(1)
    )
    print(best[["genre", "prompt_name", "rougeL", "bertscore_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()