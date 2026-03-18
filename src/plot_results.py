from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--row_metrics_csv",
        type=str,
        default="results/automatic_metrics.csv",
        help="Path to row-level metrics CSV.",
    )
    parser.add_argument(
        "--grouped_csv",
        type=str,
        default="results/metrics_by_prompt_and_genre.csv",
        help="Path to grouped metrics CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/figures",
        help="Directory to save figures and summary tables.",
    )
    return parser.parse_args()


def ensure_required_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def sort_prompt_order(df: pd.DataFrame, prompt_col: str = "prompt_name") -> pd.DataFrame:
    preferred_order = ["tldr", "bullet_points", "one_sentence", "fact_constrained"]
    existing = [p for p in preferred_order if p in df[prompt_col].unique()]
    remaining = sorted([p for p in df[prompt_col].unique() if p not in existing])
    order = existing + remaining

    df = df.copy()
    df[prompt_col] = pd.Categorical(df[prompt_col], categories=order, ordered=True)
    return df.sort_values(prompt_col)


def make_overall_prompt_table(row_df: pd.DataFrame) -> pd.DataFrame:
    row_df = row_df.copy()

    row_df["is_empty_generation"] = (
        row_df["generated_summary"].fillna("").astype(str).str.strip() == ""
    ).astype(int)

    overall = (
        row_df.groupby("prompt_name", as_index=False)
        .agg(
            rouge1=("rouge1", "mean"),
            rouge2=("rouge2", "mean"),
            rougeL=("rougeL", "mean"),
            bertscore_f1=("bertscore_f1", "mean"),
            avg_generated_length=("generated_length_words", "mean"),
            avg_compression_ratio=("compression_ratio", "mean"),
            bullet_format_rate=("is_bullet_format", "mean"),
            empty_generation_rate=("is_empty_generation", "mean"),
            n=("id", "count"),
        )
    )

    overall = sort_prompt_order(overall, "prompt_name")
    return overall


def save_bar_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    rotation: int = 0,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_grouped_bar_plot(
    grouped_df: pd.DataFrame,
    category_col: str,
    series_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    pivot_df = grouped_df.pivot(
        index=category_col,
        columns=series_col,
        values=value_col,
    )

    pivot_df = pivot_df.sort_index()
    prompt_order = ["tldr", "bullet_points", "one_sentence", "fact_constrained"]
    ordered_cols = [c for c in prompt_order if c in pivot_df.columns] + [
        c for c in pivot_df.columns if c not in prompt_order
    ]
    pivot_df = pivot_df[ordered_cols]

    categories = pivot_df.index.tolist()
    prompts = pivot_df.columns.tolist()

    x = np.arange(len(categories))
    width = 0.8 / max(len(prompts), 1)

    plt.figure(figsize=(10, 6))

    for i, prompt in enumerate(prompts):
        plt.bar(
            x + i * width - (len(prompts) - 1) * width / 2,
            pivot_df[prompt].values,
            width=width,
            label=str(prompt),
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x, categories, rotation=0)
    plt.legend(title="Prompt")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()

    row_metrics_path = Path(args.row_metrics_csv)
    grouped_path = Path(args.grouped_csv)
    output_dir = Path(args.output_dir)

    if not row_metrics_path.exists():
        raise FileNotFoundError(f"Row-level metrics file not found: {row_metrics_path}")
    if not grouped_path.exists():
        raise FileNotFoundError(f"Grouped metrics file not found: {grouped_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    row_df = pd.read_csv(row_metrics_path, keep_default_na=False)
    grouped_df = pd.read_csv(grouped_path, keep_default_na=False)

    ensure_required_columns(
        row_df,
        [
            "id",
            "prompt_name",
            "generated_summary",
            "generated_length_words",
            "compression_ratio",
            "is_bullet_format",
            "rouge1",
            "rouge2",
            "rougeL",
            "bertscore_f1",
        ],
        "automatic_metrics.csv",
    )

    ensure_required_columns(
        grouped_df,
        [
            "prompt_name",
            "genre",
            "rouge1",
            "rouge2",
            "rougeL",
            "bertscore_f1",
            "generated_length_words",
            "compression_ratio",
            "bullet_format_rate",
            "n",
        ],
        "metrics_by_prompt_and_genre.csv",
    )

    row_df = sort_prompt_order(row_df, "prompt_name")
    grouped_df = sort_prompt_order(grouped_df, "prompt_name")

    overall_df = make_overall_prompt_table(row_df)
    overall_csv_path = output_dir / "overall_prompt_metrics.csv"
    overall_df.to_csv(overall_csv_path, index=False)

    # 1) Overall ROUGE-L by prompt
    save_bar_plot(
        df=overall_df,
        x_col="prompt_name",
        y_col="rougeL",
        title="Overall ROUGE-L by Prompt",
        xlabel="Prompt",
        ylabel="ROUGE-L",
        output_path=output_dir / "overall_prompt_rougel.png",
        rotation=15,
    )

    # 2) ROUGE-L by genre and prompt
    save_grouped_bar_plot(
        grouped_df=grouped_df,
        category_col="genre",
        series_col="prompt_name",
        value_col="rougeL",
        title="ROUGE-L by Genre and Prompt",
        xlabel="Genre",
        ylabel="ROUGE-L",
        output_path=output_dir / "genre_prompt_rougel.png",
    )

    # 3) Average generated length by prompt
    save_bar_plot(
        df=overall_df,
        x_col="prompt_name",
        y_col="avg_generated_length",
        title="Average Generated Summary Length by Prompt",
        xlabel="Prompt",
        ylabel="Average Length (words)",
        output_path=output_dir / "avg_length_by_prompt.png",
        rotation=15,
    )

    # 4) BERTScore F1 by prompt
    save_bar_plot(
        df=overall_df,
        x_col="prompt_name",
        y_col="bertscore_f1",
        title="Average BERTScore F1 by Prompt",
        xlabel="Prompt",
        ylabel="BERTScore F1",
        output_path=output_dir / "bertscore_by_prompt.png",
        rotation=15,
    )

    # 5) Bullet format rate by prompt
    save_bar_plot(
        df=overall_df,
        x_col="prompt_name",
        y_col="bullet_format_rate",
        title="Bullet Format Rate by Prompt",
        xlabel="Prompt",
        ylabel="Bullet Format Rate",
        output_path=output_dir / "bullet_rate_by_prompt.png",
        rotation=15,
    )

    print(f"Saved overall prompt summary table to: {overall_csv_path}")
    print(f"Saved figures to: {output_dir}")

    print("\nOverall prompt ranking by ROUGE-L:")
    ranking = overall_df.sort_values("rougeL", ascending=False)[
        ["prompt_name", "rougeL", "bertscore_f1", "avg_generated_length", "empty_generation_rate"]
    ]
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()