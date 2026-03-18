"""Generate summaries for each prompt strategy.

Usage example:
python src/run_generation.py \
  --input_csv results/sampled_xsum_with_genre.csv \
  --output_csv results/generated_summaries.csv \
  --model_name google/flan-t5-small
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from prompt_templates import PROMPT_TEMPLATES, build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="google/flan-t5-small")
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--article_chars_for_prompt", type=int, default=3200)
    parser.add_argument("--limit_rows", type=int, default=0)
    parser.add_argument(
        "--prompt_names",
        type=str,
        default="",
        help="Comma-separated prompt names to run, e.g. bullet_points_strict or tldr,one_sentence",
    )
    return parser.parse_args()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_text(text: str) -> str:
    return " ".join(str(text).split())


def generate_one(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_input_length: int,
    max_new_tokens: int,
    num_beams: int,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            early_stopping=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def resolve_prompt_names(prompt_names_arg: str) -> List[str]:
    if not prompt_names_arg.strip():
        return list(PROMPT_TEMPLATES.keys())

    prompt_names = [p.strip() for p in prompt_names_arg.split(",") if p.strip()]
    unknown = [p for p in prompt_names if p not in PROMPT_TEMPLATES]
    if unknown:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(f"Unknown prompt(s): {unknown}. Available prompts: {available}")
    return prompt_names


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if args.limit_rows > 0:
        df = df.head(args.limit_rows).copy()

    device = get_device()
    print(f"Loading summarization model: {args.model_name}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    rows = []
    prompt_names: List[str] = resolve_prompt_names(args.prompt_names)
    print(f"Running prompts: {prompt_names}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating summaries"):
        article = str(row["document"])
        reference_summary = str(row["reference_summary"])
        genre = str(row["genre"])
        article_id = str(row["id"])

        for prompt_name in prompt_names:
            prompt = build_prompt(
                prompt_name=prompt_name,
                article=article,
                max_article_chars=args.article_chars_for_prompt,
            )
            prediction = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )

            rows.append(
                {
                    "id": article_id,
                    "genre": genre,
                    "prompt_name": prompt_name,
                    "reference_summary": clean_text(reference_summary),
                    "generated_summary": clean_text(prediction),
                    "source_length_chars": len(article),
                    "generated_length_words": len(clean_text(prediction).split()),
                    "reference_length_words": len(clean_text(reference_summary).split()),
                }
            )

    result_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Saved generations to: {output_path}")


if __name__ == "__main__":
    main()