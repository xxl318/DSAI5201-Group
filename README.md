# Comparative Analysis of Zero-shot Summarization Strategies on News Articles

## Project Overview

This project studies how different zero-shot prompt strategies affect news summarization quality under a fixed instruction-tuned language model. The main goal is to compare several prompt styles on the same dataset and analyze whether their performance changes across inferred news genres.

The project uses:
- a news summarization dataset with article-summary pairs
- zero-shot genre inference to assign genre labels
- FLAN-T5 models for zero-shot summarization
- ROUGE and BERTScore for evaluation

The project also includes a supplementary experiment with a stricter bullet-point prompt to test whether stronger formatting instructions can improve bullet-style compliance.

---

## Research Questions

This project focuses on the following questions:

1. Do different zero-shot prompt strategies produce different summarization quality?
2. Does the best prompt vary across news genres?
3. Can stricter formatting instructions improve bullet-point compliance?

---

## Main Prompt Strategies

The main experiment compares four prompt strategies:

- `tldr`
- `bullet_points`
- `one_sentence`
- `fact_constrained`

### Supplementary Prompt
- `bullet_points_strict`

This additional prompt was introduced after the main experiment to test whether stronger wording could force the model to output literal bullet lists.

---

## Repository Structure

```text
zero_shot_summarization_starter/
├── src/
│   ├── prepare_dataset.py
│   ├── run_generation.py
│   ├── evaluate_results.py
│   ├── plot_results.py
│   └── prompt_templates.py
├── results/
│   ├── sampled_xsum_with_genre.csv
│   ├── generated_summaries_flan_t5_base.csv
│   ├── automatic_metrics_flan_t5_base.csv
│   ├── metrics_by_prompt_and_genre_flan_t5_base.csv
│   ├── generated_bullet_points_strict_flan_t5_base.csv
│   ├── automatic_metrics_bullet_points_strict_flan_t5_base.csv
│   ├── automatic_metrics_flan_t5_base_plus_strict.csv
│   ├── metrics_by_prompt_and_genre_flan_t5_base_plus_strict.csv
│   └── figures_flan_t5_base_plus_strict/
├── merge_results.py
├── requirements.txt
└── README.md
```

---

## Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Step 1: Prepare Dataset and Infer Genres

This step samples articles and assigns inferred genre labels.

Example:

```bash
python src/prepare_dataset.py \
  --output_csv results/sampled_xsum_with_genre.csv \
  --samples_per_genre 10 \
  --candidate_labels business entertainment politics sport tech
```

Output:
- `results/sampled_xsum_with_genre.csv`

---

## Step 2: Run the Main Summarization Experiment

This step generates summaries for the main four prompt strategies.

Example with FLAN-T5-base:

```bash
python src/run_generation.py \
  --input_csv results/sampled_xsum_with_genre.csv \
  --output_csv results/generated_summaries_flan_t5_base.csv \
  --model_name google/flan-t5-base
```

Output:
- `results/generated_summaries_flan_t5_base.csv`

---

## Step 3: Evaluate the Main Experiment

This step computes row-level metrics and grouped metrics.

```bash
python src/evaluate_results.py \
  --input_csv results/generated_summaries_flan_t5_base.csv \
  --metrics_csv results/automatic_metrics_flan_t5_base.csv \
  --grouped_csv results/metrics_by_prompt_and_genre_flan_t5_base.csv
```

Outputs:
- `results/automatic_metrics_flan_t5_base.csv`
- `results/metrics_by_prompt_and_genre_flan_t5_base.csv`

---

## Step 4: Plot Main Experiment Results

```bash
python src/plot_results.py \
  --row_metrics_csv results/automatic_metrics_flan_t5_base.csv \
  --grouped_csv results/metrics_by_prompt_and_genre_flan_t5_base.csv \
  --output_dir results/figures_flan_t5_base
```

Outputs include:
- overall ROUGE-L by prompt
- ROUGE-L by genre and prompt
- average summary length by prompt
- average BERTScore F1 by prompt
- bullet format rate by prompt

---

## Supplementary Experiment: Strict Bullet Prompt

This experiment tests whether stronger formatting instructions improve bullet-style compliance.

### Generate Summaries for `bullet_points_strict`

```bash
python src/run_generation.py \
  --input_csv results/sampled_xsum_with_genre.csv \
  --output_csv results/generated_bullet_points_strict_flan_t5_base.csv \
  --model_name google/flan-t5-base \
  --prompt_names bullet_points_strict
```

### Evaluate Strict Bullet Prompt

```bash
python src/evaluate_results.py \
  --input_csv results/generated_bullet_points_strict_flan_t5_base.csv \
  --metrics_csv results/automatic_metrics_bullet_points_strict_flan_t5_base.csv \
  --grouped_csv results/metrics_bullet_points_strict_flan_t5_base.csv
```

### Merge Main and Supplementary Results

```bash
python merge_results.py
```

This creates:
- `results/automatic_metrics_flan_t5_base_plus_strict.csv`
- `results/metrics_by_prompt_and_genre_flan_t5_base_plus_strict.csv`

### Plot Merged Results

```bash
python src/plot_results.py \
  --row_metrics_csv results/automatic_metrics_flan_t5_base_plus_strict.csv \
  --grouped_csv results/metrics_by_prompt_and_genre_flan_t5_base_plus_strict.csv \
  --output_dir results/figures_flan_t5_base_plus_strict
```

---

## Current Main Findings

Based on the FLAN-T5-base experiment:

### Overall ROUGE-L Ranking
1. `one_sentence` — 0.2688  
2. `bullet_points` — 0.2659  
3. `bullet_points_strict` — 0.2642  
4. `fact_constrained` — 0.2632  
5. `tldr` — 0.2569  

### Observations
- `one_sentence` is the best overall prompt.
- The best prompt is not identical across all genres.
- `bullet_points_strict` does not outperform the original `bullet_points`.
- Both `bullet_points` and `bullet_points_strict` fail to produce literal bullet lists.
- Strengthening the wording of the bullet prompt does not solve format-compliance issues in this zero-shot setting.
- FLAN-T5-base is more stable than FLAN-T5-small, with no empty generations in the main run.

---

## Evaluation Metrics

This project uses the following metrics:

- **ROUGE-1**
- **ROUGE-2**
- **ROUGE-L**
- **BERTScore F1**
- **Average generated summary length**
- **Bullet format rate**
- **Empty generation rate**

### Bullet Format Rate
A summary is counted as bullet-formatted only if it includes literal bullet-style lines, such as:
- `- item`
- `• item`

In the current experiments, the bullet format rate remains 0 for all prompts, including `bullet_points` and `bullet_points_strict`.

---

## Notes and Limitations

1. Genre labels are inferred automatically rather than manually annotated, so some label noise may exist.
2. The dataset favors short single-sentence summaries, which may naturally benefit the `one_sentence` prompt.
3. Stronger prompt wording does not necessarily guarantee output-format compliance.
4. This project evaluates zero-shot summarization only and does not include fine-tuning.

---

## Suggested Report Structure

This repository supports the following report structure:

- Introduction
- Methodology
- Main Experiment
- Supplementary Experiment
- Results
- Discussion
- Limitations
- Conclusion

---

## Reproducibility

For a full reproduction of the final results used in this project, run the following steps in order:

1. `prepare_dataset.py`
2. `run_generation.py` for main prompts
3. `evaluate_results.py` for main prompts
4. `run_generation.py` for `bullet_points_strict`
5. `evaluate_results.py` for `bullet_points_strict`
6. `merge_results.py`
7. `plot_results.py`

---

## Author Notes

This repository contains both:
- the **main prompt comparison experiment**
- the **supplementary strict bullet prompt experiment**

The supplementary experiment is included because it helps test whether stronger formatting instructions improve bullet-style compliance. In the current setup, the answer is no.
