# Zero-shot Summarization Starter Project

## Project title
Comparative Analysis of Zero-shot Summarization Strategies Across Inferred BBC News Genres

## Narrowed topic
This starter project uses the XSum news summarization dataset as the source of articles and reference summaries.
Because XSum provides article-summary pairs but not explicit genre labels, the pipeline first assigns each article a genre with a zero-shot classifier over five BBC-style labels:
- business
- entertainment
- politics
- sport
- tech

Then it compares four zero-shot summarization prompt strategies on the same base instruction model.

## Default models
- Genre inference: `facebook/bart-large-mnli`
- Summarization backbone: `google/flan-t5-small`

You can switch to `google/flan-t5-base` if you have a stronger GPU and want better generation quality.

## Prompt strategies
1. `tldr`
2. `bullet_points`
3. `one_sentence`
4. `fact_constrained`

## Folder structure
```text
zero_shot_summarization_starter/
├── requirements.txt
├── README.md
├── src/
│   ├── prompt_templates.py
│   ├── prepare_dataset.py
│   ├── run_generation.py
│   └── evaluate_results.py
└── results/
    └── figures/
```

## Environment setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1: sample and label the dataset
This command downloads XSum, samples a balanced subset, and assigns a genre using zero-shot classification.

```bash
python src/prepare_dataset.py \
  --output_csv results/sampled_xsum_with_genre.csv \
  --samples_per_genre 20 \
  --candidate_labels business entertainment politics sport tech
```

Notes:
- This script first pulls a larger candidate pool from XSum.
- It labels each article with the highest-probability genre.
- It keeps a balanced sample across the five genres.

## Step 2: run summarization generation
```bash
python src/run_generation.py \
  --input_csv results/sampled_xsum_with_genre.csv \
  --output_csv results/generated_summaries.csv \
  --model_name google/flan-t5-small
```

If you are on Colab T4 and memory is enough, try:
```bash
python src/run_generation.py \
  --input_csv results/sampled_xsum_with_genre.csv \
  --output_csv results/generated_summaries.csv \
  --model_name google/flan-t5-base
```

## Step 3: evaluate the outputs
```bash
python src/evaluate_results.py \
  --input_csv results/generated_summaries.csv \
  --metrics_csv results/automatic_metrics.csv \
  --grouped_csv results/metrics_by_prompt_and_genre.csv
```

## Main output files
- `results/sampled_xsum_with_genre.csv`
- `results/generated_summaries.csv`
- `results/automatic_metrics.csv`
- `results/metrics_by_prompt_and_genre.csv`

## Suggested report contribution points
This starter pipeline already gives you material for these three contributions:
1. Prompt comparison across genres.
2. Length-control analysis across prompt styles.
3. Error analysis on factuality, omission, and over-generalization.

## Practical advice
- Start with `samples_per_genre=10` to make sure the full pipeline works.
- When the pipeline is stable, increase to 20 or 30.
- Save every intermediate CSV.
- For the report, add 5-10 manual case studies with side-by-side summaries.
