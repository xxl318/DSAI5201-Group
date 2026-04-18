# Comparative Analysis of Zero-shot Summarization Strategies on News Articles

## Project Overview

This project studies how different prompt strategies affect news summarization quality and formatting under a fixed instruction-tuned language model (FLAN-T5-base). The main goal is to compare several zero-shot prompt styles, analyze their performance across inferred news genres, and explore the architectural limitations of small models through few-shot ablation studies.

The project uses:
- A balanced subset of the XSum dataset (500 articles across 5 inferred genres)
- Zero-shot classification (`facebook/bart-large-mnli`) to assign genre labels
- `google/flan-t5-base` (250M parameters) for local inference
- ROUGE, BERTScore, and a custom Regular Expression metric for format compliance

The project consists of three main components:
1. **Main Zero-shot Experiment:** Comparing 4 baseline prompts and 1 strict-formatting prompt.
2. **Ablation Study (Few-shot):** Testing 1-shot and 3-shot learning to diagnose the 0% format compliance issue.
3. **Supplementary Experiment (Role-playing):** Evaluating whether genre-aware personas (e.g., "Act as a political editor") improve domain-specific summarization.

---

## Research Questions

1. Do different zero-shot prompt strategies produce varying summarization quality and semantic coherence?
2. Does the optimal prompt strategy dynamically change across different news genres?
3. Can strict formatting instructions or in-context examples (few-shot) force small models to output complex spatial layouts (e.g., bullet points)?
4. Do complex role-playing personas help or hinder small-parameter models in domain-specific tasks?

---

## Prompt Strategies Evaluated

### Main Zero-shot Prompts
- `tldr`: Asks for a concise 2-3 sentence TL;DR.
- `bullet_points`: Asks for exactly 3 bullet points.
- `one_sentence`: Asks for a one-sentence news lead.
- `fact_constrained`: Asks for exactly 2 sentences with factual constraints.
- `bullet_points_strict`: Explicitly requires each line to start with "-" and forbids paragraph-style output.

### Ablation Study Prompts (Few-shot)
- `bullet_points_1_shot`: Provides 1 perfect structural demonstration.
- `bullet_points_3_shot`: Provides 3 perfect structural demonstrations.

### Supplementary Prompts (Role-playing)
- `fact_constrained_politics`: Instructs the model to act as a political news editor.
- `sport_lead`: Instructs the model to act as a sports journalist.

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
│   ├── generated_summaries_all.csv
│   ├── automatic_metrics.csv
│   ├── metrics_by_prompt_and_genre.csv
│   └── figures/
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

## Pipeline and Reproducibility

For a full reproduction of the final results, execute the scripts in the following order:

### Step 1: Prepare Dataset and Infer Genres
Samples 500 articles and assigns inferred genre labels.
```bash
python src/prepare_dataset.py \
  --output_csv results/sampled_xsum_with_genre.csv \
  --samples_per_genre 100 \
  --candidate_pool_size 3000
```

### Step 2: Run Generation (Main, Ablation & Supplementary)
Generates summaries across all 8 prompt configurations in a single run.
```bash
python src/run_generation.py \
  --input_csv results/sampled_xsum_with_genre.csv \
  --output_csv results/generated_summaries_all.csv \
  --model_name google/flan-t5-base
```

### Step 3: Evaluate Results
Computes row-level and grouped metrics (ROUGE, BERTScore, Format Rate).
```bash
python src/evaluate_results.py \
  --input_csv results/generated_summaries_all.csv \
  --metrics_csv results/automatic_metrics.csv \
  --grouped_csv results/metrics_by_prompt_and_genre.csv
```

### Step 4: Plot Results
Generates all visualizations for the final report.
```bash
python src/plot_results.py \
  --row_metrics_csv results/automatic_metrics.csv \
  --grouped_csv results/metrics_by_prompt_and_genre.csv \
  --output_dir results/figures
```
---

## Current Main Findings

Based on the 500-sample evaluation using FLAN-T5-base:

### 1. Lexical vs. Semantic Divergence
- `bullet_points_strict` achieved the highest ROUGE-L (**0.2731**) because its strict wording acts as a keyword extractor.
- `fact_constrained` achieved the highest BERTScore F1 (**0.9034**) because it forces deeper semantic synthesis.

### 2. The Formatting Paradox (Ablation Study)
- The Format Success Rate for all bullet-point prompts remained at **0.00%**, regardless of zero-shot strict wording or few-shot demonstrations.
- Small legacy architectures (250M) exhibit a structural blind spot to spatial layout instructions. Providing 3-shot examples actually decreased performance due to context overload.

### 3. Role-Playing as Noise (Supplementary Experiment)
- Explicit genre-specific framing largely degraded performance. For instance, the generic `fact_constrained` outperformed the role-playing `fact_constrained_politics`.
- For lightweight models, simple instructions and direct task alignment are significantly more effective than complex personas.

---

## Evaluation Metrics

This project uses an automated evaluation pipeline tracking four dimensions:
- **ROUGE (1, 2, L):** Measures surface-level lexical overlap.
- **BERTScore F1:** Measures deeper semantic similarity and factual alignment.
- **Average Generation Length:** Measures verbosity constraint.
- **Format Success Rate:** A rule-based metric using Regex (`r"(?m)^\s*(?:-|\bullet)"`) to detect discrete formatted lines.

## Notes and Limitations

1. **Model Capacity:** The 0% formatting compliance is an architectural limitation of the 250M-parameter FLAN-T5-base model. This behavior may not generalize to massive Decoder-only models (e.g., Llama-3, GPT-4) where complex layout adherence is an emergent capability.
2. **Dataset Bias:** The evaluation relies on the XSum dataset, which inherently favors highly abstractive, single-sentence summaries. Prompt behaviors observed here might differ on longer, more extractive datasets like CNN/DailyMail.
3. **Label Noise:** Genre labels were inferred using a zero-shot classifier (`BART-large-MNLI`) to ensure a balanced dataset, meaning marginal label noise may exist compared to manual human annotation.
4. **Scope:** This project rigorously evaluates zero-shot and few-shot in-context learning. It does not involve parameter fine-tuning (LoRA/QLoRA).