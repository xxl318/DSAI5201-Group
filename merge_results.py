import pandas as pd

base = pd.read_csv("results/automatic_metrics_flan_t5_base.csv", keep_default_na=False)
strict = pd.read_csv("results/automatic_metrics_bullet_points_strict_flan_t5_base.csv", keep_default_na=False)

merged = pd.concat([base, strict], ignore_index=True)
merged.to_csv("results/automatic_metrics_flan_t5_base_plus_strict.csv", index=False)

grouped = (
    merged.groupby(["prompt_name", "genre"], as_index=False)
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
)

grouped.to_csv("results/metrics_by_prompt_and_genre_flan_t5_base_plus_strict.csv", index=False)
print("Saved merged files.")