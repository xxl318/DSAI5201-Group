"""Prompt templates for zero-shot news summarization."""

from typing import Dict


PROMPT_TEMPLATES: Dict[str, str] = {
    "tldr": (
        "You are a news editor. Read the following article and write a concise TL;DR in 2 to 3 sentences. "
        "Preserve important names, dates, locations, and numbers. Do not add unsupported details.\n\n"
        "Article:\n{article}\n"
    ),
    "bullet_points": (
        "You are a news editor. Summarize the following news article in exactly 3 bullet points. "
        "Each bullet should contain one key fact. Preserve names, dates, locations, and numbers. "
        "Do not add unsupported details.\n\n"
        "Article:\n{article}\n"
    ),
    "bullet_points_strict": (
        "You are a news editor.\n"
        "Summarize the following news article in exactly 3 bullet points.\n"
        "Each line must start with '- '.\n"
        "Output only the bullet list.\n"
        "Do not write a paragraph.\n"
        "Do not write a single sentence summary.\n"
        "Do not write any introduction or conclusion.\n"
        "Each bullet must contain one key fact.\n"
        "Preserve important names, dates, locations, and numbers.\n"
        "Do not add unsupported details.\n\n"
        "Article:\n{article}\n"
    ),
    "one_sentence": (
        "You are a newsroom assistant. Write a one-sentence news lead of at most 35 words that captures the main event, "
        "the main actors, and the outcome. Do not include opinions or unsupported details.\n\n"
        "Article:\n{article}\n"
    ),
    "sport_lead": (
        "You are a sports news editor. Write exactly one sentence as a compact sports news lead. "
        "Focus on the main event, the key person or team, and the outcome. "
        "Keep it factual and concise, with no extra background unless essential.\n\n"
        "Article:\n{article}\n"
    ),
    "fact_constrained": (
        "You are a factual summarization system. Summarize the following article in exactly 2 sentences. "
        "Sentence 1 should state the main event and actors. Sentence 2 should state the consequence or context. "
        "Keep all names, dates, and numbers accurate. No speculation. No extra details.\n\n"
        "Article:\n{article}\n"
    ),
    "fact_constrained_politics": (
        "You are a political news editor. Summarize the following article in exactly 2 sentences. "
        "Sentence 1 should state the main political event, decision, or announcement and identify the key actors. "
        "Sentence 2 should state the main consequence, response, or essential political context. "
        "Preserve names, dates, locations, and numbers accurately. Do not add unsupported details.\n\n"
        "Article:\n{article}\n"
    ),
    "bullet_points_1_shot": """Write a summary of the article in exactly 3 bullet points.
    
    Example:
    Article: The bank announced a major restructuring today, which will result in the loss of 2,000 jobs across its retail branch network. The move is designed to cut costs and shift focus towards digital banking.
    Summary: 
    - The bank is restructuring to cut costs and focus on digital banking.
    - 2,000 jobs will be lost as a result.
    - The cuts will primarily affect the retail branch network.
    
    Now, do the same for the following article:
    Article: {article}
    Summary:""",
    "bullet_points_3_shot": """Write a summary of the article in exactly 3 bullet points.

    Example 1:
    Article: The bank announced a major restructuring today, which will result in the loss of 2,000 jobs across its retail branch network. The move is designed to cut costs and shift focus towards digital banking.
    Summary: 
    - The bank is restructuring to cut costs and focus on digital banking.
    - 2,000 jobs will be lost as a result.
    - The cuts will primarily affect the retail branch network.

    Example 2:
    Article: The local football team won their final match of the season 3-1 against their rivals. The victory secured their promotion to the premier division for the first time in ten years.
    Summary:
    - The local team won their final match 3-1 against their rivals.
    - This victory secured their promotion to the premier division.
    - It is their first time in the top tier in ten years.

    Example 3:
    Article: A new study published in a medical journal suggests that moderate daily coffee consumption can reduce the risk of cardiovascular disease. Researchers analyzed data from over 100,000 participants over a decade.
    Summary:
    - A new study links moderate coffee consumption to better health.
    - It specifically reduces the risk of cardiovascular disease.
    - Researchers analyzed data from over 100,000 participants over ten years.

    Now, do the same for the following article:
    Article: {article}
    Summary:""",
}


def build_prompt(prompt_name: str, article: str, max_article_chars: int = 3200) -> str:
    if prompt_name not in PROMPT_TEMPLATES:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(f"Unknown prompt '{prompt_name}'. Available prompts: {available}")

    trimmed_article = article[:max_article_chars].strip()
    return PROMPT_TEMPLATES[prompt_name].format(article=trimmed_article)