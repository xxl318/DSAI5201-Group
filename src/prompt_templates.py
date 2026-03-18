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
    "fact_constrained": (
        "You are a factual summarization system. Summarize the following article in exactly 2 sentences. "
        "Sentence 1 should state the main event and actors. Sentence 2 should state the consequence or context. "
        "Keep all names, dates, and numbers accurate. No speculation. No extra details.\n\n"
        "Article:\n{article}\n"
    ),
}


def build_prompt(prompt_name: str, article: str, max_article_chars: int = 3200) -> str:
    if prompt_name not in PROMPT_TEMPLATES:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(f"Unknown prompt '{prompt_name}'. Available prompts: {available}")

    trimmed_article = article[:max_article_chars].strip()
    return PROMPT_TEMPLATES[prompt_name].format(article=trimmed_article)