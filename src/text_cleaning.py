"""Text cleaning and composition helpers for FNSPID news."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(value: str | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return WHITESPACE_RE.sub(" ", str(value)).strip()


def clean_text_value(value: str | None) -> str:
    text = normalize_whitespace(value)
    return text.replace("\x00", "")


def combine_title_and_summaries(
    row: pd.Series,
    text_columns: Iterable[str] = ("Article_title", "Textrank_summary", "Lexrank_summary"),
    fallback_column: str = "Article_title",
) -> str:
    parts = [clean_text_value(row.get(col)) for col in text_columns]
    combined = clean_text_value(" ".join(part for part in parts if part))
    if combined:
        return combined
    return clean_text_value(row.get(fallback_column))


def add_clean_text(
    df: pd.DataFrame,
    text_columns: Iterable[str] = ("Article_title", "Textrank_summary", "Lexrank_summary"),
) -> pd.DataFrame:
    out = df.copy()
    out["clean_text"] = out.apply(
        lambda row: combine_title_and_summaries(row, text_columns=text_columns),
        axis=1,
    )
    return out


def truncate_texts_for_transformer(texts: Iterable[str], max_chars: int = 2_000) -> list[str]:
    return [clean_text_value(text)[:max_chars] for text in texts]


def deduplicate_articles(df: pd.DataFrame) -> pd.DataFrame:
    keys = [col for col in ["ticker", "date", "Url", "Article_title"] if col in df.columns]
    if not keys:
        return df.drop_duplicates().reset_index(drop=True)
    return df.drop_duplicates(subset=keys).reset_index(drop=True)

