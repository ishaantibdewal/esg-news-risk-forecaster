"""ClimateBERT relevance scoring and weekly feature aggregation."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from tqdm.auto import tqdm

from . import config
from .text_cleaning import truncate_texts_for_transformer


def load_climatebert_pipeline(model_name: str = config.CLIMATEBERT_MODEL_NAME, device: int = -1):
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        top_k=None,
        truncation=True,
        max_length=config.TRANSFORMER_MAX_LENGTH,
        device=device,
    )


def _normalize_label(label: str) -> str:
    return label.lower().replace(" ", "_").replace("-", "_")


def _scores_to_row(scores: list[dict] | dict) -> dict[str, float | str]:
    if isinstance(scores, dict):
        scores = [scores]
    normalized = {_normalize_label(item["label"]): float(item["score"]) for item in scores}
    climate_like = [
        score
        for label, score in normalized.items()
        if "climate" in label or "yes" == label or "relevant" in label or "risk" in label
    ]
    non_climate_like = [
        score
        for label, score in normalized.items()
        if "non" in label or "no" == label or "irrelevant" in label
    ]
    relevance = max(climate_like) if climate_like else max(normalized.values(), default=0.0)
    non_climate = max(non_climate_like) if non_climate_like else max(0.0, 1.0 - relevance)
    label = max(normalized, key=normalized.get) if normalized else "unknown"
    row: dict[str, float | str] = {
        "climate_relevance_score": relevance,
        "non_climate_prob": non_climate,
        "climate_label": label,
    }
    for label_name, score in normalized.items():
        row[f"climatebert_{label_name}_prob"] = score
    return row


def score_climatebert_batch(pipe, texts: Iterable[str]) -> pd.DataFrame:
    outputs = pipe(list(texts))
    return pd.DataFrame([_scores_to_row(scores) for scores in outputs])


def score_climatebert_articles(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    batch_size: int = config.TRANSFORMER_BATCH_SIZE,
    device: int = -1,
    model_name: str = config.CLIMATEBERT_MODEL_NAME,
) -> pd.DataFrame:
    pipe = load_climatebert_pipeline(model_name=model_name, device=device)
    rows = []
    texts = truncate_texts_for_transformer(df[text_col].fillna(""))
    for start in tqdm(range(0, len(texts), batch_size), desc="ClimateBERT batches"):
        batch = texts[start : start + batch_size]
        rows.append(score_climatebert_batch(pipe, batch))
    scored = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return pd.concat([df.reset_index(drop=True), scored], axis=1)


def add_climate_finbert_interactions(df: pd.DataFrame, relevance_threshold: float = 0.5) -> pd.DataFrame:
    out = df.copy()
    out["is_climate_article"] = out["climate_relevance_score"] >= relevance_threshold
    if "finbert_negative_prob" in out.columns:
        out["climate_x_finbert_negative_prob"] = (
            out["climate_relevance_score"] * out["finbert_negative_prob"]
        )
    if "is_finbert_negative" in out.columns:
        out["is_climate_and_finbert_negative"] = out["is_climate_article"] & out["is_finbert_negative"]
    return out


def aggregate_climatebert_weekly(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 4, 12),
    relevance_threshold: float = 0.5,
) -> pd.DataFrame:
    out = add_climate_finbert_interactions(df, relevance_threshold=relevance_threshold)
    dates = pd.to_datetime(out["date"])
    out["week_end_date"] = dates + pd.to_timedelta(4 - dates.dt.weekday, unit="D")
    agg_spec = {
        "climate_article_total": ("clean_text", "size"),
        "climate_article_count": ("is_climate_article", "sum"),
        "avg_climate_relevance": ("climate_relevance_score", "mean"),
        "max_climate_relevance": ("climate_relevance_score", "max"),
    }
    if "is_climate_and_finbert_negative" in out.columns:
        agg_spec["climate_negative_count"] = ("is_climate_and_finbert_negative", "sum")
    if "climate_x_finbert_negative_prob" in out.columns:
        agg_spec["avg_climate_x_finbert_negative_prob"] = ("climate_x_finbert_negative_prob", "mean")
    weekly = out.groupby(["ticker", "week_end_date"]).agg(**agg_spec).reset_index()
    denom = weekly["climate_article_total"].clip(lower=1)
    weekly["climate_article_share"] = weekly["climate_article_count"] / denom
    weekly["climate_news_intensity"] = weekly["avg_climate_relevance"] * weekly["climate_article_total"]
    if "climate_negative_count" in weekly.columns:
        weekly["climate_negative_share"] = weekly["climate_negative_count"] / denom
    weekly = weekly.sort_values(["ticker", "week_end_date"])
    lag_cols = [
        col for col in weekly.columns if col not in {"ticker", "week_end_date"} and not col.endswith("_total")
    ]
    for lag in lags:
        for col in lag_cols:
            weekly[f"{col}_lag{lag}w"] = weekly.groupby("ticker")[col].shift(lag)
    return weekly

