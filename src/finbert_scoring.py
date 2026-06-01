"""Batch FinBERT sentiment scoring and weekly aggregation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import config
from .text_cleaning import truncate_texts_for_transformer


def load_finbert_pipeline(model_name: str = config.FINBERT_MODEL_NAME, device: int = -1):
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


def _scores_to_row(scores: list[dict] | dict) -> dict[str, float | str]:
    if isinstance(scores, dict):
        scores = [scores]
    normalized = {item["label"].lower(): float(item["score"]) for item in scores}
    pos = normalized.get("positive", normalized.get("pos", 0.0))
    neg = normalized.get("negative", normalized.get("neg", 0.0))
    neu = normalized.get("neutral", normalized.get("neu", 0.0))
    label = max({"positive": pos, "negative": neg, "neutral": neu}, key={"positive": pos, "negative": neg, "neutral": neu}.get)
    return {
        "finbert_positive_prob": pos,
        "finbert_negative_prob": neg,
        "finbert_neutral_prob": neu,
        "finbert_label": label,
        "finbert_sentiment_score": pos - neg,
    }


def score_finbert_batch(pipe, texts: Iterable[str]) -> pd.DataFrame:
    outputs = pipe(list(texts))
    return pd.DataFrame([_scores_to_row(scores) for scores in outputs])


def score_finbert_articles(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    batch_size: int = config.TRANSFORMER_BATCH_SIZE,
    device: int = -1,
    model_name: str = config.FINBERT_MODEL_NAME,
) -> pd.DataFrame:
    pipe = load_finbert_pipeline(model_name=model_name, device=device)
    rows = []
    texts = truncate_texts_for_transformer(df[text_col].fillna(""))
    for start in tqdm(range(0, len(texts), batch_size), desc="FinBERT batches"):
        batch = texts[start : start + batch_size]
        rows.append(score_finbert_batch(pipe, batch))
    scored = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return pd.concat([df.reset_index(drop=True), scored], axis=1)


def add_finbert_sentiment_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["finbert_sentiment_score"] = out["finbert_positive_prob"] - out["finbert_negative_prob"]
    out["is_finbert_negative"] = out["finbert_negative_prob"] >= np.maximum(
        out["finbert_positive_prob"], out["finbert_neutral_prob"]
    )
    return out


def aggregate_finbert_weekly(df: pd.DataFrame, lags: tuple[int, ...] = (1, 4, 12)) -> pd.DataFrame:
    out = add_finbert_sentiment_score(df)
    dates = pd.to_datetime(out["date"])
    out["week_end_date"] = dates + pd.to_timedelta(4 - dates.dt.weekday, unit="D")
    weekly = (
        out.groupby(["ticker", "week_end_date"])
        .agg(
            finbert_article_count=("clean_text", "size"),
            finbert_avg_sentiment=("finbert_sentiment_score", "mean"),
            finbert_min_sentiment=("finbert_sentiment_score", "min"),
            finbert_max_sentiment=("finbert_sentiment_score", "max"),
            finbert_sentiment_volatility=("finbert_sentiment_score", "std"),
            finbert_negative_share=("is_finbert_negative", "mean"),
            finbert_positive_share=("finbert_label", lambda s: (s == "positive").mean()),
            finbert_negative_count=("is_finbert_negative", "sum"),
        )
        .reset_index()
    )
    weekly["finbert_sentiment_volatility"] = weekly["finbert_sentiment_volatility"].fillna(0.0)
    lag_cols = [col for col in weekly.columns if col.startswith("finbert_") and col != "finbert_article_count"]
    weekly = weekly.sort_values(["ticker", "week_end_date"])
    for lag in lags:
        for col in lag_cols:
            weekly[f"{col}_lag{lag}w"] = weekly.groupby("ticker")[col].shift(lag)
    return weekly

