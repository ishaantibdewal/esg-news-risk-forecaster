"""Chunked filtering and normalization for the large FNSPID news CSV."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from . import config
from .data_loading import iter_news_chunks, write_chunk_parquet, write_parquet_safe
from .text_cleaning import add_clean_text, deduplicate_articles

LOGGER = logging.getLogger(__name__)


def standardize_news_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ticker"] = out["Stock_symbol"].astype("string").str.upper().str.strip()
    out["datetime"] = pd.to_datetime(out["Date"], errors="coerce", utc=True)
    out["date"] = out["datetime"].dt.tz_convert(None).dt.normalize()
    return out


def filter_news_chunk(
    chunk: pd.DataFrame,
    tickers: Sequence[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    out = standardize_news_columns(chunk)
    wanted = {ticker.upper() for ticker in tickers}
    mask = out["ticker"].isin(wanted) & out["date"].notna()
    if start_date is not None:
        mask &= out["date"] >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= out["date"] <= pd.Timestamp(end_date)
    out = out.loc[mask].copy()
    if out.empty:
        return out
    out = add_clean_text(out, text_columns=config.NEWS_TEXT_COLUMNS)
    keep_cols = [
        "date",
        "datetime",
        "ticker",
        "Article_title",
        "Url",
        "Publisher",
        "Author",
        "Lsa_summary",
        "Luhn_summary",
        "Textrank_summary",
        "Lexrank_summary",
        "clean_text",
    ]
    return out[[col for col in keep_cols if col in out.columns]]


def filter_news_to_universe(
    tickers: Sequence[str],
    output_path: Path,
    news_path: Path = config.NEWS_CSV_PATH,
    start_date: str | None = config.START_DATE,
    end_date: str | None = config.END_DATE,
    chunksize: int = config.NEWS_CHUNK_SIZE,
    partitioned: bool = False,
) -> pd.DataFrame | None:
    """Filter raw news safely; return DataFrame only for non-partitioned output."""
    usecols = [col for col in config.NEWS_USE_COLUMNS if col != config.FULL_ARTICLE_COLUMN]
    parts: list[pd.DataFrame] = []
    part_idx = 0
    output_dir = output_path.with_suffix("")
    for chunk in tqdm(iter_news_chunks(news_path, usecols=usecols, chunksize=chunksize), desc="news chunks"):
        filtered = filter_news_chunk(chunk, tickers=tickers, start_date=start_date, end_date=end_date)
        if filtered.empty:
            continue
        filtered = deduplicate_articles(filtered)
        if partitioned:
            write_chunk_parquet(filtered, output_dir, part_idx)
            part_idx += 1
        else:
            parts.append(filtered)
    if partitioned:
        LOGGER.info("Wrote %d filtered news parquet parts to %s", part_idx, output_dir)
        return None
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    write_parquet_safe(result, output_path)
    return result


def summarize_news_coverage(news: pd.DataFrame) -> pd.DataFrame:
    out = news.copy()
    out["year"] = pd.to_datetime(out["date"]).dt.year
    return (
        out.groupby(["ticker", "year"], as_index=False)
        .size()
        .rename(columns={"size": "article_count"})
        .sort_values(["ticker", "year"])
    )

