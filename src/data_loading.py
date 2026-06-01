"""Safe data-loading helpers for raw FNSPID files."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd

from . import config


def get_file_size_gb(path: Path) -> float:
    return path.stat().st_size / 1e9


def inspect_csv_sample(path: Path, nrows: int = 1_000, **kwargs) -> pd.DataFrame:
    """Read a bounded CSV sample. Never use this for an unbounded raw read."""
    return pd.read_csv(path, nrows=nrows, low_memory=False, **kwargs)


def iter_news_chunks(
    path: Path = config.NEWS_CSV_PATH,
    usecols: Sequence[str] | None = None,
    chunksize: int = config.NEWS_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Yield bounded chunks from the large news CSV."""
    yield from pd.read_csv(
        path,
        usecols=list(usecols) if usecols is not None else None,
        chunksize=chunksize,
        low_memory=False,
    )


def list_price_files(price_dir: Path = config.PRICE_HISTORY_DIR) -> pd.DataFrame:
    rows = []
    for path in sorted(price_dir.glob("*.csv")):
        rows.append({"ticker": path.stem.upper(), "filename": path.name, "path": str(path)})
    return pd.DataFrame(rows)


def find_price_file(ticker: str, price_dir: Path = config.PRICE_HISTORY_DIR) -> Path | None:
    ticker_upper = ticker.upper()
    for path in price_dir.glob("*.csv"):
        if path.stem.upper() == ticker_upper:
            return path
    return None


def summarize_news_tickers(
    tickers: Sequence[str] | None = None,
    path: Path = config.NEWS_CSV_PATH,
    chunksize: int = config.NEWS_CHUNK_SIZE,
) -> pd.DataFrame:
    """Count news rows by ticker using only the ticker column."""
    wanted = {t.upper() for t in tickers} if tickers is not None else None
    counts: dict[str, int] = {}
    for chunk in iter_news_chunks(path, usecols=["Stock_symbol"], chunksize=chunksize):
        symbols = chunk["Stock_symbol"].astype("string").str.upper()
        if wanted is not None:
            symbols = symbols[symbols.isin(wanted)]
        for ticker, count in symbols.value_counts(dropna=True).items():
            counts[str(ticker)] = counts.get(str(ticker), 0) + int(count)
    return (
        pd.DataFrame({"ticker": list(counts), "news_rows": list(counts.values())})
        .sort_values(["news_rows", "ticker"], ascending=[False, True])
        .reset_index(drop=True)
    )


def read_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def write_parquet_safe(df: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)


def write_chunk_parquet(df: pd.DataFrame, output_dir: Path, part_idx: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"part_{part_idx:05d}.parquet"
    df.to_parquet(path, index=False)
    return path

