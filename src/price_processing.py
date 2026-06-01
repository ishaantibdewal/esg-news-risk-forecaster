"""Price-history cleaning and benchmark handling."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from . import config
from .data_loading import find_price_file, write_parquet_safe


PRICE_COLUMN_MAP = {
    "adj close": "adj_close",
    "Adj Close": "adj_close",
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


def normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={col: PRICE_COLUMN_MAP.get(col, col) for col in df.columns}).copy()
    out.columns = [col.strip().lower().replace(" ", "_") for col in out.columns]
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")
    if "adj_close" not in out.columns:
        out["adj_close"] = out["close"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[["date", "open", "high", "low", "close", "adj_close", "volume"]]


def read_price_file(path: Path, ticker: str | None = None) -> pd.DataFrame:
    out = normalize_price_columns(pd.read_csv(path))
    out["ticker"] = ticker.upper() if ticker else path.stem.upper()
    return out


def load_universe_prices(
    tickers: Sequence[str],
    price_dir: Path = config.PRICE_HISTORY_DIR,
    aliases: dict[str, list[str]] | None = config.TICKER_ALIASES,
) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        candidates = aliases.get(ticker, [ticker]) if aliases else [ticker]
        path = None
        source_ticker = ticker
        for candidate in candidates:
            path = find_price_file(candidate, price_dir=price_dir)
            if path is not None:
                source_ticker = candidate
                break
        if path is None:
            continue
        frame = read_price_file(path, ticker=ticker)
        frame["source_ticker"] = source_ticker.upper()
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, ignore_index=True)
    return compute_daily_returns(prices)


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.dropna(subset=["date", "ticker"]).copy()
    out = out.drop_duplicates(subset=["ticker", "date"], keep="last")
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out["price_for_return"] = out["adj_close"].fillna(out["close"])
    out["return_1d"] = out.groupby("ticker")["price_for_return"].pct_change()
    return out.drop(columns=["price_for_return"])


def load_benchmark_prices(
    benchmark_preference: Sequence[str] = config.BENCHMARK_PREFERENCE,
    price_dir: Path = config.PRICE_HISTORY_DIR,
) -> tuple[str, pd.DataFrame]:
    for benchmark in benchmark_preference:
        path = find_price_file(benchmark, price_dir=price_dir)
        if path is not None:
            return benchmark.upper(), compute_daily_returns(read_price_file(path, ticker=benchmark))
    raise FileNotFoundError(f"No benchmark found from {benchmark_preference}")


def validate_price_coverage(
    tickers: Sequence[str],
    price_dir: Path = config.PRICE_HISTORY_DIR,
    aliases: dict[str, list[str]] | None = config.TICKER_ALIASES,
) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        candidates = aliases.get(ticker, [ticker]) if aliases else [ticker]
        selected_path = None
        source_ticker = None
        for candidate in candidates:
            selected_path = find_price_file(candidate, price_dir=price_dir)
            if selected_path is not None:
                source_ticker = candidate.upper()
                break
        if selected_path is None:
            rows.append(
                {
                    "ticker": ticker,
                    "source_ticker": None,
                    "price_available": False,
                    "rows": 0,
                    "min_date": pd.NaT,
                    "max_date": pd.NaT,
                }
            )
            continue
        dates = pd.read_csv(selected_path, usecols=["date"])
        parsed = pd.to_datetime(dates["date"], errors="coerce")
        rows.append(
            {
                "ticker": ticker,
                "source_ticker": source_ticker,
                "price_available": True,
                "rows": len(parsed),
                "min_date": parsed.min(),
                "max_date": parsed.max(),
                "filename": selected_path.name,
            }
        )
    return pd.DataFrame(rows)


def save_clean_prices(
    tickers: Sequence[str],
    prices_path: Path,
    benchmark_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    prices = load_universe_prices(tickers)
    benchmark, benchmark_prices = load_benchmark_prices()
    write_parquet_safe(prices, prices_path)
    write_parquet_safe(benchmark_prices, benchmark_path)
    return prices, benchmark_prices, benchmark

