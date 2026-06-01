"""Past-only price and benchmark feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_trailing_returns(df: pd.DataFrame, horizons: tuple[int, ...] = (5, 10, 21, 63)) -> pd.DataFrame:
    out = df.sort_values(["ticker", "date"]).copy()
    for horizon in horizons:
        out[f"trailing_return_{horizon}d"] = out.groupby("ticker")["adj_close"].pct_change(horizon)
    return out


def add_rolling_volatility(df: pd.DataFrame, horizons: tuple[int, ...] = (5, 10, 21, 63)) -> pd.DataFrame:
    out = df.sort_values(["ticker", "date"]).copy()
    for horizon in horizons:
        out[f"rolling_vol_{horizon}d"] = (
            out.groupby("ticker")["return_1d"]
            .transform(lambda s, h=horizon: s.rolling(h, min_periods=max(2, h // 2)).std())
        )
    return out


def _rolling_drawdown(prices: pd.Series, horizon: int) -> pd.Series:
    rolling_max = prices.rolling(horizon, min_periods=max(2, horizon // 2)).max()
    return prices / rolling_max - 1.0


def add_rolling_drawdown(df: pd.DataFrame, horizons: tuple[int, ...] = (21, 63)) -> pd.DataFrame:
    out = df.sort_values(["ticker", "date"]).copy()
    for horizon in horizons:
        out[f"rolling_drawdown_{horizon}d"] = out.groupby("ticker")["adj_close"].transform(
            lambda s, h=horizon: _rolling_drawdown(s, h)
        )
    return out


def add_moving_average_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ticker", "date"]).copy()
    ma21 = out.groupby("ticker")["adj_close"].transform(lambda s: s.rolling(21, min_periods=10).mean())
    high252 = out.groupby("ticker")["adj_close"].transform(lambda s: s.rolling(252, min_periods=63).max())
    out["close_to_ma21"] = out["adj_close"] / ma21
    out["distance_from_52w_high"] = out["adj_close"] / high252 - 1.0
    return out


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["ticker", "date"]).copy()
    rolling_volume = out.groupby("ticker")["volume"].transform(lambda s: s.rolling(21, min_periods=10).mean())
    out["rolling_avg_volume_21d"] = rolling_volume
    out["volume_surprise"] = out["volume"] / rolling_volume - 1.0
    return out


def add_benchmark_features(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    benchmark_name: str,
    beta_window: int = 63,
) -> pd.DataFrame:
    out = prices.copy()
    bench = benchmark[["date", "return_1d"]].rename(columns={"return_1d": "benchmark_return_1d"})
    out = out.merge(bench, on="date", how="left")
    out["benchmark_ticker"] = benchmark_name
    out["excess_return_1d"] = out["return_1d"] - out["benchmark_return_1d"]

    def add_group_stats(ticker: str, group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        group["ticker"] = ticker
        cov = group["return_1d"].rolling(beta_window, min_periods=21).cov(group["benchmark_return_1d"])
        var = group["benchmark_return_1d"].rolling(beta_window, min_periods=21).var()
        group["rolling_beta_63d"] = cov / var.replace(0, np.nan)
        group["rolling_corr_63d"] = group["return_1d"].rolling(beta_window, min_periods=21).corr(
            group["benchmark_return_1d"]
        )
        return group

    frames = [add_group_stats(ticker, group) for ticker, group in out.groupby("ticker")]
    return pd.concat(frames, ignore_index=True)


def add_all_price_features(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    benchmark_name: str,
) -> pd.DataFrame:
    out = add_trailing_returns(prices)
    out = add_rolling_volatility(out)
    out = add_rolling_drawdown(out)
    out = add_moving_average_features(out)
    out = add_volume_features(out)
    out = add_benchmark_features(out, benchmark=benchmark, benchmark_name=benchmark_name)
    return out


def aggregate_price_features_weekly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    out["week_end_date"] = dates + pd.to_timedelta(4 - dates.dt.weekday, unit="D")
    out = out.sort_values(["ticker", "date"])
    weekly = out.groupby(["ticker", "week_end_date"], as_index=False).last()
    drop_cols = {"date", "open", "high", "low", "close", "adj_close", "volume", "source_ticker"}
    return weekly.drop(columns=[col for col in drop_cols if col in weekly.columns])

