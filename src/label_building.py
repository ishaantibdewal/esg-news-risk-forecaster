"""Future volatility, return, and drawdown label construction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import config


def future_realized_volatility(returns: pd.Series, horizon: int) -> pd.Series:
    future_squared = pd.concat(
        [returns.shift(-step).pow(2) for step in range(1, horizon + 1)],
        axis=1,
    )
    return np.sqrt(future_squared.mean(axis=1))


def future_return(returns: pd.Series, horizon: int) -> pd.Series:
    future = pd.concat([1 + returns.shift(-step) for step in range(1, horizon + 1)], axis=1)
    return future.prod(axis=1) - 1


def _future_max_drawdown_from_prices(prices: pd.Series, horizon: int) -> pd.Series:
    values = prices.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    for idx in range(len(values)):
        window = values[idx + 1 : idx + horizon + 1]
        if len(window) < horizon or np.isnan(window).all():
            continue
        running_max = np.maximum.accumulate(window)
        drawdowns = window / running_max - 1.0
        result[idx] = np.nanmin(drawdowns)
    return pd.Series(result, index=prices.index)


def build_volatility_labels(
    prices: pd.DataFrame,
    horizons: Sequence[int] = config.HORIZONS,
    return_col: str = "return_1d",
) -> pd.DataFrame:
    out = prices.sort_values(["ticker", "date"]).copy()
    grouped = out.groupby("ticker", group_keys=False)
    for horizon in horizons:
        out[f"future_vol_{horizon}d"] = grouped[return_col].transform(
            lambda s, h=horizon: future_realized_volatility(s, h)
        )
        out[f"future_return_{horizon}d"] = grouped[return_col].transform(
            lambda s, h=horizon: future_return(s, h)
        )
    label_cols = ["ticker", "date"] + [
        col for col in out.columns if col.startswith("future_vol_") or col.startswith("future_return_")
    ]
    return out[label_cols]


def build_drawdown_labels_optional(
    prices: pd.DataFrame,
    horizons: Sequence[int] = (21, 63),
    price_col: str = "adj_close",
) -> pd.DataFrame:
    out = prices.sort_values(["ticker", "date"]).copy()
    grouped = out.groupby("ticker", group_keys=False)
    for horizon in horizons:
        out[f"future_mdd_{horizon}d"] = grouped[price_col].transform(
            lambda s, h=horizon: _future_max_drawdown_from_prices(s, h)
        )
    label_cols = ["ticker", "date"] + [col for col in out.columns if col.startswith("future_mdd_")]
    return out[label_cols]


def add_week_end_date(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    out["week_end_date"] = dates + pd.to_timedelta(4 - dates.dt.weekday, unit="D")
    return out


def aggregate_labels_weekly(labels: pd.DataFrame) -> pd.DataFrame:
    out = add_week_end_date(labels)
    out = out.sort_values(["ticker", "date"])
    return out.groupby(["ticker", "week_end_date"], as_index=False).last()


def build_high_vol_labels(
    weekly_labels: pd.DataFrame,
    horizons: Sequence[int] = config.HORIZONS,
    quantile: float = config.HIGH_VOL_QUANTILE,
) -> pd.DataFrame:
    out = weekly_labels.copy()
    for horizon in horizons:
        target = f"future_vol_{horizon}d"
        label = f"high_vol_{horizon}d"
        if target not in out.columns:
            continue
        thresholds = out.groupby("week_end_date")[target].transform(lambda s: s.quantile(quantile))
        out[label] = (out[target] >= thresholds).astype("Int64")
        out.loc[out[target].isna(), label] = pd.NA
    return out

