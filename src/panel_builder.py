"""Build and validate the final weekly modeling panel."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def make_week_end_dates(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates)
    return parsed + pd.to_timedelta(4 - parsed.dt.weekday, unit="D")


def merge_feature_groups(
    base: pd.DataFrame,
    feature_frames: Iterable[pd.DataFrame],
    keys: tuple[str, str] = ("ticker", "week_end_date"),
) -> pd.DataFrame:
    out = base.copy()
    for frame in feature_frames:
        if frame is None or frame.empty:
            continue
        out = out.merge(frame, on=list(keys), how="left")
    return out


def build_model_panel(
    price_features: pd.DataFrame,
    labels: pd.DataFrame,
    keyword_features: pd.DataFrame | None = None,
    finbert_features: pd.DataFrame | None = None,
    climatebert_features: pd.DataFrame | None = None,
    embedding_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    labels = labels.drop(columns=[col for col in ["date"] if col in labels.columns])
    base = price_features.merge(labels, on=["ticker", "week_end_date"], how="inner")
    panel = merge_feature_groups(
        base,
        [
            keyword_features,
            finbert_features,
            climatebert_features,
            embedding_features,
        ],
    )
    return panel.sort_values(["week_end_date", "ticker"]).reset_index(drop=True)


def check_missingness(panel: pd.DataFrame) -> pd.DataFrame:
    return (
        panel.isna()
        .mean()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("missing_rate", ascending=False)
    )


def check_target_distribution(panel: pd.DataFrame, target_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in target_cols:
        if col not in panel.columns:
            continue
        series = panel[col].dropna()
        rows.append(
            {
                "target": col,
                "count": int(series.shape[0]),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "p25": series.quantile(0.25),
                "median": series.median(),
                "p75": series.quantile(0.75),
                "max": series.max(),
            }
        )
    return pd.DataFrame(rows)


def check_leakage_alignment(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "date" in panel.columns:
        rows.append({"check": "date_column_present", "passes": False, "detail": "Weekly panel should use week_end_date."})
    if "week_end_date" in panel.columns:
        rows.append({"check": "week_end_date_present", "passes": True, "detail": "Panel has weekly key."})
    future_cols = [col for col in panel.columns if col.startswith("future_")]
    rows.append({"check": "future_label_columns", "passes": bool(future_cols), "detail": ", ".join(future_cols)})
    return pd.DataFrame(rows)

