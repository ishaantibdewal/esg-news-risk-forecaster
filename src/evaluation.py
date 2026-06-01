"""Evaluation metrics and ESG2Risk-style quintile analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from . import config


def _valid_pair(y_true: pd.Series, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = pd.notna(y_true) & pd.notna(y_pred)
    return np.asarray(y_true[mask]), np.asarray(y_pred)[mask]


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y, pred = _valid_pair(y_true, y_pred)
    y = y.astype(float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "spearman": float(pd.Series(y).corr(pd.Series(pred), method="spearman")),
    }


def classification_metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y, score = _valid_pair(y_true, y_score)
    y = y.astype(int)
    pred = (score >= threshold).astype(int)
    metrics = {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else np.nan,
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else np.nan,
    }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def precision_at_top_k(y_true: pd.Series, y_score: np.ndarray, fraction: float = config.TOP_RISK_FRACTION) -> float:
    data = pd.DataFrame({"y": y_true, "score": y_score}).dropna()
    if data.empty:
        return np.nan
    n = max(1, int(np.ceil(len(data) * fraction)))
    return float(data.nlargest(n, "score")["y"].mean())


def build_quintile_table(
    panel: pd.DataFrame,
    prediction_col: str,
    realized_vol_col: str,
    forward_return_col: str | None = None,
    date_col: str = "week_end_date",
    n_quantiles: int = 5,
) -> pd.DataFrame:
    data = panel.dropna(subset=[prediction_col, realized_vol_col]).copy()

    def assign_quantile(group: pd.DataFrame) -> pd.Series:
        if group[prediction_col].nunique() < 2:
            return pd.Series([np.nan] * len(group), index=group.index)
        return pd.qcut(group[prediction_col], q=min(n_quantiles, group[prediction_col].nunique()), labels=False, duplicates="drop") + 1

    data["predicted_vol_quintile"] = data.groupby(date_col, group_keys=False).apply(assign_quantile)
    agg = {
        "stock_weeks": ("ticker", "size"),
        "avg_realized_vol": (realized_vol_col, "mean"),
        "std_realized_vol": (realized_vol_col, "std"),
        "avg_predicted_vol": (prediction_col, "mean"),
    }
    if forward_return_col is not None and forward_return_col in data.columns:
        agg["avg_forward_return"] = (forward_return_col, "mean")
    return data.groupby("predicted_vol_quintile").agg(**agg).reset_index()


def compare_model_groups(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(["task", "target", "feature_group", "model"]).reset_index(drop=True)


def metric_deltas_vs_baseline(
    metrics: pd.DataFrame,
    baseline_group: str = "price",
) -> pd.DataFrame:
    rows = []
    for task, task_df in metrics.groupby("task"):
        if task == "regression":
            metric = "rmse"
            best = task_df.loc[task_df.groupby("feature_group")[metric].idxmin()]
            baseline = best.loc[best["feature_group"] == baseline_group]
            if baseline.empty:
                continue
            baseline_value = float(baseline[metric].iloc[0])
            for _, row in best.iterrows():
                rows.append(
                    {
                        "task": task,
                        "metric": metric,
                        "feature_group": row["feature_group"],
                        "model": row["model"],
                        "metric_value": row[metric],
                        "baseline_value": baseline_value,
                        "delta_vs_price": baseline_value - row[metric],
                        "pct_change_vs_price": (baseline_value - row[metric]) / baseline_value,
                        "higher_is_better": False,
                    }
                )
        elif task == "classification":
            metric = "roc_auc"
            best = task_df.loc[task_df.groupby("feature_group")[metric].idxmax()]
            baseline = best.loc[best["feature_group"] == baseline_group]
            if baseline.empty:
                continue
            baseline_value = float(baseline[metric].iloc[0])
            for _, row in best.iterrows():
                rows.append(
                    {
                        "task": task,
                        "metric": metric,
                        "feature_group": row["feature_group"],
                        "model": row["model"],
                        "metric_value": row[metric],
                        "baseline_value": baseline_value,
                        "delta_vs_price": row[metric] - baseline_value,
                        "pct_change_vs_price": (row[metric] - baseline_value) / baseline_value,
                        "higher_is_better": True,
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_regression_delta(
    predictions: pd.DataFrame,
    candidate_col: str,
    baseline_col: str,
    target_col: str,
    n_iter: int = config.BOOTSTRAP_ITERATIONS,
    random_state: int = config.RANDOM_SEED,
) -> dict[str, float]:
    data = predictions[[target_col, candidate_col, baseline_col]].dropna().reset_index(drop=True)
    if data.empty:
        return {"mean_delta_rmse": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(random_state)
    deltas = []
    for _ in range(n_iter):
        idx = rng.integers(0, len(data), len(data))
        sample = data.iloc[idx]
        baseline_rmse = np.sqrt(mean_squared_error(sample[target_col], sample[baseline_col]))
        candidate_rmse = np.sqrt(mean_squared_error(sample[target_col], sample[candidate_col]))
        deltas.append(baseline_rmse - candidate_rmse)
    return {
        "mean_delta_rmse": float(np.mean(deltas)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
    }


def bootstrap_classification_delta(
    predictions: pd.DataFrame,
    candidate_col: str,
    baseline_col: str,
    target_col: str,
    n_iter: int = config.BOOTSTRAP_ITERATIONS,
    random_state: int = config.RANDOM_SEED,
) -> dict[str, float]:
    data = predictions[[target_col, candidate_col, baseline_col]].dropna().reset_index(drop=True)
    if data.empty or data[target_col].nunique() < 2:
        return {"mean_delta_roc_auc": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(random_state)
    deltas = []
    for _ in range(n_iter):
        idx = rng.integers(0, len(data), len(data))
        sample = data.iloc[idx]
        if sample[target_col].nunique() < 2:
            continue
        baseline_auc = roc_auc_score(sample[target_col], sample[baseline_col])
        candidate_auc = roc_auc_score(sample[target_col], sample[candidate_col])
        deltas.append(candidate_auc - baseline_auc)
    if not deltas:
        return {"mean_delta_roc_auc": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    return {
        "mean_delta_roc_auc": float(np.mean(deltas)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
    }

