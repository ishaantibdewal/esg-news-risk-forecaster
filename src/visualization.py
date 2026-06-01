"""Plotting helpers for the final report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_or_return(fig, output_path: Path | None):
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=160)
        plt.close(fig)
    return fig


def plot_news_coverage(coverage: pd.DataFrame, output_path: Path | None = None):
    pivot = coverage.pivot_table(index="year", columns="ticker", values="article_count", fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(ax=ax)
    ax.set_title("News Coverage By Ticker Over Time")
    ax.set_ylabel("Articles")
    return _save_or_return(fig, output_path)


def plot_vol_distribution(panel: pd.DataFrame, target: str, output_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    panel[target].dropna().hist(ax=ax, bins=40)
    ax.set_title(f"Distribution of {target}")
    ax.set_xlabel(target)
    ax.set_ylabel("Count")
    return _save_or_return(fig, output_path)


def plot_sentiment_distribution(df: pd.DataFrame, output_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    df["finbert_sentiment_score"].dropna().hist(ax=ax, bins=40)
    ax.set_title("FinBERT Sentiment Distribution")
    ax.set_xlabel("Positive probability - negative probability")
    return _save_or_return(fig, output_path)


def plot_climate_distribution(df: pd.DataFrame, output_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    df["climate_relevance_score"].dropna().hist(ax=ax, bins=40)
    ax.set_title("ClimateBERT Relevance Distribution")
    ax.set_xlabel("Climate relevance score")
    return _save_or_return(fig, output_path)


def plot_model_comparison(metrics: pd.DataFrame, metric: str, output_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics.pivot_table(index="feature_group", columns="model", values=metric).plot(kind="bar", ax=ax)
    ax.set_title(f"Model Comparison: {metric.upper()}")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)
    return _save_or_return(fig, output_path)


def plot_metric_heatmap(
    metrics: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path | None = None,
):
    data = metrics.dropna(subset=[metric]).copy()
    pivot = data.pivot_table(index="feature_group", columns="model", values=metric)
    fig_width = max(8, 1.6 * max(1, len(pivot.columns)))
    fig_height = max(5, 0.45 * max(1, len(pivot.index)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Feature group")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for row_idx, feature_group in enumerate(pivot.index):
        for col_idx, model in enumerate(pivot.columns):
            value = pivot.loc[feature_group, model]
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"{value:.4f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label=metric)
    return _save_or_return(fig, output_path)


def plot_metric_delta_bar(
    deltas: pd.DataFrame,
    task: str,
    title: str,
    output_path: Path | None = None,
):
    data = deltas[deltas["task"] == task].copy()
    if data.empty:
        raise ValueError(f"No delta rows for task: {task}")
    data = data.sort_values("delta_vs_price")
    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(data))))
    ax.barh(data["feature_group"], data["delta_vs_price"])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Delta vs price-only baseline")
    return _save_or_return(fig, output_path)


def plot_tuned_vs_untuned(
    tuned_comparison: pd.DataFrame,
    task: str,
    title: str,
    output_path: Path | None = None,
):
    data = tuned_comparison[tuned_comparison["task"] == task].copy()
    if data.empty:
        raise ValueError(f"No tuned comparison rows for task: {task}")
    value_cols = ["untuned_value", "tuned_value"]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(data))))
    data.set_index("feature_group")[value_cols].plot.barh(ax=ax)
    ax.set_title(title)
    ax.set_xlabel(data["metric"].iloc[0])
    return _save_or_return(fig, output_path)


def plot_quintile_volatility(quintiles: pd.DataFrame, output_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(7, 5))
    quintiles.plot.bar(x="predicted_vol_quintile", y="avg_realized_vol", ax=ax, legend=False)
    ax.set_title("Predicted Volatility Quintile vs Realized Volatility")
    ax.set_xlabel("Predicted risk quintile")
    ax.set_ylabel("Average realized volatility")
    return _save_or_return(fig, output_path)


def plot_quintile_returns(quintiles: pd.DataFrame, output_path: Path | None = None):
    if "avg_forward_return" not in quintiles.columns:
        raise ValueError("Quintile table does not contain avg_forward_return.")
    fig, ax = plt.subplots(figsize=(7, 5))
    quintiles.plot.bar(x="predicted_vol_quintile", y="avg_forward_return", ax=ax, legend=False)
    ax.set_title("Predicted Volatility Quintile vs Forward Returns")
    ax.set_xlabel("Predicted risk quintile")
    ax.set_ylabel("Average forward return")
    return _save_or_return(fig, output_path)


def plot_quintile_comparison(
    quintiles: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    output_path: Path | None = None,
):
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = quintiles.pivot_table(index="predicted_vol_quintile", columns="prediction_model", values=value_col)
    pivot.plot.bar(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted risk quintile")
    ax.set_ylabel(ylabel)
    return _save_or_return(fig, output_path)


def plot_feature_importance(importances: pd.DataFrame, output_path: Path | None = None, top_n: int = 25):
    data = importances.sort_values("importance", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.25)))
    data.sort_values("importance").plot.barh(x="feature", y="importance", ax=ax, legend=False)
    ax.set_title("Feature Importance")
    return _save_or_return(fig, output_path)


def plot_case_study_timeline(df: pd.DataFrame, ticker: str, output_path: Path | None = None):
    data = df[df["ticker"] == ticker].sort_values("week_end_date")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(data["week_end_date"], data.get("adj_close", data.get("close")))
    axes[0].set_ylabel("Price")
    axes[1].plot(data["week_end_date"], data.get("article_count", 0))
    axes[1].set_ylabel("News count")
    axes[2].plot(data["week_end_date"], data.get("finbert_negative_share", 0))
    axes[2].set_ylabel("FinBERT negative")
    axes[3].plot(data["week_end_date"], data.get("future_vol_21d", 0))
    axes[3].set_ylabel("Future vol")
    fig.suptitle(f"Case Study Timeline: {ticker}")
    return _save_or_return(fig, output_path)

