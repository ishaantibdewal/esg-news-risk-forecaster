"""Command-line orchestration for the ESG2Risk-inspired pipeline.

The commands are intentionally stage-based so expensive work such as scanning
the 23 GB news CSV or transformer scoring can be run, cached, and resumed.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from . import config
from .data_loading import get_file_size_gb, inspect_csv_sample, summarize_news_tickers, write_parquet_safe
from .esg_keywords import aggregate_keyword_features_weekly, score_esg_keywords
from .label_building import (
    aggregate_labels_weekly,
    build_drawdown_labels_optional,
    build_high_vol_labels,
    build_volatility_labels,
)
from .modeling import (
    get_feature_groups,
    make_time_split,
    predict_model,
    train_classification_models,
    train_regression_models,
    tune_models_on_validation,
)
from .evaluation import (
    bootstrap_classification_delta,
    bootstrap_regression_delta,
    build_quintile_table,
    classification_metrics,
    compare_model_groups,
    metric_deltas_vs_baseline,
    precision_at_top_k,
    regression_metrics,
)
from .climatebert_scoring import aggregate_climatebert_weekly, score_climatebert_articles
from .finbert_scoring import add_finbert_sentiment_score, aggregate_finbert_weekly, score_finbert_articles
from .news_filtering import filter_news_to_universe, summarize_news_coverage
from .panel_builder import build_model_panel, check_leakage_alignment, check_missingness, check_target_distribution
from .paths import ensure_directories, interim_path, output_path, processed_path
from .price_features import add_all_price_features, aggregate_price_features_weekly
from .price_processing import load_benchmark_prices, load_universe_prices, validate_price_coverage
from .utils import setup_logging
from .visualization import (
    plot_climate_distribution,
    plot_metric_delta_bar,
    plot_metric_heatmap,
    plot_model_comparison,
    plot_news_coverage,
    plot_quintile_comparison,
    plot_quintile_returns,
    plot_quintile_volatility,
    plot_sentiment_distribution,
    plot_tuned_vs_untuned,
    plot_vol_distribution,
)

LOGGER = logging.getLogger(__name__)


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join([header, sep, *rows])


def _add_news_week_flag(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    signal_cols = [col for col in ["esg_article_count", "finbert_article_count", "climate_article_count"] if col in out.columns]
    if not signal_cols:
        out["has_esg_news_week"] = False
        return out
    out["has_esg_news_week"] = out[signal_cols].fillna(0).sum(axis=1) >= config.NEWS_WEEK_ARTICLE_MIN
    return out


def resolve_default_universe() -> list[str]:
    coverage = validate_price_coverage(config.DEFAULT_TICKERS)
    end_date = pd.Timestamp(config.END_DATE) if config.END_DATE else coverage["max_date"].max()
    cutoff = end_date - pd.Timedelta(days=90)
    usable = coverage.loc[
        coverage["price_available"] & (pd.to_datetime(coverage["max_date"]) >= cutoff),
        "ticker",
    ].tolist()
    return usable


def command_setup(_: argparse.Namespace) -> None:
    ensure_directories()
    LOGGER.info("Created project output directories.")


def command_inspect(args: argparse.Namespace) -> None:
    ensure_directories()
    sample = inspect_csv_sample(config.NEWS_CSV_PATH, nrows=args.nrows)
    sample.to_csv(output_path("tables", "news_sample_columns.csv"), index=False)
    price_coverage = validate_price_coverage(config.DEFAULT_TICKERS + config.BENCHMARK_PREFERENCE)
    price_coverage.to_csv(output_path("tables", "price_coverage.csv"), index=False)
    summary = pd.DataFrame(
        [
            {"item": "news_csv_gb", "value": get_file_size_gb(config.NEWS_CSV_PATH)},
            {"item": "price_file_count", "value": len(list(config.PRICE_HISTORY_DIR.glob("*.csv")))},
            {"item": "news_columns", "value": "|".join(sample.columns)},
        ]
    )
    summary.to_csv(output_path("tables", "raw_data_inventory.csv"), index=False)
    if args.scan_news:
        tickers = sorted(set(config.DEFAULT_TICKERS + [alias for values in config.TICKER_ALIASES.values() for alias in values]))
        news_counts = summarize_news_tickers(tickers=tickers)
        news_counts.to_csv(output_path("tables", "news_ticker_coverage.csv"), index=False)
        universe = price_coverage.merge(news_counts, on="ticker", how="left")
    else:
        universe = price_coverage.copy()
        universe["news_rows"] = pd.NA
    selected = set(resolve_default_universe())
    universe["selected_default_universe"] = universe["ticker"].isin(selected)
    universe.to_csv(output_path("tables", "universe_coverage.csv"), index=False)


def command_filter_news(args: argparse.Namespace) -> None:
    ensure_directories()
    tickers = args.tickers or resolve_default_universe()
    filtered = filter_news_to_universe(
        tickers=tickers,
        output_path=interim_path("news_filtered_raw.parquet"),
        start_date=args.start_date,
        end_date=args.end_date,
        chunksize=args.chunksize,
        partitioned=args.partitioned,
    )
    if filtered is not None:
        write_parquet_safe(filtered, interim_path("news_cleaned_text.parquet"))
        coverage = summarize_news_coverage(filtered)
        coverage.to_csv(output_path("tables", "news_coverage_by_ticker_year.csv"), index=False)


def command_prices_labels(_: argparse.Namespace) -> None:
    ensure_directories()
    tickers = resolve_default_universe()
    prices = load_universe_prices(tickers)
    benchmark_name, benchmark = load_benchmark_prices()
    write_parquet_safe(prices, interim_path("price_history_cleaned.parquet"))
    write_parquet_safe(benchmark, interim_path("benchmark_history_cleaned.parquet"))
    labels = build_volatility_labels(prices)
    labels_weekly = build_high_vol_labels(aggregate_labels_weekly(labels))
    write_parquet_safe(labels_weekly, processed_path("labels_volatility_5d_10d_21d_63d.parquet"))
    drawdowns = build_drawdown_labels_optional(prices)
    write_parquet_safe(aggregate_labels_weekly(drawdowns), processed_path("labels_drawdown_optional.parquet"))
    pd.DataFrame({"benchmark": [benchmark_name]}).to_csv(output_path("tables", "selected_benchmark.csv"), index=False)


def command_features(_: argparse.Namespace) -> None:
    ensure_directories()
    prices = pd.read_parquet(interim_path("price_history_cleaned.parquet"))
    benchmark = pd.read_parquet(interim_path("benchmark_history_cleaned.parquet"))
    benchmark_name = pd.read_csv(output_path("tables", "selected_benchmark.csv"))["benchmark"].iloc[0]
    price_features = aggregate_price_features_weekly(add_all_price_features(prices, benchmark, benchmark_name))
    write_parquet_safe(price_features, processed_path("price_features_weekly.parquet"))

    news_path = interim_path("news_cleaned_text.parquet")
    if news_path.exists():
        news = pd.read_parquet(news_path)
        keyword_articles = score_esg_keywords(news)
        write_parquet_safe(keyword_articles, interim_path("news_esg_keyword_filtered.parquet"))
        keyword_weekly = aggregate_keyword_features_weekly(keyword_articles)
        write_parquet_safe(keyword_weekly, processed_path("news_keyword_features_weekly.parquet"))


def _load_transformer_input(esg_only: bool = True, max_articles: int | None = None) -> pd.DataFrame:
    path = interim_path("news_esg_keyword_filtered.parquet")
    if not path.exists():
        path = interim_path("news_cleaned_text.parquet")
    news = pd.read_parquet(path)
    if esg_only and "is_esg_keyword_article" in news.columns:
        news = news.loc[news["is_esg_keyword_article"]].copy()
    news = news.dropna(subset=["clean_text"]).sort_values(["date", "ticker"]).reset_index(drop=True)
    if max_articles is not None:
        news = news.head(max_articles).copy()
    return news


def command_finbert(args: argparse.Namespace) -> None:
    ensure_directories()
    news = _load_transformer_input(esg_only=not args.all_news, max_articles=args.max_articles)
    scored = score_finbert_articles(
        news,
        batch_size=args.batch_size,
        device=args.device,
        model_name=args.model_name,
    )
    scored = add_finbert_sentiment_score(scored)
    write_parquet_safe(scored, interim_path("news_finbert_scored.parquet"))
    write_parquet_safe(aggregate_finbert_weekly(scored), processed_path("news_finbert_features_weekly.parquet"))


def command_climatebert(args: argparse.Namespace) -> None:
    ensure_directories()
    news = _load_transformer_input(esg_only=not args.all_news, max_articles=args.max_articles)
    finbert_path = interim_path("news_finbert_scored.parquet")
    if finbert_path.exists():
        finbert_cols = [
            "ticker",
            "date",
            "Url",
            "Article_title",
            "finbert_negative_prob",
            "finbert_positive_prob",
            "finbert_neutral_prob",
            "finbert_label",
            "finbert_sentiment_score",
            "is_finbert_negative",
        ]
        finbert = pd.read_parquet(finbert_path)
        keys = [col for col in ["ticker", "date", "Url", "Article_title"] if col in news.columns and col in finbert.columns]
        if keys:
            news = news.merge(finbert[[col for col in finbert_cols if col in finbert.columns]], on=keys, how="left")
    scored = score_climatebert_articles(
        news,
        batch_size=args.batch_size,
        device=args.device,
        model_name=args.model_name,
    )
    write_parquet_safe(scored, interim_path("news_climatebert_scored.parquet"))
    write_parquet_safe(aggregate_climatebert_weekly(scored), processed_path("news_climatebert_features_weekly.parquet"))


def command_panel(_: argparse.Namespace) -> None:
    ensure_directories()
    price_features = pd.read_parquet(processed_path("price_features_weekly.parquet"))
    labels = pd.read_parquet(processed_path("labels_volatility_5d_10d_21d_63d.parquet"))
    optional = {}
    for key, filename in [
        ("keyword_features", "news_keyword_features_weekly.parquet"),
        ("finbert_features", "news_finbert_features_weekly.parquet"),
        ("climatebert_features", "news_climatebert_features_weekly.parquet"),
        ("embedding_features", "news_embedding_features_weekly.parquet"),
    ]:
        path = processed_path(filename)
        optional[key] = pd.read_parquet(path) if path.exists() else None
    panel = build_model_panel(price_features, labels, **optional)
    panel = panel[pd.to_datetime(panel["week_end_date"]) >= pd.Timestamp(config.MODEL_START_DATE)].copy()
    panel = _add_news_week_flag(panel)
    write_parquet_safe(panel, processed_path("model_panel_weekly.parquet"))
    check_missingness(panel).to_csv(output_path("tables", "panel_missingness.csv"), index=False)
    check_target_distribution(panel, [config.PRIMARY_REGRESSION_TARGET, config.REPRODUCTION_REGRESSION_TARGET]).to_csv(
        output_path("tables", "target_distribution.csv"), index=False
    )
    check_leakage_alignment(panel).to_csv(output_path("tables", "leakage_checks.csv"), index=False)


def command_model_eval(_: argparse.Namespace) -> None:
    ensure_directories()
    panel = pd.read_parquet(processed_path("model_panel_weekly.parquet"))
    panel = _add_news_week_flag(panel)
    split = make_time_split(panel)
    groups = get_feature_groups(panel)
    rows = []
    predictions = panel[
        [
            "ticker",
            "week_end_date",
            "has_esg_news_week",
            config.PRIMARY_REGRESSION_TARGET,
            config.PRIMARY_CLASSIFICATION_TARGET,
            "future_return_21d",
        ]
    ].copy()
    model_groups = {
        "price": groups["price"],
        "keyword": groups["keyword"],
        "finbert": groups["finbert"],
        "climatebert": groups["climatebert"],
        "all_news": groups["all_news"],
        "price_keyword": groups["price_keyword"],
        "price_finbert": groups["price_finbert"],
        "price_climatebert": groups["price_climatebert"],
        "price_finbert_climatebert": groups["price_finbert_climatebert"],
        "price_all_news": groups["price_all_news"],
        "full": groups["all"],
    }
    evaluation_slices = {
        "all_weeks": split.test,
        "news_weeks": split.test & panel["has_esg_news_week"],
    }
    for group_name, features in model_groups.items():
        if not features:
            continue
        reg_models = train_regression_models(panel, features, split=split)
        for model_name, model in reg_models.items():
            pred = predict_model(model, panel.loc[split.test], features, task="regression")
            pred_col = f"reg__{group_name}__{model_name}"
            predictions[pred_col] = pd.NA
            predictions.loc[split.test, pred_col] = pred
            for slice_name, slice_mask in evaluation_slices.items():
                eval_mask = slice_mask & split.test
                if not eval_mask.any():
                    continue
                slice_pred = predictions.loc[eval_mask, pred_col].astype(float).to_numpy()
                rows.append(
                    {
                        "evaluation_slice": slice_name,
                        "task": "regression",
                        "target": config.PRIMARY_REGRESSION_TARGET,
                        "feature_group": group_name,
                        "model": model_name,
                        "test_rows": int(eval_mask.sum()),
                        **regression_metrics(panel.loc[eval_mask, config.PRIMARY_REGRESSION_TARGET], slice_pred),
                    }
                )
        if config.PRIMARY_CLASSIFICATION_TARGET in panel.columns:
            clf_models = train_classification_models(panel, features, split=split)
            for model_name, model in clf_models.items():
                score = predict_model(model, panel.loc[split.test], features, task="classification")
                score_col = f"clf__{group_name}__{model_name}"
                predictions[score_col] = pd.NA
                predictions.loc[split.test, score_col] = score
                for slice_name, slice_mask in evaluation_slices.items():
                    eval_mask = slice_mask & split.test
                    if not eval_mask.any():
                        continue
                    slice_score = predictions.loc[eval_mask, score_col].astype(float).to_numpy()
                    rows.append(
                        {
                            "evaluation_slice": slice_name,
                            "task": "classification",
                            "target": config.PRIMARY_CLASSIFICATION_TARGET,
                            "feature_group": group_name,
                            "model": model_name,
                            "test_rows": int(eval_mask.sum()),
                            **classification_metrics(panel.loc[eval_mask, config.PRIMARY_CLASSIFICATION_TARGET], slice_score),
                            "precision_at_top_10pct": precision_at_top_k(
                                panel.loc[eval_mask, config.PRIMARY_CLASSIFICATION_TARGET], slice_score
                            ),
                        }
                    )
    metrics = compare_model_groups(rows)
    metrics.to_csv(output_path("tables", "model_metrics.csv"), index=False)
    predictions.to_csv(output_path("tables", "test_predictions.csv"), index=False)

    deltas = metric_deltas_vs_baseline(metrics.loc[metrics["evaluation_slice"] == "all_weeks"])
    deltas.to_csv(output_path("tables", "metric_deltas_vs_price.csv"), index=False)

    regression_all = metrics[(metrics["evaluation_slice"] == "all_weeks") & (metrics["task"] == "regression")]
    classification_all = metrics[(metrics["evaluation_slice"] == "all_weeks") & (metrics["task"] == "classification")]
    quintile_frames = []
    robustness_rows = []
    if not regression_all.empty:
        best_by_group = regression_all.loc[regression_all.groupby("feature_group")["rmse"].idxmin()]
        price_best = best_by_group[best_by_group["feature_group"] == "price"].iloc[0]
        news_aug = best_by_group[best_by_group["feature_group"].str.startswith("price_")]
        news_aug = news_aug[news_aug["feature_group"] != "price"]
        best_news = news_aug.sort_values("rmse").iloc[0] if not news_aug.empty else price_best
        for label, row in [("price_baseline", price_best), ("best_news_augmented", best_news)]:
            pred_col = f"reg__{row['feature_group']}__{row['model']}"
            quintiles = build_quintile_table(
                predictions.dropna(subset=[pred_col]),
                prediction_col=pred_col,
                realized_vol_col=config.PRIMARY_REGRESSION_TARGET,
                forward_return_col="future_return_21d",
            )
            quintiles["prediction_model"] = label
            quintiles["feature_group"] = row["feature_group"]
            quintiles["model"] = row["model"]
            quintile_frames.append(quintiles)
        if best_news["feature_group"] != "price":
            robustness_rows.append(
                {
                    "task": "regression",
                    "candidate": best_news["feature_group"],
                    "candidate_model": best_news["model"],
                    "baseline": "price",
                    "baseline_model": price_best["model"],
                    **bootstrap_regression_delta(
                        predictions,
                        candidate_col=f"reg__{best_news['feature_group']}__{best_news['model']}",
                        baseline_col=f"reg__{price_best['feature_group']}__{price_best['model']}",
                        target_col=config.PRIMARY_REGRESSION_TARGET,
                    ),
                }
            )
    if not classification_all.empty:
        best_by_group = classification_all.loc[classification_all.groupby("feature_group")["roc_auc"].idxmax()]
        price_best = best_by_group[best_by_group["feature_group"] == "price"].iloc[0]
        news_aug = best_by_group[best_by_group["feature_group"].str.startswith("price_")]
        news_aug = news_aug[news_aug["feature_group"] != "price"]
        best_news = news_aug.sort_values("roc_auc", ascending=False).iloc[0] if not news_aug.empty else price_best
        if best_news["feature_group"] != "price":
            robustness_rows.append(
                {
                    "task": "classification",
                    "candidate": best_news["feature_group"],
                    "candidate_model": best_news["model"],
                    "baseline": "price",
                    "baseline_model": price_best["model"],
                    **bootstrap_classification_delta(
                        predictions,
                        candidate_col=f"clf__{best_news['feature_group']}__{best_news['model']}",
                        baseline_col=f"clf__{price_best['feature_group']}__{price_best['model']}",
                        target_col=config.PRIMARY_CLASSIFICATION_TARGET,
                    ),
                }
            )
    if quintile_frames:
        pd.concat(quintile_frames, ignore_index=True).to_csv(output_path("tables", "predicted_risk_quintiles.csv"), index=False)
    if robustness_rows:
        pd.DataFrame(robustness_rows).to_csv(output_path("tables", "bootstrap_metric_deltas.csv"), index=False)


def command_tuned_model_eval(args: argparse.Namespace) -> None:
    ensure_directories()
    panel = pd.read_parquet(processed_path("model_panel_weekly.parquet"))
    panel = _add_news_week_flag(panel)
    split = make_time_split(panel)
    groups = get_feature_groups(panel)
    selected_groups = args.groups or ["price", "price_climatebert", "price_all_news", "full"]
    rows = []
    tuning_records = []
    predictions = panel[
        [
            "ticker",
            "week_end_date",
            "has_esg_news_week",
            config.PRIMARY_REGRESSION_TARGET,
            config.PRIMARY_CLASSIFICATION_TARGET,
            "future_return_21d",
        ]
    ].copy()
    evaluation_slices = {
        "all_weeks": split.test,
        "news_weeks": split.test & panel["has_esg_news_week"],
    }
    for group_name in selected_groups:
        features = groups["all"] if group_name == "full" else groups.get(group_name, [])
        if not features:
            LOGGER.warning("Skipping %s because it has no features.", group_name)
            continue
        reg_models, reg_records = tune_models_on_validation(
            panel,
            features,
            config.PRIMARY_REGRESSION_TARGET,
            split,
            task="regression",
        )
        if not reg_records.empty:
            reg_records["feature_group"] = group_name
            tuning_records.append(reg_records)
        for model_name, model in reg_models.items():
            pred = predict_model(model, panel.loc[split.test], features, task="regression")
            pred_col = f"tuned_reg__{group_name}__{model_name}"
            predictions[pred_col] = pd.NA
            predictions.loc[split.test, pred_col] = pred
            for slice_name, slice_mask in evaluation_slices.items():
                eval_mask = slice_mask & split.test
                if not eval_mask.any():
                    continue
                rows.append(
                    {
                        "evaluation_slice": slice_name,
                        "task": "regression",
                        "target": config.PRIMARY_REGRESSION_TARGET,
                        "feature_group": group_name,
                        "model": model_name,
                        "test_rows": int(eval_mask.sum()),
                        **regression_metrics(
                            panel.loc[eval_mask, config.PRIMARY_REGRESSION_TARGET],
                            predictions.loc[eval_mask, pred_col].astype(float).to_numpy(),
                        ),
                    }
                )
        if config.PRIMARY_CLASSIFICATION_TARGET in panel.columns:
            clf_models, clf_records = tune_models_on_validation(
                panel,
                features,
                config.PRIMARY_CLASSIFICATION_TARGET,
                split,
                task="classification",
            )
            if not clf_records.empty:
                clf_records["feature_group"] = group_name
                tuning_records.append(clf_records)
            for model_name, model in clf_models.items():
                score = predict_model(model, panel.loc[split.test], features, task="classification")
                score_col = f"tuned_clf__{group_name}__{model_name}"
                predictions[score_col] = pd.NA
                predictions.loc[split.test, score_col] = score
                for slice_name, slice_mask in evaluation_slices.items():
                    eval_mask = slice_mask & split.test
                    if not eval_mask.any():
                        continue
                    slice_score = predictions.loc[eval_mask, score_col].astype(float).to_numpy()
                    rows.append(
                        {
                            "evaluation_slice": slice_name,
                            "task": "classification",
                            "target": config.PRIMARY_CLASSIFICATION_TARGET,
                            "feature_group": group_name,
                            "model": model_name,
                            "test_rows": int(eval_mask.sum()),
                            **classification_metrics(panel.loc[eval_mask, config.PRIMARY_CLASSIFICATION_TARGET], slice_score),
                            "precision_at_top_10pct": precision_at_top_k(
                                panel.loc[eval_mask, config.PRIMARY_CLASSIFICATION_TARGET], slice_score
                            ),
                        }
                    )
    tuned_metrics = compare_model_groups(rows)
    tuned_metrics.to_csv(output_path("tables", "tuned_model_metrics.csv"), index=False)
    predictions.to_csv(output_path("tables", "tuned_test_predictions.csv"), index=False)
    if tuning_records:
        tuning_table = pd.concat(tuning_records, ignore_index=True)
        tuning_table.to_csv(output_path("tables", "hyperparameter_tuning_trials.csv"), index=False)

    base_path = output_path("tables", "model_metrics.csv")
    if base_path.exists() and not tuned_metrics.empty:
        base = pd.read_csv(base_path)
        comparison_rows = []
        for task, metric, ascending in [
            ("regression", "rmse", True),
            ("classification", "roc_auc", False),
        ]:
            base_task = base[(base["evaluation_slice"] == "all_weeks") & (base["task"] == task)]
            tuned_task = tuned_metrics[(tuned_metrics["evaluation_slice"] == "all_weeks") & (tuned_metrics["task"] == task)]
            for group_name in selected_groups:
                base_group = base_task[base_task["feature_group"] == group_name]
                tuned_group = tuned_task[tuned_task["feature_group"] == group_name]
                if base_group.empty or tuned_group.empty:
                    continue
                base_best = base_group.sort_values(metric, ascending=ascending).iloc[0]
                tuned_best = tuned_group.sort_values(metric, ascending=ascending).iloc[0]
                delta = (base_best[metric] - tuned_best[metric]) if task == "regression" else (tuned_best[metric] - base_best[metric])
                comparison_rows.append(
                    {
                        "task": task,
                        "metric": metric,
                        "feature_group": group_name,
                        "untuned_model": base_best["model"],
                        "untuned_value": base_best[metric],
                        "tuned_model": tuned_best["model"],
                        "tuned_value": tuned_best[metric],
                        "improvement": delta,
                    }
                )
        pd.DataFrame(comparison_rows).to_csv(output_path("tables", "tuned_vs_untuned.csv"), index=False)


def command_report(_: argparse.Namespace) -> None:
    ensure_directories()
    panel = pd.read_parquet(processed_path("model_panel_weekly.parquet"))
    panel = _add_news_week_flag(panel)
    metrics = pd.read_csv(output_path("tables", "model_metrics.csv"))
    quintiles_path = output_path("tables", "predicted_risk_quintiles.csv")
    quintiles = pd.read_csv(quintiles_path) if quintiles_path.exists() else pd.DataFrame()
    deltas_path = output_path("tables", "metric_deltas_vs_price.csv")
    deltas = pd.read_csv(deltas_path) if deltas_path.exists() else pd.DataFrame()
    bootstrap_path = output_path("tables", "bootstrap_metric_deltas.csv")
    bootstrap = pd.read_csv(bootstrap_path) if bootstrap_path.exists() else pd.DataFrame()
    tuned_path = output_path("tables", "tuned_vs_untuned.csv")
    tuned_comparison = pd.read_csv(tuned_path) if tuned_path.exists() else pd.DataFrame()
    tuned_metrics_path = output_path("tables", "tuned_model_metrics.csv")
    tuned_metrics = pd.read_csv(tuned_metrics_path) if tuned_metrics_path.exists() else pd.DataFrame()

    coverage_path = output_path("tables", "news_coverage_by_ticker_year.csv")
    if coverage_path.exists():
        plot_news_coverage(pd.read_csv(coverage_path), output_path("figures", "news_coverage_by_ticker_year.png"))
    plot_vol_distribution(panel, config.PRIMARY_REGRESSION_TARGET, output_path("figures", "future_vol_21d_distribution.png"))

    finbert_path = interim_path("news_finbert_scored.parquet")
    if finbert_path.exists():
        plot_sentiment_distribution(pd.read_parquet(finbert_path), output_path("figures", "finbert_sentiment_distribution.png"))
    climate_path = interim_path("news_climatebert_scored.parquet")
    if climate_path.exists():
        plot_climate_distribution(pd.read_parquet(climate_path), output_path("figures", "climatebert_relevance_distribution.png"))

    metrics_all = metrics.loc[metrics["evaluation_slice"] == "all_weeks"] if "evaluation_slice" in metrics.columns else metrics
    metrics_news = metrics.loc[metrics["evaluation_slice"] == "news_weeks"] if "evaluation_slice" in metrics.columns else pd.DataFrame()

    regression = metrics_all.loc[metrics_all["task"] == "regression"]
    if not regression.empty:
        plot_model_comparison(regression, "rmse", output_path("figures", "model_comparison_rmse.png"))
        plot_model_comparison(regression, "mae", output_path("figures", "model_comparison_mae.png"))
    classification = metrics_all.loc[metrics_all["task"] == "classification"]
    if not classification.empty:
        plot_model_comparison(classification, "roc_auc", output_path("figures", "model_comparison_roc_auc.png"))

    for slice_name, slice_metrics in {
        "all_weeks": metrics_all,
        "news_weeks": metrics_news,
    }.items():
        if slice_metrics.empty:
            continue
        reg_slice = slice_metrics.loc[slice_metrics["task"] == "regression"]
        if not reg_slice.empty:
            for metric in ["rmse", "mae", "spearman"]:
                plot_metric_heatmap(
                    reg_slice,
                    metric,
                    f"All Regression Models: {metric.upper()} ({slice_name.replace('_', ' ')})",
                    output_path("figures", f"all_models_regression_{metric}_{slice_name}.png"),
                )
        clf_slice = slice_metrics.loc[slice_metrics["task"] == "classification"]
        if not clf_slice.empty:
            for metric in ["roc_auc", "pr_auc", "f1", "precision_at_top_10pct"]:
                plot_metric_heatmap(
                    clf_slice,
                    metric,
                    f"All Classification Models: {metric.upper()} ({slice_name.replace('_', ' ')})",
                    output_path("figures", f"all_models_classification_{metric}_{slice_name}.png"),
                )

    if not tuned_metrics.empty:
        tuned_all = (
            tuned_metrics.loc[tuned_metrics["evaluation_slice"] == "all_weeks"]
            if "evaluation_slice" in tuned_metrics.columns
            else tuned_metrics
        )
        tuned_news = (
            tuned_metrics.loc[tuned_metrics["evaluation_slice"] == "news_weeks"]
            if "evaluation_slice" in tuned_metrics.columns
            else pd.DataFrame()
        )
        for slice_name, slice_metrics in {
            "all_weeks": tuned_all,
            "news_weeks": tuned_news,
        }.items():
            if slice_metrics.empty:
                continue
            reg_slice = slice_metrics.loc[slice_metrics["task"] == "regression"]
            if not reg_slice.empty:
                plot_metric_heatmap(
                    reg_slice,
                    "rmse",
                    f"Tuned Regression Models: RMSE ({slice_name.replace('_', ' ')})",
                    output_path("figures", f"all_models_tuned_regression_rmse_{slice_name}.png"),
                )
            clf_slice = slice_metrics.loc[slice_metrics["task"] == "classification"]
            if not clf_slice.empty:
                plot_metric_heatmap(
                    clf_slice,
                    "roc_auc",
                    f"Tuned Classification Models: ROC-AUC ({slice_name.replace('_', ' ')})",
                    output_path("figures", f"all_models_tuned_classification_roc_auc_{slice_name}.png"),
                )

    if not deltas.empty:
        plot_metric_delta_bar(
            deltas,
            "regression",
            "Regression Delta vs Price-Only Baseline",
            output_path("figures", "metric_deltas_vs_price_regression.png"),
        )
        plot_metric_delta_bar(
            deltas,
            "classification",
            "Classification Delta vs Price-Only Baseline",
            output_path("figures", "metric_deltas_vs_price_classification.png"),
        )

    if not tuned_comparison.empty:
        plot_tuned_vs_untuned(
            tuned_comparison,
            "regression",
            "Tuned vs Untuned Regression Models",
            output_path("figures", "tuned_vs_untuned_regression.png"),
        )
        plot_tuned_vs_untuned(
            tuned_comparison,
            "classification",
            "Tuned vs Untuned Classification Models",
            output_path("figures", "tuned_vs_untuned_classification.png"),
        )

    if not quintiles.empty:
        if "prediction_model" in quintiles.columns:
            plot_quintile_comparison(
                quintiles,
                "avg_realized_vol",
                "Predicted Volatility Quintiles: Price Baseline vs Best News Model",
                "Average realized volatility",
                output_path("figures", "quintile_realized_volatility.png"),
            )
        else:
            plot_quintile_volatility(quintiles, output_path("figures", "quintile_realized_volatility.png"))
        if "avg_forward_return" in quintiles.columns:
            if "prediction_model" in quintiles.columns:
                plot_quintile_comparison(
                    quintiles,
                    "avg_forward_return",
                    "Predicted Volatility Quintiles vs Forward Returns",
                    "Average forward return",
                    output_path("figures", "quintile_forward_returns.png"),
                )
            else:
                plot_quintile_returns(quintiles, output_path("figures", "quintile_forward_returns.png"))

    best_reg = regression.sort_values("rmse").head(10)
    best_clf = classification.sort_values("roc_auc", ascending=False).head(10)
    best_reg_news = (
        metrics_news.loc[metrics_news["task"] == "regression"].sort_values("rmse").head(10)
        if not metrics_news.empty
        else pd.DataFrame()
    )
    best_clf_news = (
        metrics_news.loc[metrics_news["task"] == "classification"].sort_values("roc_auc", ascending=False).head(10)
        if not metrics_news.empty
        else pd.DataFrame()
    )
    summary = [
        "# ESG2Risk-Inspired Project Results Summary",
        "",
        "This report summarizes outputs generated by the improved DSC 148 pipeline.",
        "",
        "## Data Outputs",
        "",
        f"- Final weekly panel rows: {len(panel):,}",
        f"- Tickers: {panel['ticker'].nunique():,}",
        f"- Date range: {panel['week_end_date'].min()} to {panel['week_end_date'].max()}",
        f"- ESG/news ticker-weeks: {int(panel['has_esg_news_week'].sum()):,}",
        f"- Modeling starts at: {config.MODEL_START_DATE}",
        "",
        "## Best Regression Models By RMSE: All Test Weeks",
        "",
        _df_to_markdown(best_reg[["feature_group", "model", "rmse", "mae", "r2", "spearman"]])
        if not best_reg.empty
        else "No regression metrics available.",
        "",
        "## Best Classification Models By ROC-AUC: All Test Weeks",
        "",
        _df_to_markdown(best_clf[["feature_group", "model", "roc_auc", "pr_auc", "f1", "precision_at_top_10pct"]])
        if not best_clf.empty
        else "No classification metrics available.",
        "",
        "## Best Regression Models By RMSE: ESG/News Test Weeks Only",
        "",
        _df_to_markdown(best_reg_news[["feature_group", "model", "rmse", "mae", "r2", "spearman", "test_rows"]])
        if not best_reg_news.empty
        else "No news-week regression metrics available.",
        "",
        "## Best Classification Models By ROC-AUC: ESG/News Test Weeks Only",
        "",
        _df_to_markdown(best_clf_news[["feature_group", "model", "roc_auc", "pr_auc", "f1", "precision_at_top_10pct", "test_rows"]])
        if not best_clf_news.empty
        else "No news-week classification metrics available.",
        "",
        "## Metric Deltas vs Price-Only Baseline",
        "",
        _df_to_markdown(deltas) if not deltas.empty else "No delta table available.",
        "",
        "## Bootstrap Delta Checks",
        "",
        _df_to_markdown(bootstrap) if not bootstrap.empty else "No bootstrap table available.",
        "",
        "## Hyperparameter Tuning Check",
        "",
        _df_to_markdown(tuned_comparison) if not tuned_comparison.empty else "No tuned-model comparison available.",
        "",
        "## ESG2Risk-Style Quintile Analysis",
        "",
        _df_to_markdown(quintiles) if not quintiles.empty else "No quintile table available.",
        "",
        "## Interpretation Notes",
        "",
        "- The modeling panel is restricted to the news-coverage era instead of using decades of price-only history.",
        "- Metrics are reported both on all test weeks and on ESG/news weeks only.",
        "- News-only and price-plus-news ablations are included to make the incremental value of text features clearer.",
        "- The pipeline uses QQQ as the benchmark because it is available in FNSPID price history.",
        "- Transformer scoring was run on ESG keyword-filtered articles, not the full 23 GB news file.",
        "- The project remains academic and data-mining focused; dashboard work is optional future work.",
    ]
    output_path("reports", "results_summary.md").write_text("\n".join(summary), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("setup").set_defaults(func=command_setup)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--nrows", type=int, default=1_000)
    inspect_parser.add_argument("--scan-news", action="store_true")
    inspect_parser.set_defaults(func=command_inspect)

    filter_parser = subparsers.add_parser("filter-news")
    filter_parser.add_argument("--chunksize", type=int, default=config.NEWS_CHUNK_SIZE)
    filter_parser.add_argument("--start-date", default=config.START_DATE)
    filter_parser.add_argument("--end-date", default=config.END_DATE)
    filter_parser.add_argument("--partitioned", action="store_true")
    filter_parser.add_argument("--tickers", nargs="*")
    filter_parser.set_defaults(func=command_filter_news)

    subparsers.add_parser("prices-labels").set_defaults(func=command_prices_labels)
    subparsers.add_parser("features").set_defaults(func=command_features)

    finbert_parser = subparsers.add_parser("finbert")
    finbert_parser.add_argument("--batch-size", type=int, default=config.TRANSFORMER_BATCH_SIZE)
    finbert_parser.add_argument("--device", type=int, default=-1)
    finbert_parser.add_argument("--model-name", default=config.FINBERT_MODEL_NAME)
    finbert_parser.add_argument("--all-news", action="store_true")
    finbert_parser.add_argument("--max-articles", type=int)
    finbert_parser.set_defaults(func=command_finbert)

    climatebert_parser = subparsers.add_parser("climatebert")
    climatebert_parser.add_argument("--batch-size", type=int, default=config.TRANSFORMER_BATCH_SIZE)
    climatebert_parser.add_argument("--device", type=int, default=-1)
    climatebert_parser.add_argument("--model-name", default=config.CLIMATEBERT_MODEL_NAME)
    climatebert_parser.add_argument("--all-news", action="store_true")
    climatebert_parser.add_argument("--max-articles", type=int)
    climatebert_parser.set_defaults(func=command_climatebert)

    subparsers.add_parser("panel").set_defaults(func=command_panel)
    subparsers.add_parser("model-eval").set_defaults(func=command_model_eval)

    tuned_parser = subparsers.add_parser("tuned-model-eval")
    tuned_parser.add_argument("--groups", nargs="*")
    tuned_parser.set_defaults(func=command_tuned_model_eval)

    subparsers.add_parser("report").set_defaults(func=command_report)
    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

