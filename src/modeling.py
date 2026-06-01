"""Time-aware modeling utilities for regression and classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


@dataclass(frozen=True)
class TimeSplit:
    train: pd.Series
    validation: pd.Series
    test: pd.Series


def get_feature_groups(panel: pd.DataFrame) -> dict[str, list[str]]:
    excluded_prefixes = ("future_", "high_", "benchmark_ticker")
    excluded = {"ticker", "week_end_date", "finbert_label", "climate_label", "has_esg_news_week"}
    numeric_cols = [
        col
        for col in panel.select_dtypes(include=[np.number, "boolean"]).columns
        if col not in excluded and not col.startswith(excluded_prefixes)
    ]
    price_prefixes = (
        "trailing_",
        "rolling_",
        "close_",
        "distance_",
        "volume_",
        "benchmark_",
        "excess_",
    )
    keyword_prefixes = (
        "article_count",
        "esg_",
        "env_",
        "social_",
        "gov_",
        "controversy_",
    )
    keyword_terms = ("keyword", "is_esg", "is_controversy")
    climate_prefixes = (
        "climate_",
        "avg_climate",
        "max_climate",
        "non_climate",
    )
    groups = {
        "price": [col for col in numeric_cols if col.startswith(price_prefixes)],
        "keyword": [
            col
            for col in numeric_cols
            if col.startswith(keyword_prefixes) or any(term in col for term in keyword_terms)
        ],
        "finbert": [col for col in numeric_cols if col.startswith("finbert_")],
        "climatebert": [col for col in numeric_cols if col.startswith(climate_prefixes) or "climate_" in col],
        "embedding": [col for col in numeric_cols if col.startswith(("embed_", "svd_"))],
    }
    groups["news"] = sorted(set(groups["keyword"] + groups["finbert"] + groups["climatebert"]))
    groups["all_news"] = sorted(set(groups["news"] + groups["embedding"]))
    groups["all"] = sorted(set(groups["price"] + groups["all_news"]))
    groups["price_keyword"] = sorted(set(groups["price"] + groups["keyword"]))
    groups["price_finbert"] = sorted(set(groups["price"] + groups["finbert"]))
    groups["price_climatebert"] = sorted(set(groups["price"] + groups["climatebert"]))
    groups["price_finbert_climatebert"] = sorted(set(groups["price"] + groups["finbert"] + groups["climatebert"]))
    groups["price_all_news"] = sorted(set(groups["price"] + groups["all_news"]))
    return groups


def make_time_split(
    panel: pd.DataFrame,
    date_col: str = "week_end_date",
    train_start: str = config.MODEL_START_DATE,
    train_end: str = config.TRAIN_END_DATE,
    validation_start: str = config.VALIDATION_START_DATE,
    validation_end: str = config.VALIDATION_END_DATE,
    test_start: str = config.TEST_START_DATE,
) -> TimeSplit:
    dates = pd.to_datetime(panel[date_col])
    return TimeSplit(
        train=(dates >= pd.Timestamp(train_start)) & (dates <= pd.Timestamp(train_end)),
        validation=(dates >= pd.Timestamp(validation_start)) & (dates <= pd.Timestamp(validation_end)),
        test=dates >= pd.Timestamp(test_start),
    )


def build_preprocessor(feature_cols: list[str], scale: bool = False) -> ColumnTransformer:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(steps)
    return ColumnTransformer([("numeric", numeric_pipe, feature_cols)], remainder="drop")


def _regression_estimators(random_state: int = config.RANDOM_SEED) -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=random_state, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=random_state),
    }


def _classification_estimators(random_state: int = config.RANDOM_SEED) -> dict[str, object]:
    return {
        "logistic": LogisticRegression(max_iter=1_000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=random_state, n_jobs=-1, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_state),
    }


def _regression_param_grids() -> dict[str, list[dict]]:
    return {
        "ridge": [{"alpha": [0.1, 1.0, 10.0, 50.0]}],
        "random_forest": [
            {
                "n_estimators": [300],
                "min_samples_leaf": [2, 5, 10],
                "max_features": ["sqrt", 0.7, 1.0],
            }
        ],
        "gradient_boosting": [
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3],
            }
        ],
        "hist_gradient_boosting": [
            {
                "max_iter": [100, 200],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_leaf_nodes": [15, 31],
                "l2_regularization": [0.0, 0.1],
            }
        ],
    }


def _classification_param_grids() -> dict[str, list[dict]]:
    return {
        "logistic": [{"C": [0.1, 1.0, 10.0]}],
        "random_forest": [
            {
                "n_estimators": [300],
                "min_samples_leaf": [2, 5, 10],
                "max_features": ["sqrt", 0.7, 1.0],
            }
        ],
        "gradient_boosting": [
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3],
            }
        ],
        "hist_gradient_boosting": [
            {
                "max_iter": [100, 200],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_leaf_nodes": [15, 31],
                "l2_regularization": [0.0, 0.1],
            }
        ],
    }


def _fit_models(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    split: TimeSplit,
    task: Literal["regression", "classification"],
) -> dict[str, Pipeline]:
    data = panel.loc[split.train, feature_cols + [target_col]].dropna(subset=[target_col])
    X = data[feature_cols]
    y = data[target_col].astype(int) if task == "classification" else data[target_col]
    estimators = _classification_estimators() if task == "classification" else _regression_estimators()
    fitted = {}
    for name, estimator in estimators.items():
        scale = name in {"ridge", "logistic"}
        model = Pipeline(
            [
                ("preprocessor", build_preprocessor(feature_cols, scale=scale)),
                ("model", estimator),
            ]
        )
        model.fit(X, y)
        fitted[name] = model
    return fitted


def _make_pipeline(name: str, estimator: object, feature_cols: list[str]) -> Pipeline:
    scale = name in {"ridge", "logistic"}
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(feature_cols, scale=scale)),
            ("model", estimator),
        ]
    )


def tune_models_on_validation(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    split: TimeSplit,
    task: Literal["regression", "classification"],
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Tune hyperparameters on validation, then refit winners on train+validation."""
    estimators = _classification_estimators() if task == "classification" else _regression_estimators()
    grids = _classification_param_grids() if task == "classification" else _regression_param_grids()

    train = panel.loc[split.train, feature_cols + [target_col]].dropna(subset=[target_col])
    validation = panel.loc[split.validation, feature_cols + [target_col]].dropna(subset=[target_col])
    refit = panel.loc[split.train | split.validation, feature_cols + [target_col]].dropna(subset=[target_col])
    if train.empty or validation.empty or refit.empty:
        return {}, pd.DataFrame()

    X_train = train[feature_cols]
    y_train = train[target_col].astype(int) if task == "classification" else train[target_col]
    X_val = validation[feature_cols]
    y_val = validation[target_col].astype(int) if task == "classification" else validation[target_col]
    X_refit = refit[feature_cols]
    y_refit = refit[target_col].astype(int) if task == "classification" else refit[target_col]

    fitted: dict[str, Pipeline] = {}
    records = []
    for name, estimator in estimators.items():
        best_score = np.inf if task == "regression" else -np.inf
        best_params: dict | None = None
        for params in ParameterGrid(grids.get(name, [{}])):
            candidate = _make_pipeline(name, clone(estimator), feature_cols)
            candidate.set_params(**{f"model__{key}": value for key, value in params.items()})
            candidate.fit(X_train, y_train)
            if task == "regression":
                pred = candidate.predict(X_val)
                score = float(np.sqrt(mean_squared_error(y_val, pred)))
                is_better = score < best_score
            else:
                if hasattr(candidate.named_steps["model"], "predict_proba"):
                    pred = candidate.predict_proba(X_val)[:, 1]
                else:
                    pred = candidate.predict(X_val)
                score = float(roc_auc_score(y_val, pred)) if y_val.nunique() > 1 else np.nan
                is_better = pd.notna(score) and score > best_score
            records.append(
                {
                    "task": task,
                    "model": name,
                    "target": target_col,
                    "validation_metric": "rmse" if task == "regression" else "roc_auc",
                    "validation_score": score,
                    "params": repr(params),
                }
            )
            if is_better:
                best_score = score
                best_params = params
        if best_params is None:
            continue
        final_model = _make_pipeline(name, clone(estimator), feature_cols)
        final_model.set_params(**{f"model__{key}": value for key, value in best_params.items()})
        final_model.fit(X_refit, y_refit)
        fitted[name] = final_model
    return fitted, pd.DataFrame(records)


def train_regression_models(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = config.PRIMARY_REGRESSION_TARGET,
    split: TimeSplit | None = None,
) -> dict[str, Pipeline]:
    split = split or make_time_split(panel)
    return _fit_models(panel, feature_cols, target_col, split, task="regression")


def train_classification_models(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = config.PRIMARY_CLASSIFICATION_TARGET,
    split: TimeSplit | None = None,
) -> dict[str, Pipeline]:
    split = split or make_time_split(panel)
    return _fit_models(panel, feature_cols, target_col, split, task="classification")


def predict_model(model: Pipeline, panel: pd.DataFrame, feature_cols: list[str], task: str) -> np.ndarray:
    X = panel[feature_cols]
    if task == "classification" and hasattr(model.named_steps["model"], "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)

