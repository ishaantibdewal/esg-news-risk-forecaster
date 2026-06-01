"""Small Streamlit demo for browsing generated ESG risk predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"


@st.cache_data
def load_predictions() -> pd.DataFrame:
    tuned_path = TABLES_DIR / "tuned_test_predictions.csv"
    base_path = TABLES_DIR / "test_predictions.csv"
    path = tuned_path if tuned_path.exists() else base_path
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path, parse_dates=["week_end_date"])
    prediction_cols = [col for col in data.columns if col.startswith(("tuned_reg__", "tuned_clf__", "reg__", "clf__"))]
    keep = [
        "ticker",
        "week_end_date",
        "has_esg_news_week",
        "future_vol_21d",
        "high_vol_21d",
        "future_return_21d",
        *prediction_cols,
    ]
    return data[[col for col in keep if col in data.columns]]


@st.cache_data
def load_metrics() -> pd.DataFrame:
    path = TABLES_DIR / "tuned_model_metrics.csv"
    if not path.exists():
        path = TABLES_DIR / "model_metrics.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def label_for_prediction_col(col: str) -> str:
    parts = col.split("__")
    if len(parts) != 3:
        return col
    task, feature_group, model = parts
    task_name = "Volatility" if "reg" in task else "High-volatility probability"
    return f"{task_name}: {feature_group} / {model}"


def main() -> None:
    st.set_page_config(page_title="ESG News Risk Forecaster", layout="wide")
    st.title("ESG News Risk Forecaster")
    st.write(
        "Browse generated 2023 test predictions from the DSC 148 ESG-news volatility pipeline. "
        "The demo uses saved pipeline outputs, so it runs without re-scoring transformers."
    )

    predictions = load_predictions()
    metrics = load_metrics()
    if predictions.empty:
        st.error("No prediction table found. Run the pipeline through model evaluation first.")
        return

    pred_cols = [col for col in predictions.columns if col.startswith(("tuned_reg__", "tuned_clf__", "reg__", "clf__"))]
    scored = predictions.dropna(subset=pred_cols, how="all").copy()
    if scored.empty:
        st.error("Prediction columns exist, but no scored test rows were found.")
        return

    left, right = st.columns([1, 2])
    with left:
        ticker = st.selectbox("Ticker", sorted(scored["ticker"].dropna().unique()))
        ticker_rows = scored.loc[scored["ticker"] == ticker].sort_values("week_end_date")
        week = st.selectbox(
            "Week end date",
            ticker_rows["week_end_date"].dt.strftime("%Y-%m-%d").tolist(),
            index=len(ticker_rows) - 1,
        )
        selected_col = st.selectbox(
            "Prediction to inspect",
            pred_cols,
            format_func=label_for_prediction_col,
        )

    row = ticker_rows.loc[ticker_rows["week_end_date"].dt.strftime("%Y-%m-%d") == week].iloc[0]
    prediction = row.get(selected_col)
    is_classifier = "__clf__" in selected_col or selected_col.startswith("tuned_clf__") or selected_col.startswith("clf__")

    with right:
        st.subheader(f"{ticker} risk snapshot for {week}")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Predicted score", f"{prediction:.4f}" if pd.notna(prediction) else "n/a")
        metric_cols[1].metric("Actual 21d vol", f"{row['future_vol_21d']:.4f}")
        metric_cols[2].metric("Actual high-vol label", int(row["high_vol_21d"]))
        metric_cols[3].metric("ESG/news week", "yes" if bool(row["has_esg_news_week"]) else "no")
        if is_classifier and pd.notna(prediction):
            st.info("Scores above 0.50 are interpreted as high-volatility predictions in the saved metrics.")

    st.subheader("Same-week peer comparison")
    same_week = scored.loc[scored["week_end_date"].dt.strftime("%Y-%m-%d") == week].copy()
    same_week["selected_prediction"] = same_week[selected_col]
    st.dataframe(
        same_week[
            [
                "ticker",
                "has_esg_news_week",
                "selected_prediction",
                "future_vol_21d",
                "high_vol_21d",
                "future_return_21d",
            ]
        ].sort_values("selected_prediction", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    if not metrics.empty:
        st.subheader("Model metrics")
        st.dataframe(metrics, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
