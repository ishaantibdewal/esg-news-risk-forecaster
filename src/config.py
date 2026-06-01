"""Project configuration for the ESG2Risk-inspired volatility pipeline."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FNSPID_DIR = RAW_DATA_DIR / "fnspid"
NEWS_CSV_PATH = FNSPID_DIR / "nasdaq_external_data.csv"
PRICE_HISTORY_DIR = FNSPID_DIR / "full_history"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"

DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "JPM",
    "BAC",
    "XOM",
    "CVX",
    "COST",
    "WMT",
    "UNH",
    "JNJ",
    "HD",
    "DIS",
    "NFLX",
    "AMD",
    "INTC",
]

TICKER_ALIASES = {
    "META": ["META", "FB"],
    "GOOGL": ["GOOGL", "GOOG"],
}

BENCHMARK_PREFERENCE = ["QQQ", "SPY"]
HORIZONS = [5, 10, 21, 63]
PRIMARY_REGRESSION_TARGET = "future_vol_21d"
REPRODUCTION_REGRESSION_TARGET = "future_vol_10d"
PRIMARY_CLASSIFICATION_TARGET = "high_vol_21d"

PANEL_FREQUENCY = "W-FRI"
HIGH_VOL_QUANTILE = 0.75
TOP_RISK_FRACTION = 0.10

START_DATE = None
END_DATE = "2023-12-31"
MODEL_START_DATE = "2016-01-01"
TRAIN_END_DATE = "2021-12-31"
VALIDATION_START_DATE = "2022-01-01"
VALIDATION_END_DATE = "2022-12-31"
TEST_START_DATE = "2023-01-01"
NEWS_WEEK_ARTICLE_MIN = 1
BOOTSTRAP_ITERATIONS = 500

NEWS_CHUNK_SIZE = 100_000
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_MAX_LENGTH = 256
RANDOM_SEED = 148

FINBERT_MODEL_NAME = "ProsusAI/finbert"
CLIMATEBERT_MODEL_NAME = "climatebert/distilroberta-base-climate-detector"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

NEWS_TEXT_COLUMNS = [
    "Article_title",
    "Textrank_summary",
    "Lexrank_summary",
]

NEWS_USE_COLUMNS = [
    "Date",
    "Article_title",
    "Stock_symbol",
    "Url",
    "Publisher",
    "Author",
    "Lsa_summary",
    "Luhn_summary",
    "Textrank_summary",
    "Lexrank_summary",
]

FULL_ARTICLE_COLUMN = "Article"

