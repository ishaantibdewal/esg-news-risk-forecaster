"""Path helpers used throughout the project."""

from __future__ import annotations

from pathlib import Path

from . import config


def get_project_root() -> Path:
    """Return the project root directory."""
    return config.PROJECT_ROOT


def ensure_directories() -> None:
    """Create non-raw project output directories if they do not exist."""
    for path in [
        config.INTERIM_DIR,
        config.PROCESSED_DIR,
        config.OUTPUTS_DIR,
        config.FIGURES_DIR,
        config.TABLES_DIR,
        config.MODELS_DIR,
        config.REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def raw_fnspid_path() -> Path:
    return config.FNSPID_DIR


def news_csv_path() -> Path:
    return config.NEWS_CSV_PATH


def price_history_dir() -> Path:
    return config.PRICE_HISTORY_DIR


def interim_path(name: str) -> Path:
    return config.INTERIM_DIR / name


def processed_path(name: str) -> Path:
    return config.PROCESSED_DIR / name


def output_path(kind: str, name: str) -> Path:
    roots = {
        "figures": config.FIGURES_DIR,
        "tables": config.TABLES_DIR,
        "models": config.MODELS_DIR,
        "reports": config.REPORTS_DIR,
    }
    if kind not in roots:
        raise ValueError(f"Unknown output kind: {kind}")
    return roots[kind] / name

