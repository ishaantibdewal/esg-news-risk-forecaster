"""ESG keyword dictionaries and weekly keyword feature aggregation."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


ENV_KEYWORDS = [
    "climate",
    "carbon",
    "emissions",
    "greenhouse gas",
    "renewable",
    "solar",
    "wind",
    "clean energy",
    "fossil fuel",
    "pollution",
    "waste",
    "water",
    "environmental",
    "sustainability",
    "biodiversity",
    "oil spill",
    "transition risk",
    "net zero",
    "methane",
    "deforestation",
    "wildfire",
    "drought",
    "flood",
    "toxic spill",
    "hazardous waste",
    "epa",
    "clean air act",
    "clean water act",
    "carbon offset",
    "carbon credit",
    "scope 1",
    "scope 2",
    "scope 3",
    "decarbonization",
    "electrification",
    "electric vehicle",
    "ev battery",
    "battery recycling",
    "energy transition",
    "climate disclosure",
    "climate regulation",
    "environmental protection agency",
    "emissions target",
    "carbon capture",
    "carbon tax",
    "renewable energy",
    "sustainable aviation fuel",
    "water scarcity",
    "wastewater",
    "recycling",
    "plastic waste",
    "chemical spill",
    "contamination",
    "toxic emissions",
]

SOCIAL_KEYWORDS = [
    "labor",
    "employee",
    "worker",
    "strike",
    "union",
    "workplace",
    "safety",
    "discrimination",
    "diversity",
    "inclusion",
    "privacy",
    "customer safety",
    "human rights",
    "community",
    "supply chain",
    "supplier",
    "health",
    "wage",
    "harassment",
    "workplace injury",
    "product safety",
    "data breach",
    "cybersecurity",
    "layoff",
    "layoffs",
    "child labor",
    "forced labor",
    "unionization",
    "osha",
    "wage theft",
    "employee lawsuit",
    "consumer protection",
    "sexual harassment",
    "workplace discrimination",
    "racial discrimination",
    "gender discrimination",
    "unsafe working conditions",
    "worker safety",
    "labor violation",
    "labor dispute",
    "collective bargaining",
    "minimum wage",
    "pay equity",
    "employee turnover",
    "human capital",
    "customer privacy",
    "data privacy",
    "consumer lawsuit",
    "product recall",
    "supply chain disruption",
    "supplier misconduct",
    "modern slavery",
    "human trafficking",
    "community impact",
]

GOV_KEYWORDS = [
    "board",
    "director",
    "ceo",
    "cfo",
    "executive compensation",
    "shareholder",
    "audit",
    "accounting",
    "fraud",
    "bribery",
    "corruption",
    "lawsuit",
    "settlement",
    "investigation",
    "regulatory",
    "antitrust",
    "compliance",
    "governance",
    "proxy",
    "sec investigation",
    "doj investigation",
    "insider trading",
    "accounting restatement",
    "material weakness",
    "board independence",
    "activist investor",
    "proxy fight",
    "shareholder proposal",
    "executive misconduct",
    "class action",
    "antitrust lawsuit",
    "securities fraud",
    "regulatory probe",
    "regulatory investigation",
    "compliance failure",
    "internal controls",
    "audit committee",
    "bribery investigation",
    "foreign corrupt practices act",
    "fcpa",
    "money laundering",
    "sanctions violation",
    "whistleblower complaint",
    "restatement",
    "governance failure",
    "board oversight",
    "shareholder rights",
    "executive pay",
    "ceo resignation",
    "cfo resignation",
    "management shakeup",
]

CONTROVERSY_KEYWORDS = [
    "lawsuit",
    "scandal",
    "fine",
    "penalty",
    "violation",
    "probe",
    "investigation",
    "recall",
    "fraud",
    "breach",
    "misconduct",
    "whistleblower",
    "sanctions",
    "corruption",
    "settlement",
    "class action",
    "data breach",
    "cyberattack",
    "hacking",
    "regulatory probe",
    "sec probe",
    "doj probe",
    "criminal investigation",
    "civil investigation",
    "antitrust investigation",
    "antitrust lawsuit",
    "environmental violation",
    "labor violation",
    "safety violation",
    "sanctions violation",
    "compliance failure",
    "accounting restatement",
    "material weakness",
    "insider trading",
    "executive misconduct",
    "sexual harassment",
    "toxic spill",
    "chemical spill",
    "product recall",
    "privacy violation",
    "consumer lawsuit",
    "employee lawsuit",
    "whistleblower complaint",
    "money laundering",
    "bribery investigation",
    "corruption probe",
    "fraud investigation",
    "regulatory fine",
    "settlement agreement",
]


def compile_keyword_patterns(keywords: Iterable[str]) -> list[re.Pattern[str]]:
    return [
        re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", flags=re.IGNORECASE)
        for keyword in keywords
    ]


def count_keywords(text: str, patterns: list[re.Pattern[str]]) -> int:
    if not isinstance(text, str) or not text:
        return 0
    return sum(len(pattern.findall(text)) for pattern in patterns)


def score_esg_keywords(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    out = df.copy()
    patterns = {
        "env_keyword_count": compile_keyword_patterns(ENV_KEYWORDS),
        "social_keyword_count": compile_keyword_patterns(SOCIAL_KEYWORDS),
        "gov_keyword_count": compile_keyword_patterns(GOV_KEYWORDS),
        "controversy_keyword_count": compile_keyword_patterns(CONTROVERSY_KEYWORDS),
    }
    for col, compiled in patterns.items():
        out[col] = out[text_col].fillna("").map(lambda text: count_keywords(text, compiled))
    out["esg_keyword_total"] = (
        out["env_keyword_count"] + out["social_keyword_count"] + out["gov_keyword_count"]
    )
    out["is_esg_keyword_article"] = out["esg_keyword_total"] > 0
    out["is_controversy_article"] = out["controversy_keyword_count"] > 0
    return out


def _week_end(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series)
    return dates + pd.to_timedelta(4 - dates.dt.weekday, unit="D")


def aggregate_keyword_features_weekly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["week_end_date"] = _week_end(out["date"])
    if "is_esg_keyword_article" not in out.columns:
        out = score_esg_keywords(out)
    out["is_env_article"] = out["env_keyword_count"] > 0
    out["is_social_article"] = out["social_keyword_count"] > 0
    out["is_gov_article"] = out["gov_keyword_count"] > 0
    grouped = out.groupby(["ticker", "week_end_date"])
    weekly = grouped.agg(
        article_count=("clean_text", "size"),
        esg_article_count=("is_esg_keyword_article", "sum"),
        controversy_article_count=("is_controversy_article", "sum"),
        env_article_count=("is_env_article", "sum"),
        social_article_count=("is_social_article", "sum"),
        gov_article_count=("is_gov_article", "sum"),
        env_keyword_count=("env_keyword_count", "sum"),
        social_keyword_count=("social_keyword_count", "sum"),
        gov_keyword_count=("gov_keyword_count", "sum"),
        controversy_keyword_count=("controversy_keyword_count", "sum"),
        esg_keyword_total=("esg_keyword_total", "sum"),
    ).reset_index()
    denom = weekly["article_count"].clip(lower=1)
    weekly["esg_article_share"] = weekly["esg_article_count"] / denom
    weekly["controversy_article_share"] = weekly["controversy_article_count"] / denom
    weekly["env_article_share"] = weekly["env_article_count"] / denom
    weekly["social_article_share"] = weekly["social_article_count"] / denom
    weekly["gov_article_share"] = weekly["gov_article_count"] / denom
    weekly["esg_news_intensity"] = weekly["esg_keyword_total"] / denom
    return add_lagged_keyword_features(weekly)


def add_lagged_keyword_features(weekly: pd.DataFrame, lags: tuple[int, ...] = (1, 4, 12)) -> pd.DataFrame:
    out = weekly.sort_values(["ticker", "week_end_date"]).copy()
    lag_cols = [
        "article_count",
        "esg_article_count",
        "esg_article_share",
        "controversy_article_share",
        "env_article_share",
        "social_article_share",
        "gov_article_share",
        "esg_news_intensity",
    ]
    for lag in lags:
        for col in lag_cols:
            out[f"{col}_lag{lag}w"] = out.groupby("ticker")[col].shift(lag)
    return out

