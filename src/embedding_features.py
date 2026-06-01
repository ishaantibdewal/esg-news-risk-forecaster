"""Optional transformer/sentence embedding features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from . import config
from .text_cleaning import truncate_texts_for_transformer


def load_embedding_model(model_name: str = config.EMBEDDING_MODEL_NAME):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_text_batch(model, texts: list[str]) -> np.ndarray:
    return np.asarray(model.encode(texts, show_progress_bar=False))


def score_article_embeddings(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    batch_size: int = 64,
    model_name: str = config.EMBEDDING_MODEL_NAME,
) -> tuple[pd.DataFrame, np.ndarray]:
    model = load_embedding_model(model_name)
    texts = truncate_texts_for_transformer(df[text_col].fillna(""))
    vectors = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        vectors.append(embed_text_batch(model, texts[start : start + batch_size]))
    embeddings = np.vstack(vectors) if vectors else np.empty((0, 0))
    return df.reset_index(drop=True), embeddings


def tfidf_svd_article_features(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    n_components: int = 50,
    max_features: int = 20_000,
) -> tuple[pd.DataFrame, np.ndarray, TfidfVectorizer, TruncatedSVD]:
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=2, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(df[text_col].fillna(""))
    reducer = TruncatedSVD(n_components=n_components, random_state=config.RANDOM_SEED)
    features = reducer.fit_transform(matrix)
    return df.reset_index(drop=True), features, vectorizer, reducer


def reduce_embeddings(embeddings: np.ndarray, n_components: int = 50) -> tuple[np.ndarray, TruncatedSVD]:
    reducer = TruncatedSVD(n_components=n_components, random_state=config.RANDOM_SEED)
    return reducer.fit_transform(embeddings), reducer


def aggregate_embeddings_weekly(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    prefix: str = "embed",
) -> pd.DataFrame:
    if len(df) != len(embeddings):
        raise ValueError("DataFrame and embeddings must have matching row counts.")
    out = df[["ticker", "date"]].copy()
    dates = pd.to_datetime(out["date"])
    out["week_end_date"] = dates + pd.to_timedelta(4 - dates.dt.weekday, unit="D")
    embed_cols = [f"{prefix}_{idx:02d}" for idx in range(embeddings.shape[1])]
    embed_df = pd.DataFrame(embeddings, columns=embed_cols)
    out = pd.concat([out.reset_index(drop=True), embed_df], axis=1)
    return out.groupby(["ticker", "week_end_date"], as_index=False)[embed_cols].mean()

