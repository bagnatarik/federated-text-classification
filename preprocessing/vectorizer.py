"""
Global TF-IDF vectorizer (fit once, reuse everywhere).

What this module provides:
- fit_and_save_vectorizer: fit a TfidfVectorizer on cleaned text and persist it.
- load_vectorizer / save_vectorizer: IO helpers around joblib.
- transform_texts: transform a list/Series of texts to a (sparse) TF-IDF matrix.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def fit_tfidf(
    texts: Iterable[str],
    *,
    min_df: int | float = 2,
    max_df: int | float = 0.9,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int | None = 50000,
    lowercase: bool = False,  # text is already lowercased during cleaning
    norm: str = "l2",
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Fit a global TfidfVectorizer with sensible defaults for spam classification.
    """
    vec = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=lowercase,
        norm=norm,
        sublinear_tf=sublinear_tf,
    )
    vec.fit(texts)
    return vec

def transform_texts(vec: TfidfVectorizer, texts: Iterable[str]):
    """
    Transform texts to a sparse TF-IDF matrix using an already-fitted vectorizer.
    """
    return vec.transform(texts)

def save_vectorizer(vec: TfidfVectorizer, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, path)
    print(f"Vectorizer saved to {path}.", end="\n\n")

def load_vectorizer(path: str) -> TfidfVectorizer:
    vec = joblib.load(path)
    print(f"Vectorizer loaded from {path}.", end="\n\n")
    return vec

def fit_and_save_vectorizer(
    input_csv: str,
    output_pkl: str,
    text_col: str = "text",
    min_df: int | float = 2,
    max_df: int | float = 0.9,
    ngram: Tuple[int, int] = (1, 2),
    max_features: int | None = 50000,
) -> None:
    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not in {input_csv}")
    texts = df[text_col].astype("string").fillna("").tolist()

    vec = fit_tfidf(
        texts,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram,
        max_features=max_features,
        lowercase=False,
    )
    save_vectorizer(vec, output_pkl)

# === Main script ===

def run_vectorizer(input_csv: str, output_pkl: str, text_col: str = "text", min_df: int | float = 2, max_df: int | float = 0.9, ngram: Tuple[int, int] = (1, 2), max_features: int | None = 50000) -> None:
    fit_and_save_vectorizer(
        input_csv=input_csv,
        output_pkl=output_pkl,
        text_col=text_col,
        min_df=min_df,
        max_df=max_df,
        ngram=(ngram[0], ngram[1]),
        max_features=max_features if max_features > 0 else None,
    )