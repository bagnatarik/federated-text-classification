"""
Deterministic text cleaning for the Enron Spam dataset.

What it does:
1) Build the training text as: text_final = [subject + "\n\n"] + (message or text)
2) Normalize the text:
   - Unicode NFKC, lowercase, strip
   - Remove URLs, emails, HTML tags
   - Remove numbers (optional), punctuation
   - Collapse multiple whitespaces/newlines
   - Remove English stopwords (scikit-learn list)
   - (Optional) Lemmatization if spaCy is available and --lemmatize is passed
3) Drop rows with missing/empty text after cleaning
4) Drop exact duplicate texts (keep first)
5) Save a MINIMAL training CSV: columns = [message_id, text, label]
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# === REGULAR EXPRESSIONS ===

RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_NUM = re.compile(r"\b\d+\b")
RE_PUNCT = re.compile(r"[^\w\s]")  # punctuation (keeps alnum and underscore)
RE_MULTI_SPACE = re.compile(r"\s+")
RE_LINEBREAKS = re.compile(r"(?:\r\n|\r|\n)+")

STOPWORDS = set(ENGLISH_STOP_WORDS)

# === FUNCTIONS ===

def _normalize_unicode(text: str) -> str:
    """Normalize unicode to NFKC and unescape HTML entities."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    return text

def _clean_text_core(
    s: str,
    *,
    lowercase: bool = True,
    remove_numbers: bool = False,
    remove_stopwords: bool = True,
    lemmatize: bool = False,
    _nlp=None,
) -> str:
    """Apply core cleaning steps to a single string."""
    if not s:
        return ""

    # Unicode & basic normalization
    s = _normalize_unicode(s)

    # Lowercasing
    if lowercase:
        s = s.lower()

    # Strip HTML tags early
    s = RE_HTML_TAG.sub(" ", s)

    # Remove URLs and emails
    s = RE_URL.sub(" ", s)
    s = RE_EMAIL.sub(" ", s)

    # Optional: remove stand-alone numbers
    if remove_numbers:
        s = RE_NUM.sub(" ", s)

    # Remove punctuation
    s = RE_PUNCT.sub(" ", s)

    # Collapse line breaks and spaces
    s = RE_LINEBREAKS.sub("\n", s)         # keep single "\n" for paragraph boundary
    s = RE_MULTI_SPACE.sub(" ", s)         # collapse spaces
    s = s.strip()

    # Stopwords removal (token-level)
    if remove_stopwords:
        tokens = [tok for tok in s.split() if tok not in STOPWORDS]
        s = " ".join(tokens)

    # Optional: lemmatization via spaCy small English model (if provided)
    if lemmatize and _nlp is not None:
        # spaCy prefers sentences; we pass the whole text
        doc = _nlp(s)
        lemmas = [t.lemma_ for t in doc if t.lemma_ not in (" ", "", "\n", "\t")]
        s = " ".join(lemmas).strip()

    return s

def _choose_body(row: pd.Series, prefer_message: bool = True) -> str:
    """Choose the body field between 'message' and 'text'."""

    if prefer_message and "message" in row:
        msg = row["message"]
        if not pd.isna(msg) and str(msg).strip():
            return str(msg)
    
    if "text" in row:
        txt = row["text"]
        return "" if pd.isna(txt) else str(txt)
    
    return ""


def build_raw_text(row: pd.Series, keep_subject: bool = True, prefer_message: bool = True) -> str:
    """
    Build the raw text for cleaning:
      text_raw = [subject + "\n\n"] + (message or text)
    """

    subject_value = row.get("subject", "")
    subject = "" if pd.isna(subject_value) else str(subject_value)
    
    body = _choose_body(row, prefer_message=prefer_message)
    
    if keep_subject and subject.strip():
        return f"{subject}\n\n{body}"
    return body

def clean_dataframe(
    df: pd.DataFrame,
    *,
    keep_subject: bool = True,
    prefer_message: bool = True,
    drop_duplicates: bool = True,
    min_tokens: int = 1,
    lowercase: bool = True,
    remove_numbers: bool = False,
    remove_stopwords: bool = True,
    lemmatize: bool = False,
) -> pd.DataFrame:
    """
    Clean the dataframe and return a MINIMAL training view with columns:
    [message_id, text, label].

    Steps:
      - Build 'text_raw' from subject + body
      - Clean 'text_raw' → 'text'
      - Drop empty/too-short texts
      - Drop exact duplicate texts
      - Keep only [message_id, text, label]
    """
    # Optional spaCy loading (only if lemmatize=True)
    nlp = None
    if lemmatize:
        try:
            import spacy  # type: ignore
            try:
                nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
            except OSError:
                # Model not available; warn and skip lemmatization
                print(
                    "[WARN] spaCy model 'en_core_web_sm' not found; skipping lemmatization.",
                    file=sys.stderr,
                )
                lemmatize = False
        except Exception as e:
            print(f"[WARN] spaCy unavailable ({e}); skipping lemmatization.", file=sys.stderr)
            lemmatize = False

    # Ensure required columns exist
    for col in ("message_id", "label"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from input CSV.")

    # Build raw text for cleaning
    df = df.copy()
    df["text_raw"] = df.apply(
        lambda r: build_raw_text(r, keep_subject=keep_subject, prefer_message=prefer_message), axis=1
    )

    # Apply cleaning core
    df["text"] = df["text_raw"].apply(
        lambda x: _clean_text_core(
            x,
            lowercase=lowercase,
            remove_numbers=remove_numbers,
            remove_stopwords=remove_stopwords,
            lemmatize=lemmatize,
            _nlp=nlp,
        )
    )

    # Drop rows with empty text or too short (#tokens < min_tokens)
    tokens_len = df["text"].str.split().apply(len)
    before = len(df)
    df = df.loc[tokens_len >= int(min_tokens)].copy()
    removed_short = before - len(df)

    # Drop exact duplicates on cleaned text
    removed_dupes = 0
    if drop_duplicates:
        before2 = len(df)
        df = df.drop_duplicates(subset=["text"], keep="first").copy()
        removed_dupes = before2 - len(df)

    # Keep only the minimal training columns
    out = df[["message_id", "text", "label"]].copy()

    # Report
    print(
        f"Cleaning summary : kept={len(out)} | removed_short={removed_short} | removed_dupes={removed_dupes}"
    )

    return out

def load_csv(input_csv: str) -> pd.DataFrame:
    """Load CSV and ensure basic dtypes for robustness."""
    df = pd.read_csv(input_csv)
    # Basic dtype coercions
    if "message_id" in df.columns:
        df["message_id"] = df["message_id"].astype("string")
    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
        # If any missing labels, drop them (cannot train without label)
        before = len(df)
        df = df.dropna(subset=["label"]).copy()
        if len(df) < before:
            print(f"[INFO] Dropped {before - len(df)} rows with missing label.")
        df["label"] = df["label"].astype("int8")
    # Optional text columns to string
    for c in ("text", "label_text", "subject", "message", "date"):
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df

def save_csv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Clean data saved to {path}.", end="\n\n")

# === Main function ===

def run_cleaning(input_csv: str, output_csv: str):
    dataframe = load_csv(input_csv)

    cleaned = clean_dataframe(dataframe)

    save_csv(cleaned, output_csv)
