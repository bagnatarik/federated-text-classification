"""
Quality Assessment (QA) before cleaning for the Enron Spam dataset.

This script reads a CSV (raw or train view), enforces/validates dtypes, and computes:
- Basic overview (shape, columns, dtypes)
- Missing values per column
- Duplicates (message_id duplicates, exact text duplicates)
- Class balance (label numeric + label_text if present)
- Text length statistics (chars & tokens) for text-like columns
- Token/noise estimates (URL, email, non-alphanumeric ratios)
- Date diagnostics if a normalized 'date_iso' column is present:
  * count non-null, min/max, yearly counts
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

EXPECTED_TEXT_COLUMNS = ["text", "subject", "message"]
LABEL_COL = "label"
LABEL_TEXT_COL = "label_text"
MESSAGE_ID_COL = "message_id"
DATE_ISO_COL = "date_iso"

RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
RE_NON_ALNUM = re.compile(r"[^A-Za-z0-9\s]+")


def format_csv_file_type(path: str) -> pd.DataFrame:
    """
        Load CSV to DataFrame with minimal dtype normalization for robustness.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with normalized dtypes.
    """
    # Load the CSV file
    df = pd.read_csv(path)
    # Soft coercions if columns exist
    if MESSAGE_ID_COL in df.columns:
        df[MESSAGE_ID_COL] = df[MESSAGE_ID_COL].astype("string")

    if LABEL_COL in df.columns:
        # stays numeric for ML; int8 is enough
        try:
            df[LABEL_COL] = df[LABEL_COL].astype("int8")
        except Exception:
            # if weird types, coerce via to_numeric
            df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").astype("Int64")

    if LABEL_TEXT_COL in df.columns:
        df[LABEL_TEXT_COL] = df[LABEL_TEXT_COL].astype("string")

    for c in EXPECTED_TEXT_COLUMNS:
        if c in df.columns:
            df[c] = df[c].astype("string")

    if "date" in df.columns and DATE_ISO_COL not in df.columns:
        # optional: leave raw 'date' as string, users may parse later
        df["date"] = df["date"].astype("string")

    if DATE_ISO_COL in df.columns:
        # ensure proper datetime dtype if present
        df[DATE_ISO_COL] = pd.to_datetime(df[DATE_ISO_COL], errors="coerce")

    return df

def choose_body_series(df: pd.DataFrame, prefer_message: bool = True) -> pd.Series:
    """
    Choose a body text series for noise/length stats:
    - Prefer 'message' when available and non-empty.
    - Fallback to 'text'.
    """
    msg = df["message"] if "message" in df.columns else pd.Series([], dtype="string")
    txt = df["text"] if "text" in df.columns else pd.Series([], dtype="string")

    if "message" in df.columns:
        # Use message if not null/empty, else fallback to text
        cond = msg.fillna("").str.len() > 0
        return msg.where(cond, txt).astype("string")
    elif "text" in df.columns:
        return txt.astype("string")
    else:
        # No suitable column → return empty series to avoid crashes
        return pd.Series([""], index=df.index, dtype="string")

# === QA metrics ===

def compute_basic_overview(df: pd.DataFrame) -> Dict[str, Any]:
    dtypes = {c: str(dt) for c, dt in df.dtypes.items()}
    return {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": dtypes,
    }


def compute_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    total = len(df)
    miss = df.isna().sum().to_dict()
    miss_ratio = {k: (float(v) / total if total > 0 else 0.0) for k, v in miss.items()}
    return {"count": miss, "ratio": miss_ratio}


def compute_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    # message_id dupes
    if MESSAGE_ID_COL in df.columns:
        dup_count = int(df[MESSAGE_ID_COL].duplicated(keep=False).sum())
        res["duplicate_message_id_rows"] = dup_count
    # exact text dupes (on chosen body and also on 'text' if present)
    if "text" in df.columns:
        res["duplicate_text_rows_exact"] = int(df["text"].duplicated(keep=False).sum())
    body = choose_body_series(df)
    res["duplicate_body_rows_exact"] = int(body.duplicated(keep=False).sum())
    return res

def compute_class_balance(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if LABEL_COL in df.columns:
        vals = df[LABEL_COL].dropna().astype("int64")
        cnt = Counter(vals.tolist())
        total = sum(cnt.values())
        ratio = {str(k): (v / total if total > 0 else 0.0) for k, v in cnt.items()}
        out["label_numeric"] = {"count": {str(k): int(v) for k, v in cnt.items()}, "ratio": ratio}

    if LABEL_TEXT_COL in df.columns:
        vals2 = df[LABEL_TEXT_COL].dropna().astype("string")
        cnt2 = Counter(vals2.tolist())
        total2 = sum(cnt2.values())
        ratio2 = {str(k): (v / total2 if total2 > 0 else 0.0) for k, v in cnt2.items()}
        out["label_text"] = {"count": {str(k): int(v) for k, v in cnt2.items()}, "ratio": ratio2}

    # Quick consistency check if both exist
    if LABEL_COL in df.columns and LABEL_TEXT_COL in df.columns:
        # Expected mapping: 0->ham, 1->spam
        # We'll just report mismatches proportion if any
        mapping = {0: "ham", 1: "spam"}
        mask_both = df[LABEL_COL].notna() & df[LABEL_TEXT_COL].notna()
        sub = df.loc[mask_both, [LABEL_COL, LABEL_TEXT_COL]]
        mismatches = int((sub[LABEL_TEXT_COL].str.lower() != sub[LABEL_COL].map(mapping).str.lower()).sum())
        out["consistency_numeric_vs_text"] = {
            "rows_compared": int(mask_both.sum()),
            "mismatches": mismatches,
            "mismatch_ratio": (mismatches / int(mask_both.sum())) if int(mask_both.sum()) > 0 else 0.0,
        }
    return out

def _length_stats_from_series(s: pd.Series) -> Dict[str, Any]:
    """
        Compute character and token length stats for a string series.
    """
    s = s.fillna("").astype("string")
    # character lengths
    char_lens = s.str.len().astype("float64")

    # simple whitespace tokenization for token count
    token_lens = s.str.split().apply(len).astype("float64")

    def _quantiles(a: pd.Series) -> Dict[str, float]:
        if a.size == 0:
            return {"min": 0, "q25": 0, "median": 0, "q75": 0, "max": 0}
        q = a.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
        return {"min": float(q.loc[0.0]), "q25": float(q.loc[0.25]), "median": float(q.loc[0.5]),
                "q75": float(q.loc[0.75]), "max": float(q.loc[1.0])}

    char_stats = {
        "count": int(char_lens.size),
        "mean": float(char_lens.mean()) if char_lens.size else 0.0,
        "std": float(char_lens.std(ddof=1)) if char_lens.size > 1 else 0.0,
        **_quantiles(char_lens),
    }
    token_stats = {
        "count": int(token_lens.size),
        "mean": float(token_lens.mean()) if token_lens.size else 0.0,
        "std": float(token_lens.std(ddof=1)) if token_lens.size > 1 else 0.0,
        **_quantiles(token_lens),
    }
    return {"chars": char_stats, "tokens": token_stats}

def compute_length_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
        Compute length stats for available text-like columns.
    """
    out: Dict[str, Any] = {}
    for col in EXPECTED_TEXT_COLUMNS:
        if col in df.columns:
            out[col] = _length_stats_from_series(df[col])
    # Also include a 'body' selection used later in modeling
    out["body"] = _length_stats_from_series(choose_body_series(df))
    return out

def compute_noise_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Estimate text noise over the chosen body:
      - url_ratio: share of rows containing ≥1 URL
      - email_ratio: share of rows containing ≥1 email
      - non_alnum_char_ratio: average share of non-alphanumeric characters per row
      - empty_text_ratio: share of empty/whitespace-only rows
    """
    s = choose_body_series(df).fillna("").astype("string")
    n = len(s)
    if n == 0:
        return {
            "url_ratio": 0.0,
            "email_ratio": 0.0,
            "non_alnum_char_ratio": 0.0,
            "empty_text_ratio": 0.0,
        }

    has_url = s.str.contains(RE_URL)
    has_email = s.str.contains(RE_EMAIL)
    # Compute non-alnum ratio per row
    def _non_alnum_ratio(x: str) -> float:
        if not x or x.strip() == "":
            return 0.0
        total = len(x)
        non_alnum = len(RE_NON_ALNUM.findall(x))
        return non_alnum / total if total > 0 else 0.0

    non_alnum_ratios = s.apply(_non_alnum_ratio)
    empty_ratio = float((s.str.strip().str.len() == 0).mean())

    return {
        "url_ratio": float(has_url.mean()),
        "email_ratio": float(has_email.mean()),
        "non_alnum_char_ratio": float(non_alnum_ratios.mean()),
        "empty_text_ratio": empty_ratio,
    }


def compute_date_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute simple date diagnostics if 'date_iso' is present."""
    if DATE_ISO_COL not in df.columns:
        return {"available": False}

    ser = df[DATE_ISO_COL]
    non_null = ser.notna()
    if non_null.sum() == 0:
        return {"available": True, "non_null": 0}

    s = ser[non_null]
    by_year = s.dt.year.value_counts(dropna=True).sort_index().to_dict()
    return {
        "available": True,
        "non_null": int(non_null.sum()),
        "min": s.min().strftime("%Y-%m-%d %H:%M:%S"),
        "max": s.max().strftime("%Y-%m-%d %H:%M:%S"),
        "by_year": {str(int(k)): int(v) for k, v in by_year.items() if pd.notna(k)},
    }

def _print_human_summary(r: Dict[str, Any]) -> None:
    shape = r["basic_overview"]["shape"]
    print("=== QA BEFORE CLEANING SUMMARY ===")
    print(f"Rows: {shape['rows']} | Cols: {shape['cols']}")
    # Class balance quick look
    cb = r.get("class_balance", {})
    if "label_numeric" in cb:
        counts = cb["label_numeric"]["count"]
        print(f"Label counts (numeric): {counts}")
    if "label_text" in cb:
        counts_t = cb["label_text"]["count"]
        print(f"Label counts (text): {counts_t}")
    # Duplicates quick look
    dups = r.get("duplicates", {})
    print(f"Duplicate message_id rows: {dups.get('duplicate_message_id_rows', 0)}")
    print(f"Duplicate text rows (exact on 'text'): {dups.get('duplicate_text_rows_exact', 0)}")
    print(f"Duplicate body rows (exact): {dups.get('duplicate_body_rows_exact', 0)}")
    # Noise quick look
    nm = r.get("noise_metrics", {})
    if nm:
        print(
            f"Noise: url_ratio={nm.get('url_ratio', 0):.3f}, "
            f"email_ratio={nm.get('email_ratio', 0):.3f}, "
            f"non_alnum_char_ratio={nm.get('non_alnum_char_ratio', 0):.3f}, "
            f"empty_text_ratio={nm.get('empty_text_ratio', 0):.3f}"
        )
    # Dates quick look
    ds = r.get("date_stats", {})
    if ds.get("available", False):
        print(
            f"Dates: non_null={ds.get('non_null', 0)}, "
            f"min={ds.get('min', 'NA')}, max={ds.get('max', 'NA')}"
        )
    print("==================\n")

def run_quality_check(input_csv: str, output_json: str):
    """
        Run full QA suite and write a JSON report to output_json.
    """
    df = format_csv_file_type(input_csv)

    report: Dict[str, Any] = {
        "source_csv": input_csv,
        "basic_overview": compute_basic_overview(df),
        "missing_values": compute_missing_values(df),
        "duplicates": compute_duplicates(df),
        "class_balance": compute_class_balance(df),
        "length_stats": compute_length_stats(df),
        "noise_metrics": compute_noise_metrics(df),
        "date_stats": compute_date_stats(df),
    }

    # JSON
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Console summary
    print(f"QA report written to: {output_json}", end="\n\n")
    _print_human_summary(report)

    return report