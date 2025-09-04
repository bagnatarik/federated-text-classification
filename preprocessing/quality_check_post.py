# src/preprocessing/quality_check_post.py
"""
Post-cleaning Quality Assessment (QA) for the Enron Spam dataset.

What this script does:
- Run a QA report on the CLEANED CSV (after your cleaning step).
- Optionally, if you provide a RAW/BEFORE CSV, it will also produce a BEFORE vs AFTER
  comparison JSON to highlight improvements (duplicates, empties, lengths, etc.).

Outputs:
- qa_after json file
- qa_compare json file
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Reuse same regexes as in pre-QA for consistency
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
RE_NON_ALNUM = re.compile(r"[^A-Za-z0-9\s]+")

# -----------------------------
# Basic loaders
# -----------------------------
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Minimal coercions for robustness
    for c in ("message_id", "text"):
        if c in df.columns:
            df[c] = df[c].astype("string")
    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    return df

# === QA METRICS ===

def basic_overview(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
    }


def missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    total = len(df)
    c = df.isna().sum().to_dict()
    r = {k: (float(v) / total if total else 0.0) for k, v in c.items()}
    return {"count": c, "ratio": r}


def duplicate_counts(df: pd.DataFrame) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    if "message_id" in df.columns:
        res["duplicate_message_id_rows"] = int(df["message_id"].duplicated(keep=False).sum())
    if "text" in df.columns:
        res["duplicate_text_rows_exact"] = int(df["text"].duplicated(keep=False).sum())
    return res


def class_balance(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "label" in df.columns:
        vals = df["label"].dropna().astype("int64")
        counts = vals.value_counts(dropna=False).to_dict()
        total = int(vals.shape[0])
        ratio = {str(k): (int(v) / total if total else 0.0) for k, v in counts.items()}
        out["label_numeric"] = {
            "count": {str(int(k)): int(v) for k, v in counts.items()},
            "ratio": ratio,
        }
    return out


def length_stats(df: pd.DataFrame) -> Dict[str, Any]:
    def _stats(s: pd.Series) -> Dict[str, Any]:
        s = s.fillna("").astype("string")
        char_l = s.str.len()
        tok_l = s.str.split().apply(len)

        def q(a: pd.Series) -> Dict[str, float]:
            if a.size == 0:
                return {"min": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "max": 0.0}
            qs = a.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
            return {
                "min": float(qs.loc[0.0]),
                "q25": float(qs.loc[0.25]),
                "median": float(qs.loc[0.5]),
                "q75": float(qs.loc[0.75]),
                "max": float(qs.loc[1.0]),
            }

        return {
            "chars": {
                "count": int(char_l.size),
                "mean": float(char_l.mean()) if char_l.size else 0.0,
                "std": float(char_l.std(ddof=1)) if char_l.size > 1 else 0.0,
                **q(char_l),
            },
            "tokens": {
                "count": int(tok_l.size),
                "mean": float(tok_l.mean()) if tok_l.size else 0.0,
                "std": float(tok_l.std(ddof=1)) if tok_l.size > 1 else 0.0,
                **q(tok_l),
            },
        }

    out: Dict[str, Any] = {}
    if "text" in df.columns:
        out["text"] = _stats(df["text"])
    return out


def noise_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    s = df["text"].fillna("").astype("string") if "text" in df.columns else pd.Series([], dtype="string")
    n = len(s)
    if n == 0:
        return {"url_ratio": 0.0, "email_ratio": 0.0, "non_alnum_char_ratio": 0.0, "empty_text_ratio": 0.0}

    has_url = s.str.contains(RE_URL)
    has_email = s.str.contains(RE_EMAIL)

    def _non_alnum_ratio(x: str) -> float:
        if not x or x.strip() == "":
            return 0.0
        total = len(x)
        non_alnum = len(RE_NON_ALNUM.findall(x))
        return (non_alnum / total) if total else 0.0

    nonal = s.apply(_non_alnum_ratio)
    empty = (s.str.strip().str.len() == 0).mean()
    return {
        "url_ratio": float(has_url.mean()),
        "email_ratio": float(has_email.mean()),
        "non_alnum_char_ratio": float(nonal.mean()),
        "empty_text_ratio": float(empty),
    }


def run_after_report(after_csv: str, out_after: str) -> Dict[str, Any]:
    df = load_csv(after_csv)
    report = {
        "source_csv": after_csv,
        "basic_overview": basic_overview(df),
        "missing_values": missing_values(df),
        "duplicates": duplicate_counts(df),
        "class_balance": class_balance(df),
        "length_stats": length_stats(df),
        "noise_metrics": noise_metrics(df),
    }
    Path(out_after).parent.mkdir(parents=True, exist_ok=True)
    with open(out_after, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"QA (after) written to {out_after}", end="\n\n")
    return report


# -----------------------------
# BEFORE vs AFTER comparison
# -----------------------------
def _safe_get(d: Dict[str, Any], path: list[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_compare(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Compute deltas (before to after) for key indicators."""
    comp: Dict[str, Any] = {
        "rows": {
            "before": _safe_get(before, ["basic_overview", "shape", "rows"], 0),
            "after": _safe_get(after, ["basic_overview", "shape", "rows"], 0),
        },
        "duplicates": {
            "text_rows_exact": {
                "before": _safe_get(before, ["duplicates", "duplicate_body_rows_exact"], _safe_get(before, ["duplicates", "duplicate_text_rows_exact"], 0)),
                "after": _safe_get(after, ["duplicates", "duplicate_text_rows_exact"], 0),
            }
        },
        "empties": {
            "before_empty_text_ratio": _safe_get(before, ["noise_metrics", "empty_text_ratio"], 0.0),
            "after_empty_text_ratio": _safe_get(after, ["noise_metrics", "empty_text_ratio"], 0.0),
        },
        "noise_non_alnum_char_ratio": {
            "before": _safe_get(before, ["noise_metrics", "non_alnum_char_ratio"], 0.0),
            "after": _safe_get(after, ["noise_metrics", "non_alnum_char_ratio"], 0.0),
        },
        "length_text_tokens_median": {
            "before": _safe_get(before, ["length_stats", "body", "tokens", "median"], _safe_get(before, ["length_stats", "text", "tokens", "median"], 0.0)),
            "after": _safe_get(after, ["length_stats", "text", "tokens", "median"], 0.0),
        },
        "length_text_chars_median": {
            "before": _safe_get(before, ["length_stats", "body", "chars", "median"], _safe_get(before, ["length_stats", "text", "chars", "median"], 0.0)),
            "after": _safe_get(after, ["length_stats", "text", "chars", "median"], 0.0),
        },
        "class_balance": {
            "before_label_ratio": _safe_get(before, ["class_balance", "label_numeric", "ratio"], {}),
            "after_label_ratio": _safe_get(after, ["class_balance", "label_numeric", "ratio"], {}),
        },
    }

    # Add simple deltas
    def delta(a, b):
        if a is None or b is None:
            return None
        try:
            return float(b) - float(a)
        except Exception:
            return None

    comp["duplicates"]["text_rows_exact"]["delta"] = delta(
        comp["duplicates"]["text_rows_exact"]["before"],
        comp["duplicates"]["text_rows_exact"]["after"],
    )
    comp["empties"]["delta"] = delta(
        comp["empties"]["before_empty_text_ratio"], comp["empties"]["after_empty_text_ratio"]
    )
    comp["noise_non_alnum_char_ratio"]["delta"] = delta(
        comp["noise_non_alnum_char_ratio"]["before"], comp["noise_non_alnum_char_ratio"]["after"]
    )
    comp["length_text_tokens_median"]["delta"] = delta(
        comp["length_text_tokens_median"]["before"], comp["length_text_tokens_median"]["after"]
    )
    comp["length_text_chars_median"]["delta"] = delta(
        comp["length_text_chars_median"]["before"], comp["length_text_chars_median"]["after"]
    )
    return comp

def run_quality_check_post(after_csv: str, out_after: str, before_json: str, out_compare: str) -> None:
    # Always run AFTER report on the cleaned CSV
    qa_after = run_after_report(after_csv, out_after)

    # Build BEFORE-vs-AFTER comparison (Optional)
    if before_json:
        if Path(before_json).exists():
            with open(before_json, "r", encoding="utf-8") as f:
                qa_before = json.load(f)
        else:
            qa_before = None

        if qa_before is not None:
            comp = build_compare(qa_before, qa_after)
            Path(out_compare).parent.mkdir(parents=True, exist_ok=True)
            with open(out_compare, "w", encoding="utf-8") as f:
                json.dump(comp, f, ensure_ascii=False, indent=2)
            print(f"BEFORE vs AFTER comparison written to {out_compare}.", end="\n\n")
        else:
            print("No BEFORE data available then skipped comparison.", end="\n\n")