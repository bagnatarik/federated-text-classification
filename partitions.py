"""
Data partitioning for Federated Learning on the Enron Spam dataset.

This module creates client partitions over the cleaned CSV (expected columns: ['message_id','text','label']).
Two strategies are supported:
  - IID (stratified): each client receives an approximately equal, class-balanced share.
  - non-IID (Dirichlet): per-class proportions for each client sampled from a Dirichlet distribution.

Features:
- Reproducible splits with --seed
- Optional global holdout set for centralized evaluation (--holdout-frac)
- Minimum samples per client guard (--min-per-client)
- Saves a self-contained JSON describing partitions and meta

Usage examples:
    # 5 IID clients, no holdout
    python -m src.partition --csv data/enron_clean.csv --out results/partitions_iid.json \
        --num-clients 5 --strategy iid --seed 42

    # 20 non-IID clients via Dirichlet alpha=0.3 with 10% global holdout
    python -m src.partition --csv data/enron_clean.csv --out results/partitions_non_iid.json \
        --num-clients 20 --strategy dirichlet --alpha 0.3 --holdout-frac 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# === UTILS ===

def _set_seed(seed: int | None):
    if seed is not None:
        np.random.seed(seed)

def load_clean_csv(path: str) -> pd.DataFrame:
    """Load cleaned CSV and sanity-check required columns."""
    df = pd.read_csv(path)
    required = {"message_id", "text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    # Normalize dtypes
    df["message_id"] = df["message_id"].astype("string")
    df["text"] = df["text"].astype("string")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    # Drop rows with missing label (cannot train without label)
    before = len(df)
    df = df.dropna(subset=["label"]).copy()
    if len(df) < before:
        print(f"[INFO] Dropped {before - len(df)} rows with missing label.")
    df["label"] = df["label"].astype(int)
    return df

# === IID (Stratified) PARTITIONING ===

def stratified_iid_indices(
    y: np.ndarray,
    num_clients: int,
    seed: int | None = None,
) -> Dict[int, List[int]]:
    """
    Create stratified IID partitions:
      - For each class, shuffle its indices then split into num_clients chunks as evenly as possible.
      - Concatenate per-class chunks for each client.
    """
    _set_seed(seed)
    y = np.asarray(y, dtype=int)
    clients: Dict[int, List[int]] = {cid: [] for cid in range(num_clients)}
    classes = np.unique(y)

    for c in classes:
        idx_c = np.where(y == c)[0]
        np.random.shuffle(idx_c)
        # Split as evenly as possible
        splits = np.array_split(idx_c, num_clients)
        for cid, chunk in enumerate(splits):
            clients[cid].extend(chunk.tolist())

    # Shuffle each client's indices (mix classes)
    for cid in range(num_clients):
        np.random.shuffle(clients[cid])
    return clients


# === NON-IID (Dirichlet) partitioning ===

def dirichlet_non_iid_indices(
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int | None = None,
    min_per_client: int = 1,
) -> Dict[int, List[int]]:
    """
    Create non-IID partitions using a Dirichlet distribution over class proportions.
    For each class k:
      1) Sample p_k ~ Dirichlet(alpha * 1_{num_clients})
      2) Allocate the indices of class k to clients according to p_k (multinomial w/o replacement)
    Ensures (best effort) each client gets at least 'min_per_client' samples overall.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0 for Dirichlet.")
    _set_seed(seed)
    y = np.asarray(y, dtype=int)
    n = y.shape[0]
    classes = np.unique(y)

    # Prepare containers
    client_indices: Dict[int, List[int]] = {cid: [] for cid in range(num_clients)}

    # Work per class
    for c in classes:
        idx_c = np.where(y == c)[0]
        np.random.shuffle(idx_c)
        m = len(idx_c)
        # Sample proportion vector for this class
        pk = np.random.dirichlet(alpha * np.ones(num_clients))
        # Convert proportions to integer allocations that sum to m
        # Start with floor, then distribute the remainder
        alloc = np.floor(pk * m).astype(int)
        remainder = m - int(alloc.sum())
        if remainder > 0:
            # distribute remainder according to largest fractional parts
            frac = pk * m - alloc
            extra = np.argsort(frac)[::-1][:remainder]
            alloc[extra] += 1

        # Now slice idx_c accordingly
        start = 0
        for cid in range(num_clients):
            cnt = int(alloc[cid])
            if cnt > 0:
                client_indices[cid].extend(idx_c[start : start + cnt].tolist())
            start += cnt

    # Optional: best-effort guard for min_per_client (reassign from largest pools)
    # Only if some client has < min_per_client and there are donors with > min_per_client.
    sizes = np.array([len(v) for v in client_indices.values()], dtype=int)
    need = np.where(sizes < min_per_client)[0]
    if len(need) > 0:
        donors = np.where(sizes > min_per_client)[0]
        if len(donors) > 0:
            # For simplicity, reassign random samples from largest donors
            for cid_need in need:
                deficit = min_per_client - sizes[cid_need]
                for d in donors:
                    if deficit <= 0:
                        break
                    if sizes[d] <= min_per_client:
                        continue
                    take = min(deficit, sizes[d] - min_per_client)
                    if take <= 0:
                        continue
                    # move 'take' examples
                    moved = client_indices[d][-take:]
                    del client_indices[d][-take:]
                    client_indices[cid_need].extend(moved)
                    sizes[d] -= take
                    sizes[cid_need] += take
                    deficit -= take

    # Final shuffle per client
    for cid in range(num_clients):
        np.random.shuffle(client_indices[cid])

    return client_indices

# === Holdout split (global) ===

def make_global_holdout(
    n: int,
    holdout_frac: float = 0.0,
    seed: int | None = None,
) -> Tuple[List[int], List[int]]:
    """
    Create a global holdout set (by indices) for centralized evaluation.
    Returns (train_indices, holdout_indices).
    """
    if holdout_frac <= 0.0:
        return list(range(n)), []
    if not (0.0 < holdout_frac < 1.0):
        raise ValueError("holdout_frac must be in (0, 1).")

    _set_seed(seed)
    all_idx = np.arange(n)
    np.random.shuffle(all_idx)
    hsz = int(round(holdout_frac * n))
    holdout = all_idx[:hsz].tolist()
    remain = all_idx[hsz:].tolist()
    return remain, holdout

def build_partitions(
    df: pd.DataFrame,
    num_clients: int,
    strategy: str = "iid",
    alpha: float = 0.5,
    seed: int | None = 42,
    holdout_frac: float = 0.0,
    min_per_client: int = 1,
) -> Dict:
    """
    Build partitions dict with metadata.
    Structure:
    {
      "meta": {...},
      "partitions": {"0": [...], "1": [...], ...},
      "holdout": [...]
    }
    """
    n = len(df)
    y = df["label"].to_numpy()
    # Global holdout first
    train_idx, holdout_idx = make_global_holdout(n, holdout_frac=holdout_frac, seed=seed)
    y_train = y[train_idx]

    if strategy.lower() == "iid":
        parts = stratified_iid_indices(y_train, num_clients=num_clients, seed=seed)
    elif strategy.lower() in {"non_iid", "non-iid", "dirichlet"}:
        parts = dirichlet_non_iid_indices(
            y_train,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            min_per_client=min_per_client,
        )
    else:
        raise ValueError("Unknown strategy. Use 'iid' or 'dirichlet'.")

    # Map back to global indices
    partitions_global: Dict[str, List[int]] = {}
    for cid, local_ids in parts.items():
        global_ids = [train_idx[i] for i in local_ids]
        partitions_global[str(cid)] = sorted(global_ids)

    meta = {
        "num_clients": num_clients,
        "strategy": strategy,
        "alpha": alpha if strategy.lower() in {"non_iid", "non-iid", "dirichlet"} else None,
        "seed": seed,
        "holdout_frac": holdout_frac,
        "min_per_client": min_per_client,
        "num_rows_total": n,
        "num_rows_holdout": len(holdout_idx),
        "num_rows_partitioned": sum(len(v) for v in partitions_global.values()),
        "class_distribution_total": {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
    }

    # Size sanity
    sizes = {cid: len(idx_list) for cid, idx_list in partitions_global.items()}
    meta["client_sizes"] = {str(k): int(v) for k, v in sizes.items()}

    return {
        "meta": meta,
        "partitions": partitions_global,
        "holdout": sorted(holdout_idx),
    }

def save_partitions(obj: Dict, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"Partitions saved to {out_path}.", end="\n\n")

# === Main script ===

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create FL client partitions (IID or Dirichlet non-IID).")
    p.add_argument("--csv", type=str, required=True, help="Path to cleaned CSV (message_id,text,label).")
    p.add_argument("--out", type=str, required=True, help="Output JSON path for partitions.")
    p.add_argument("--num-clients", type=int, required=True, help="Number of clients (K).")
    p.add_argument("--strategy", type=str, default="iid", choices=["iid", "dirichlet"], help="Partitioning strategy.")
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration (smaller = more skew).")
    p.add_argument("--holdout-frac", type=float, default=0.0, help="Global holdout fraction (0..1).")
    p.add_argument("--min-per-client", type=int, default=1, help="Min samples per client (best-effort for dirichlet).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    df = load_clean_csv(args.csv)
    parts_obj = build_partitions(
        df=df,
        num_clients=args.num_clients,
        strategy=args.strategy,
        alpha=args.alpha,
        seed=args.seed,
        holdout_frac=args.holdout_frac,
        min_per_client=args.min_per_client,
    )
    save_partitions(parts_obj, args.out)

    # Short console summary
    meta = parts_obj["meta"]
    print("\n=== PARTITION SUMMARY ===")
    print(f"Strategy: {meta['strategy']} | K={meta['num_clients']} | seed={meta['seed']}")
    if meta['strategy'] == "dirichlet":
        print(f"alpha={meta['alpha']}")
    print(f"Total rows: {meta['num_rows_total']}")
    print(f"Holdout rows: {meta['num_rows_holdout']} (frac={meta['holdout_frac']})")
    print("Client sizes:", meta["client_sizes"])
    print("=========================\n")


if __name__ == "__main__":
    main()