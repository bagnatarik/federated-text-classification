# src/server_app.py
"""
Flower ServerApp for Federated Learning on Enron Spam (TF-IDF + PyTorch linear classifier).

Features:
- FedAvg strategy with:
  * on_fit_config_fn: send per-round hyperparams to clients (local_epochs, lr, clip_c, sigma)
  * fit/evaluate metrics aggregation (weighted by num_examples)
- Saves per-round:
  * aggregated (global) weights → NPZ files
  * aggregated metrics JSONL (one line per round)
- Scenario name and output directory set via env vars.

Run:
    # simplest
    flwr run --app src/server_app.py

    # with custom env (bash)
    export ROUNDS=10
    export FRACTION_FIT=1.0
    export FRACTION_EVAL=1.0
    export MIN_FIT_CLIENTS=5
    export LOCAL_EPOCHS=1
    export LR=0.001
    export CLIP_C=0.0
    export SIGMA=0.0
    export SCENARIO_NAME="S1_FedAvg_IID_5"
    flwr run --app src/server_app.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import Metrics, NDArrays, Scalar, ndarrays_to_parameters

import numpy as np


# ----------------------------
# Env helpers
# ----------------------------

def _get_str(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default

def _get_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    try:
        return float(v) if v else default
    except Exception:
        return default

def _get_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    try:
        return int(v) if v else default
    except Exception:
        return default


# Orchestration config (server-side)
ROUNDS          = _get_int("ROUNDS", 10)
FRACTION_FIT    = _get_float("FRACTION_FIT", 1.0)
FRACTION_EVAL   = _get_float("FRACTION_EVAL", 1.0)
MIN_FIT_CLIENTS = _get_int("MIN_FIT_CLIENTS", 2)
MIN_EVAL_CLIENTS= _get_int("MIN_EVAL_CLIENTS", 2)
MIN_AVAILABLE   = _get_int("MIN_AVAILABLE_CLIENTS", 2)

# What we send to clients each round (can be overridden here)
LOCAL_EPOCHS    = _get_int("LOCAL_EPOCHS", 1)
LR              = _get_float("LR", 1e-3)
CLIP_C          = _get_float("CLIP_C", 0.0)   # 0 => disabled
SIGMA           = _get_float("SIGMA", 0.0)    # 0 => disabled

# Output
SCENARIO_NAME   = _get_str("SCENARIO_NAME", "S1_FedAvg_IID_5")
OUT_DIR         = Path(_get_str("OUTPUT_DIR", "results")) / "fl_runs" / SCENARIO_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_JSONL   = OUT_DIR / "metrics_rounds.jsonl"


# ----------------------------
# Aggregation helpers
# ----------------------------

def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted avg of client fit metrics by num_examples."""
    if not metrics:
        return {}
    total_examples = sum(num for num, _ in metrics)
    out: Dict[str, float] = {}
    for num, m in metrics:
        w = num / total_examples if total_examples > 0 else 0.0
        for k, v in m.items():
            if isinstance(v, (int, float)):
                out[k] = out.get(k, 0.0) + float(v) * w
    out["num_examples"] = float(total_examples)
    return out

def weighted_average_eval(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted avg of client eval metrics by num_examples (expects 'val_acc' key)."""
    if not metrics:
        return {}
    total_examples = sum(num for num, _ in metrics)
    out: Dict[str, float] = {}
    for num, m in metrics:
        w = num / total_examples if total_examples > 0 else 0.0
        for k, v in m.items():
            if isinstance(v, (int, float)):
                out[k] = out.get(k, 0.0) + float(v) * w
    out["num_examples"] = float(total_examples)
    return out


# ----------------------------
# Persistence helpers
# ----------------------------

def save_global_weights_ndarrays(weights: Optional[NDArrays], round_idx: int) -> None:
    """Save global model weights as a .npz archive per round."""
    if weights is None:
        return
    arrs = {f"w{i}": w for i, w in enumerate(weights)}
    path = OUT_DIR / f"global_weights_round_{round_idx:03d}.npz"
    np.savez_compressed(path, **arrs)

def append_round_metrics(
    round_idx: int,
    loss_fit: Optional[float],
    metrics_fit: Optional[Metrics],
    loss_eval: Optional[float],
    metrics_eval: Optional[Metrics],
) -> None:
    """Append a JSON line for this round into metrics_rounds.jsonl."""
    record = {
        "round": round_idx,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "fit": {
            "loss": loss_fit if loss_fit is not None else None,
            "metrics": metrics_fit or {},
        },
        "evaluate": {
            "loss": loss_eval if loss_eval is not None else None,
            "metrics": metrics_eval or {},
        },
    }
    with METRICS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ----------------------------
# Strategy (FedAvg) with hooks
# ----------------------------

def fit_config_fn(server_round: int) -> Dict[str, Scalar]:
    """Config sent to clients before each fit()."""
    return {
        "local_epochs": float(LOCAL_EPOCHS),
        "lr": float(LR),
        "clip_c": float(CLIP_C),
        "sigma": float(SIGMA),
        "server_round": float(server_round),
    }

def eval_config_fn(server_round: int) -> Dict[str, Scalar]:
    """Config sent to clients before each evaluate()."""
    return {"server_round": float(server_round)}

class SavingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that saves global weights and metrics after each round."""

    def aggregate_fit(  # type: ignore[override]
        self,
        server_round: int,
        results: List[fl.server.client_proxy.FitRes],
        failures: List[BaseException],
    ):
        agg_res = super().aggregate_fit(server_round, results, failures)
        if agg_res is None:
            # Nothing to save
            append_round_metrics(server_round, None, None, None, None)
            return agg_res

        parameters_agg, metrics_agg = agg_res
        # Convert aggregated parameters to ndarrays to save
        weights: NDArrays = fl.common.parameters_to_ndarrays(parameters_agg)
        save_global_weights_ndarrays(weights, server_round)

        # Also log aggregated fit metrics
        # Note: Flower does not directly return weighted loss here; we'll store metrics_agg.
        append_round_metrics(server_round, None, metrics_agg, None, None)
        return parameters_agg, metrics_agg

    def aggregate_evaluate(  # type: ignore[override]
        self,
        server_round: int,
        results: List[fl.server.client_proxy.EvaluateRes],
        failures: List[BaseException],
    ):
        agg_res = super().aggregate_evaluate(server_round, results, failures)
        if agg_res is None:
            # still append an empty entry for evaluate phase
            append_round_metrics(server_round, None, None, None, None)
            return agg_res

        loss_agg, metrics_agg = agg_res
        # Append to JSONL (we don't have fit loss here, so we keep it None)
        append_round_metrics(server_round, None, None, float(loss_agg), metrics_agg)
        return loss_agg, metrics_agg


def build_strategy() -> fl.server.strategy.Strategy:
    """Create the FedAvg strategy with our hooks and aggregations."""
    strategy = SavingFedAvg(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVAL,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVAL_CLIENTS,
        min_available_clients=MIN_AVAILABLE,
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=eval_config_fn,
        # Let the first selected client initialize weights (simplest for TF-IDF linear model)
        initial_parameters=None,
        # Aggregate client-reported metrics
        fit_metrics_aggregation_fn=weighted_average_fit,
        evaluate_metrics_aggregation_fn=weighted_average_eval,
    )
    return strategy


# ----------------------------
# ServerApp entrypoint
# ----------------------------

def start_server(context: fl.common.Context) -> fl.server.ServerConfig:
    """Called by Flower to configure the server run."""
    # Configure number of rounds
    return fl.server.ServerConfig(num_rounds=ROUNDS)

def server_initializer(context: fl.common.Context) -> fl.server.strategy.Strategy:
    """Return the strategy instance (called once at startup)."""
    # Ensure output dir exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Write a small run manifest
    manifest = {
        "scenario": SCENARIO_NAME,
        "rounds": ROUNDS,
        "fraction_fit": FRACTION_FIT,
        "fraction_evaluate": FRACTION_EVAL,
        "min_fit_clients": MIN_FIT_CLIENTS,
        "min_evaluate_clients": MIN_EVAL_CLIENTS,
        "min_available_clients": MIN_AVAILABLE,
        "client_config": {
            "local_epochs": LOCAL_EPOCHS,
            "lr": LR,
            "clip_c": CLIP_C,
            "sigma": SIGMA,
        },
        "started_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with (OUT_DIR / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return build_strategy()


# Expose the ServerApp for `flwr run --app src/server_app.py`
app = fl.server.ServerApp(
    config_fn=start_server,
    strategy_fn=server_initializer,
)
