# fl_project/server_app.py
"""fl-project: Flower ServerApp for Enron TF-IDF + Logistic (PyTorch)."""

import os
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fl_project.task import Net, get_weights

# chemins (serveur a besoin du vocab size pour init les poids)
VECTORIZER_PKL = os.environ.get("VECTORIZER_PKL", "../../results/vectorizer.pkl")
OUT_DIR = Path(os.environ.get("OUTPUT_DIR", "../../results")) / "fl_runs" / os.environ.get("SCENARIO_NAME", "S1_FedAvg_IID")


def _manifest(context: Context) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scenario": os.environ.get("SCENARIO_NAME", "S1_FedAvg_IID"),
        "run_config": dict(context.run_config),
        "started_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with (OUT_DIR / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _initial_parameters_from_vectorizer() -> bytes:
    # Charge le vectorizer pour connaître la dimension d'entrée
    vec = joblib.load(VECTORIZER_PKL)
    input_dim = int(len(getattr(vec, "vocabulary_", {})))
    if input_dim <= 0:
        raise RuntimeError("Vectorizer vocabulary is empty. Fit the vectorizer first.")
    ndarrays = get_weights(Net(input_dim=input_dim))
    return ndarrays_to_parameters(ndarrays)


def server_fn(context: Context) -> ServerAppComponents:
    # Lire hyperparams depuis run_config (ou valeurs par défaut)
    num_rounds   = int(context.run_config.get("num-server-rounds", 10))
    fraction_fit = float(context.run_config.get("fraction-fit", 1.0))
    fraction_eval= float(context.run_config.get("fraction-eval", 1.0))
    min_avail    = int(context.run_config.get("min-available-clients", 2))

    # HParams envoyés aux clients à chaque round
    local_epochs = int(context.run_config.get("local-epochs", 1))
    lr           = float(context.run_config.get("lr", 1e-3))
    clip_c       = float(context.run_config.get("clip-c", 0.0))
    sigma        = float(context.run_config.get("sigma", 0.0))  # facultatif si tu ajoutes du bruit côté client

    def on_fit_config(server_round: int):
        return {
            "local-epochs": float(local_epochs),
            "lr": float(lr),
            "clip-c": float(clip_c),
            "sigma": float(sigma),
            "server_round": float(server_round),
        }

    # Stratégie FedAvg
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        min_available_clients=min_avail,
        on_fit_config_fn=on_fit_config,
        initial_parameters=_initial_parameters_from_vectorizer(),
    )

    # Config serveur
    config = ServerConfig(num_rounds=num_rounds)

    # Manifeste de run
    _manifest(context)

    return ServerAppComponents(strategy=strategy, config=config)


# ServerApp (utilisé par `flwr run .` ou `flwr run --app fl_project/server_app.py`)
app = ServerApp(server_fn=server_fn)
