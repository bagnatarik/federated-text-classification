# src/client_app.py
"""
Flower ClientApp for Federated Learning on Enron Spam (TF-IDF + PyTorch linear classifier).

Key ideas:
- Each client loads its own data slice from a partition JSON (indices per client id).
- Reuse the global TF-IDF vectorizer (fitted once in Phase A).
- Train a tiny PyTorch classifier (LogReg-like) locally for a few epochs.
- Optional DP simulation: L2 gradient clipping (clip_C) + Gaussian noise on returned weights (sigma).

How to run (examples):
    # Server (in another terminal)
    # flwr run --app src/server_app.py

    # Clients (simulate K clients)
    # flwr run --app src/client_app.py --num-clients K

Environment variables (tune without editing code):
    DATA_CSV            default: "data/enron_clean.csv"
    PARTITIONS_JSON     default: "results/partitions_iid.json"
    VECTORIZER_PKL      default: "results/vectorizer.pkl"
    CLIENT_TEST_FRAC    default: "0.1"        # per-client local val/test fraction
    LOCAL_EPOCHS        default: "1"
    BATCH_SIZE          default: "128"
    LR                  default: "0.001"
    CLIP_C              default: "0"          # 0 or empty -> disabled, else clip grad norm to this value
    SIGMA               default: "0"          # 0 or empty -> disabled, else add N(0, sigma^2) to returned weights
    DEVICE              default: "auto"       # "cpu", "cuda", or "auto"
    SEED                default: "42"

Notes:
- The model is intentionally simple (single Linear layer). Keep parity with baseline.
- If you used a different model in baseline, adjust the architecture accordingly.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import flwr as fl
from flwr.common import NDArrays, Scalar
from flwr.client import NumPyClient, ClientApp


# ----------------------------
# Config & utilities
# ----------------------------

def getenv_str(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default

def getenv_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    try:
        return float(v) if v else default
    except Exception:
        return default

def getenv_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    try:
        return int(v) if v else default
    except Exception:
        return default

def select_device(device_env: str = "auto") -> torch.device:
    if device_env.lower() == "cpu":
        return torch.device("cpu")
    if device_env.lower() == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_CSV        = getenv_str("DATA_CSV", "data/enron_clean.csv")
PARTITIONS_JSON = getenv_str("PARTITIONS_JSON", "results/partitions_iid.json")
VECTORIZER_PKL  = getenv_str("VECTORIZER_PKL", "results/vectorizer.pkl")
CLIENT_TEST_FRAC= getenv_float("CLIENT_TEST_FRAC", 0.1)
LOCAL_EPOCHS    = getenv_int("LOCAL_EPOCHS", 1)
BATCH_SIZE      = getenv_int("BATCH_SIZE", 128)
LR              = getenv_float("LR", 1e-3)
CLIP_C          = getenv_float("CLIP_C", 0.0)      # <= 0 disables clipping
SIGMA           = getenv_float("SIGMA", 0.0)       # <= 0 disables noise
DEVICE          = select_device(getenv_str("DEVICE", "auto"))
SEED            = getenv_int("SEED", 42)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ----------------------------
# Data & model
# ----------------------------

class SparseTfidfDataset(Dataset):
    """Wrap a (sparse) TF-IDF matrix (X) and a label vector (y) for PyTorch."""
    def __init__(self, X, y):
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.X[idx]
        if hasattr(row, "toarray"):
            row = row.toarray()[0]
        x = torch.from_numpy(row).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


class LogisticClassifier(nn.Module):
    """Simple logistic regression-like classifier: single Linear layer."""
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def load_partitions(path: str) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # JSON is expected to have {"partitions": {"0": [...], "1":[...], ...}}
    parts = obj.get("partitions", obj)  # support raw dict as well
    return parts


def load_client_data(cid: int) -> Tuple[DataLoader, DataLoader, int]:
    """
    Load data for a specific client:
    - Read the global cleaned CSV
    - Select indices from partitions JSON for this client id
    - Load TF-IDF vectorizer and transform texts
    - Split client data into local train/val
    """
    df = pd.read_csv(DATA_CSV)
    if not {"message_id", "text", "label"}.issubset(df.columns):
        raise RuntimeError("DATA_CSV must contain columns: message_id, text, label")

    parts = load_partitions(PARTITIONS_JSON)
    key = str(cid)
    if key not in parts:
        raise RuntimeError(f"Client id {cid} not found in partitions file {PARTITIONS_JSON}")

    idx = parts[key]
    if len(idx) == 0:
        raise RuntimeError(f"Client id {cid} has no samples in partitions.")

    df_c = df.iloc[idx].reset_index(drop=True)
    texts = df_c["text"].astype("string").fillna("").tolist()
    y = df_c["label"].astype(int).to_numpy()

    # Transform with global TF-IDF
    vec = joblib.load(VECTORIZER_PKL)
    X = vec.transform(texts)

    # Local split (train/val)
    n = X.shape[0]
    if n < 2:
        # too small to split; put all in train and duplicate for val
        train_loader = DataLoader(SparseTfidfDataset(X, y), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(SparseTfidfDataset(X, y), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        input_dim = X.shape[1]
        return train_loader, val_loader, input_dim

    val_sz = max(1, int(math.floor(CLIENT_TEST_FRAC * n)))
    train_sz = n - val_sz
    # Shuffle indices reproducibly
    rng = np.random.default_rng(SEED + cid)
    perm = rng.permutation(n)
    train_ids = perm[:train_sz]
    val_ids = perm[train_sz:]

    X_train = X[train_ids]
    y_train = y[train_ids]
    X_val = X[val_ids]
    y_val = y[val_ids]

    train_loader = DataLoader(SparseTfidfDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(SparseTfidfDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    input_dim = X.shape[1]
    return train_loader, val_loader, input_dim


def get_weights(model: nn.Module) -> NDArrays:
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]

def set_weights(model: nn.Module, weights: NDArrays) -> None:
    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.copy_(torch.from_numpy(w).to(p.device))

# === Local train/eval loops ===

def train_locally(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    clip_c: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    total_loss = 0.0
    total_n = 0

    for _ in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            # Optional L2 gradient clipping
            if clip_c and clip_c > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_c))

            optimizer.step()

            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            total_n += bs

    avg_loss = total_loss / max(1, total_n)
    return avg_loss


@torch.no_grad()
def evaluate_locally(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_n = 0
    correct = 0

    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == yb).sum().item())
        total_loss += float(loss.item()) * xb.size(0)
        total_n += xb.size(0)

    avg_loss = total_loss / max(1, total_n)
    acc = correct / max(1, total_n)
    return avg_loss, acc


def add_gaussian_noise_to_weights(weights: NDArrays, sigma: float) -> NDArrays:
    if sigma <= 0:
        return weights
    noisy: NDArrays = []
    for w in weights:
        noisy.append(w + np.random.normal(loc=0.0, scale=sigma, size=w.shape).astype(w.dtype))
    return noisy

# === Flower client ===

class EnronClient(NumPyClient):
    """NumPyClient implementing fit/evaluate for one federated client."""

    def __init__(self, cid: int):
        super().__init__()
        self.cid = cid
        self.train_loader, self.val_loader, input_dim = load_client_data(cid)
        self.model = LogisticClassifier(input_dim=input_dim, num_classes=2).to(DEVICE)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        # Called by Flower to initialize client with server weights
        return get_weights(self.model)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        # Set global weights
        set_weights(self.model, parameters)

        # Read optional per-round config overrides (fallback to env)
        epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
        lr = float(config.get("lr", LR))
        clip_c = float(config.get("clip_c", CLIP_C))
        sigma = float(config.get("sigma", SIGMA))

        # Train locally
        train_loss = train_locally(
            self.model,
            self.train_loader,
            epochs=epochs,
            lr=lr,
            clip_c=clip_c if clip_c > 0 else 0.0,
            device=DEVICE,
        )

        # Prepare updated weights
        new_weights = get_weights(self.model)

        # Optional DP-sim: add Gaussian noise on returned weights
        if sigma and sigma > 0:
            new_weights = add_gaussian_noise_to_weights(new_weights, sigma=sigma)

        num_examples = len(self.train_loader.dataset)
        metrics: Dict[str, Scalar] = {
            "train_loss": float(train_loss),
            "cid": str(self.cid),
            "epochs": float(epochs),
        }
        if clip_c and clip_c > 0:
            metrics["clip_c"] = float(clip_c)
        if sigma and sigma > 0:
            metrics["sigma"] = float(sigma)

        return new_weights, num_examples, metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        # Evaluate with provided (global) parameters
        set_weights(self.model, parameters)
        val_loss, val_acc = evaluate_locally(self.model, self.val_loader, device=DEVICE)
        num_examples = len(self.val_loader.dataset)
        metrics: Dict[str, Scalar] = {"val_acc": float(val_acc), "cid": str(self.cid)}
        return float(val_loss), num_examples, metrics


# === ClientApp factory ===

def client_fn(context: fl.common.Context) -> fl.client.Client:
    """
    Factory called by Flower to instantiate the client for a given 'cid'.
    The 'cid' is provided by the simulation runtime.
    """
    cid_str = context.node_config["cid"]
    cid = int(cid_str)
    return EnronClient(cid)


# Expose the ClientApp for 'flwr run --app src/client_app.py'
app = ClientApp(client_fn)
