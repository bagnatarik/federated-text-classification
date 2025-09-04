# fl_project/task.py
"""fl-project: Enron text classification with TF-IDF + PyTorch (logistic)."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
import math
from typing import Tuple, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------
# Env/config helpers
# -----------------------
def _get_s(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default

def _get_f(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    try:
        return float(v) if v else default
    except Exception:
        return default

def _get_i(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    try:
        return int(v) if v else default
    except Exception:
        return default


DATA_CSV        = _get_s("DATA_CSV", "data/enron_clean.csv")
PARTITIONS_JSON = _get_s("PARTITIONS_JSON", "../../results/partitions_iid.json")
VECTORIZER_PKL  = _get_s("VECTORIZER_PKL", "../../results/vectorizer.pkl")
CLIENT_TEST_FRAC= _get_f("CLIENT_TEST_FRAC", 0.1)
BATCH_SIZE      = _get_i("BATCH_SIZE", 128)
SEED            = _get_i("SEED", 42)

torch.manual_seed(SEED)
np.random.seed(SEED)


# -----------------------
# Data structures
# -----------------------
class SparseTfidfDataset(Dataset):
    """PyTorch dataset wrapping a (sparse) TF-IDF matrix and labels."""
    def __init__(self, X, y):
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        row = self.X[idx]
        if hasattr(row, "toarray"):
            row = row.toarray()[0]
        x = torch.from_numpy(row).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


class Net(nn.Module):
    """Logistic Regression-like linear classifier."""
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# -----------------------
# IO helpers
# -----------------------
def _load_partitions(path: str) -> dict:
    import json
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj.get("partitions", obj)

def _load_vectorizer(path: str):
    return joblib.load(path)


# -----------------------
# Public API used by template
# -----------------------
def load_data(partition_id: int, num_partitions: int):
    """
    Template-compatible loader.
    - `partition_id` correspond au client id (clé dans le JSON de partitions).
    - `num_partitions` est ignoré (on suit le JSON), mais conservé pour compatibilité.
    Retourne (trainloader, valloader).
    """
    # CSV (message_id, text, label)
    df = pd.read_csv(DATA_CSV)
    if not {"message_id", "text", "label"}.issubset(df.columns):
        raise RuntimeError("DATA_CSV must contain columns: message_id, text, label")

    # Partitions
    parts = _load_partitions(PARTITIONS_JSON)
    key = str(int(partition_id))
    if key not in parts:
        raise RuntimeError(f"partition-id {partition_id} not found in {PARTITIONS_JSON}")
    idx = parts[key]
    if len(idx) == 0:
        raise RuntimeError(f"partition-id {partition_id} has no samples.")

    df_c = df.iloc[idx].reset_index(drop=True)
    texts = df_c["text"].astype("string").fillna("").tolist()
    y = df_c["label"].astype(int).to_numpy()

    # Vectorizer
    vec = _load_vectorizer(VECTORIZER_PKL)
    X = vec.transform(texts)
    input_dim = int(X.shape[1])

    # Split local train/val
    n = X.shape[0]
    if n < 2:
        trainloader = DataLoader(SparseTfidfDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)
        valloader   = DataLoader(SparseTfidfDataset(X, y), batch_size=BATCH_SIZE, shuffle=False)
        return trainloader, valloader, input_dim

    val_sz = max(1, int(math.floor(CLIENT_TEST_FRAC * n)))
    train_sz = n - val_sz
    rng = np.random.default_rng(SEED + partition_id)
    perm = rng.permutation(n)
    tr_ids, va_ids = perm[:train_sz], perm[train_sz:]

    X_tr, y_tr = X[tr_ids], y[tr_ids]
    X_va, y_va = X[va_ids], y[va_ids]

    trainloader = DataLoader(SparseTfidfDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    valloader   = DataLoader(SparseTfidfDataset(X_va, y_va), batch_size=BATCH_SIZE, shuffle=False)
    return trainloader, valloader, input_dim


def train(net: nn.Module, trainloader: DataLoader, epochs: int, device: torch.device, lr: float = 1e-3, clip_c: float = 0.0):
    """Entraînement local (CrossEntropy + Adam), avec clipping L2 optionnel."""
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()
    tot_loss, tot_n = 0.0, 0
    for _ in range(epochs):
        for xb, yb in trainloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = net(xb)
            loss = criterion(logits, yb)
            loss.backward()
            if clip_c and clip_c > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(clip_c))
            optimizer.step()
            bs = xb.size(0)
            tot_loss += float(loss.item()) * bs
            tot_n += bs
    return tot_loss / max(1, tot_n)


@torch.no_grad()
def test(net: nn.Module, valloader: DataLoader, device: torch.device):
    """Évaluation locale (loss moyen + accuracy)."""
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    net.eval()
    tot_loss, tot_n, correct = 0.0, 0, 0
    for xb, yb in valloader:
        xb, yb = xb.to(device), yb.to(device)
        logits = net(xb)
        loss = criterion(logits, yb)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == yb).sum().item())
        tot_loss += float(loss.item()) * xb.size(0)
        tot_n += xb.size(0)
    acc = correct / max(1, tot_n)
    return (tot_loss / max(1, tot_n)), acc


def get_weights(net: nn.Module):
    return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net: nn.Module, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
