"""
Centralized baseline training using the global TF-IDF vectorizer and a simple PyTorch classifier.

Outputs:
- results/baseline_model.pt
- results/baseline_metrics.json
- results/plots/baseline_confusion_matrix.png
- results/seed.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# === Sparse TF-IDF Dataset ===

class SparseTfidfDataset(Dataset):
    """
    Wrap a (sparse) TF-IDF matrix (X) and a label vector (y) for PyTorch.
    Converts rows to dense on-the-fly to keep memory reasonable.
    """
    def __init__(self, X, y):
        self.X = X  # scipy sparse (csr)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert one row to dense vector
        row = self.X[idx]
        if hasattr(row, "toarray"):
            row = row.toarray()[0]
        x = torch.from_numpy(row).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

# === Model ===

class LogisticClassifier(nn.Module):
    """
    Logistic Regression-like classifier implemented with a single Linear layer.
    Input dim = vocab size; Output dim = 2 (ham/spam).
    """
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# === Training ===

def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_y, all_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += float(loss.item()) * xb.size(0)
        pred = torch.argmax(logits, dim=1)
        all_y.extend(yb.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_y, all_pred)
    f1 = f1_score(all_y, all_pred, average="macro")
    return avg_loss, acc, f1, np.array(all_y), np.array(all_pred)

def plot_confusion(cm: np.ndarray, out_path: str, labels=("ham", "spam")):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks([0,1], labels)
    plt.yticks([0,1], labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# === Main script ===

def run_baseline(
    csv_path: str,
    vec_path: str,
    out_dir: str = "results",
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 3,
    ):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "seed.txt", "w", encoding="utf-8") as f:
        f.write(str(random_state))

    # Load data
    df = pd.read_csv(csv_path)
    for col in ("text", "label"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from {csv_path}")
    X_text = df["text"].astype("string").fillna("").tolist()
    y = df["label"].astype(int).to_numpy()

    # Load vectorizer and transform
    vec = joblib.load(vec_path)
    X = vec.transform(X_text)  # sparse csr

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Datasets & loaders
    train_loader = DataLoader(SparseTfidfDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(SparseTfidfDataset(X_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)

    # Model
    input_dim = X.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogisticClassifier(input_dim=input_dim, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with early stopping
    best_f1 = -1.0
    best_state = None
    patience_left = patience

    history = []
    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_f1, _, _ = eval_model(model, test_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1})
        print(f"[Epoch {epoch:02d}] train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={va_acc:.4f} | val_f1={va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = model.state_dict()
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    # Load best state (if any)
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on test
    va_loss, va_acc, va_f1, y_true, y_pred = eval_model(model, test_loader, criterion, device)
    report = classification_report(y_true, y_pred, target_names=["ham", "spam"], output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # Save artifacts
    model_path = str(Path(out_dir) / "baseline_model.pt")
    torch.save(model.state_dict(), model_path)
    plot_confusion(cm, str(Path(out_dir) / "plots" / "baseline_confusion_matrix.png"))

    metrics = {
        "val_loss": va_loss,
        "val_acc": va_acc,
        "val_f1_macro": va_f1,
        "classification_report": report,
        "history": history,
        "model_path": model_path,
        "vectorizer_path": vec_path,
        "input_csv": csv_path,
    }
    with open(Path(out_dir) / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Baseline training done.")
    print(f"  - Acc: {va_acc:.4f} | F1-macro: {va_f1:.4f}")
    print(f"  - Saved: {model_path}")
    print(f"  - Metrics: {Path(out_dir) / 'baseline_metrics.json'}")
    print(f"  - Confusion matrix plot: {Path(out_dir) / 'plots' / 'baseline_confusion_matrix.png'}")