#!/usr/bin/env python3
"""Deep-learning experiment: PyTorch MLP/GRU sequence models."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import (
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET,
    TARGET_DATE_COL,
    TEST_SPLIT_DATE,
    VAL_SPLIT_DATE,
    ensure_dirs,
    parse_int_csv_env,
    set_seed,
)
from evaluation import best_threshold, describe_splits, load_dataset, metric_row, split_masks


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


if TORCH_AVAILABLE:

    class GRUNet(nn.Module):
        def __init__(self, n_features: int, hidden: int = 24, dropout: float = 0.25):
            super().__init__()
            self.gru = nn.GRU(n_features, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1]).squeeze(-1)


    class FlatMLP(nn.Module):
        def __init__(self, lookback: int, n_features: int, hidden: int = 64, dropout: float = 0.35):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(lookback * n_features, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)


def set_torch_seed(seed: int = RANDOM_STATE) -> None:
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(max(1, int(os.getenv("TORCH_THREADS", "4"))))


def torch_device():
    if not TORCH_AVAILABLE:
        return None
    if os.getenv("DL_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_sequence_network(kind: str, lookback: int, n_features: int):
    if kind == "GRU":
        return GRUNet(n_features)
    if kind == "MLP":
        return FlatMLP(lookback, n_features)
    raise ValueError(kind)


def tensor_on_device(values: np.ndarray, device):
    return torch.tensor(values, dtype=torch.float32, device=device)


def predict_proba(model, xs: np.ndarray, mask: np.ndarray, device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(tensor_on_device(xs[mask], device))
        return torch.sigmoid(logits).detach().cpu().numpy()


def state_dict_cpu(model) -> Dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def fit_sequence_epochs(model, xs: np.ndarray, ys: np.ndarray, mask: np.ndarray, epochs: int, device) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(xs[mask], dtype=torch.float32), torch.tensor(ys[mask], dtype=torch.float32)),
        batch_size=64,
        shuffle=False,
    )
    for _ in range(max(1, int(epochs))):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()


def make_sequences(df: pd.DataFrame, features: List[str], lookback: int):
    values = df[features].to_numpy(dtype=np.float32)
    target = (df[TARGET] > 0).astype(np.float32).to_numpy()
    xs, ys, row_idx = [], [], []
    for idx in range(lookback - 1, len(df)):
        xs.append(values[idx - lookback + 1 : idx + 1])
        ys.append(target[idx])
        row_idx.append(idx)
    return np.stack(xs), np.asarray(ys, dtype=np.float32), np.asarray(row_idx)


def train_sequence_model(
    df: pd.DataFrame,
    features: List[str],
    lookback: int,
    kind: str,
    max_epochs: int,
    patience: int,
) -> Tuple[Dict, Dict]:
    device = torch_device()
    xs, ys, row_idx = make_sequences(df, features, lookback)
    split_dates = pd.to_datetime(df.loc[row_idx, TARGET_DATE_COL]).reset_index(drop=True)
    train_mask = (split_dates < VAL_SPLIT_DATE).to_numpy()
    val_mask = ((split_dates >= VAL_SPLIT_DATE) & (split_dates < TEST_SPLIT_DATE)).to_numpy()
    train_full_mask = (split_dates < TEST_SPLIT_DATE).to_numpy()
    test_mask = (split_dates >= TEST_SPLIT_DATE).to_numpy()

    mean = xs[train_mask].reshape(-1, xs.shape[-1]).mean(axis=0)
    std = xs[train_mask].reshape(-1, xs.shape[-1]).std(axis=0) + 1e-6
    xs_for_selection = (xs - mean) / std

    model = make_sequence_network(kind, lookback, len(features)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(xs_for_selection[train_mask], dtype=torch.float32), torch.tensor(ys[train_mask], dtype=torch.float32)),
        batch_size=64,
        shuffle=False,
    )

    best_score = -np.inf
    best_state = None
    best_epoch = 1
    bad_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_proba = predict_proba(model, xs_for_selection, val_mask, device)
        threshold = best_threshold(ys[val_mask].astype(int), val_proba, metric="F1_macro")
        row = metric_row("tmp", "val", ys[val_mask].astype(int), val_proba, threshold)
        score = row["AUC"] + row["F1_macro"]
        if score > best_score:
            best_score = score
            best_state = state_dict_cpu(model)
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.to(device)
    val_proba = predict_proba(model, xs_for_selection, val_mask, device)

    threshold = best_threshold(ys[val_mask].astype(int), val_proba, metric="F1_macro")

    final_mean = xs[train_full_mask].reshape(-1, xs.shape[-1]).mean(axis=0)
    final_std = xs[train_full_mask].reshape(-1, xs.shape[-1]).std(axis=0) + 1e-6
    xs_for_final = (xs - final_mean) / final_std
    final_model = make_sequence_network(kind, lookback, len(features)).to(device)
    fit_sequence_epochs(final_model, xs_for_final, ys, train_full_mask, best_epoch, device)
    test_proba = predict_proba(final_model, xs_for_final, test_mask, device)

    name = "DL_%s_L%s" % (kind, lookback)
    extra = {
            "Experiment": "deep_learning",
            "ModelType": kind,
            "Device": str(device),
            "Lookback": lookback,
        "Epochs": best_epoch,
        "FinalRefit": "train_full",
        "ThresholdMode": "val_f1_macro",
    }
    rows = [
        metric_row(name, "val", ys[val_mask].astype(int), val_proba, threshold, extra),
        metric_row(name, "test", ys[test_mask].astype(int), test_proba, threshold, extra),
        metric_row("%s_th05" % name, "test", ys[test_mask].astype(int), test_proba, 0.5, dict(extra, ThresholdMode="fixed_0.5")),
    ]

    val_predictions = pd.DataFrame(
        {
            "Experiment": "deep_learning",
            "Model": name,
            "Split": "val",
            "date": df.loc[row_idx[val_mask], "date"].astype(str).to_numpy(),
            "target": ys[val_mask].astype(int),
            "proba_up": val_proba.astype(float),
            "pred_val_threshold": (val_proba >= threshold).astype(int),
            "pred_05": (val_proba >= 0.5).astype(int),
            "threshold": threshold,
        }
    )
    test_predictions = pd.DataFrame(
        {
            "Experiment": "deep_learning",
            "Model": name,
            "Split": "test",
            "date": df.loc[row_idx[test_mask], "date"].astype(str).to_numpy(),
            "target": ys[test_mask].astype(int),
            "proba_up": test_proba.astype(float),
            "pred_val_threshold": (test_proba >= threshold).astype(int),
            "pred_05": (test_proba >= 0.5).astype(int),
            "threshold": threshold,
        }
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state_dict_cpu(final_model),
            "features": features,
            "lookback": lookback,
            "kind": kind,
            "device": str(device),
            "threshold": threshold,
            "mean": final_mean,
            "std": final_std,
            "selected_epoch": best_epoch,
            "selection_mean": mean,
            "selection_std": std,
        },
        MODELS_DIR / ("%s.pt" % name),
    )
    return {"rows": rows, "val_predictions": val_predictions, "test_predictions": test_predictions}, {"name": name, "score": best_score}


def main() -> None:
    ensure_dirs()
    seed = set_seed()
    set_torch_seed(seed)
    df, features = load_dataset()
    masks = split_masks(df)
    lookbacks = parse_int_csv_env("DL_LOOKBACKS", "5,10,20,40")
    kinds = [k.strip().upper() for k in os.getenv("DL_KINDS", "MLP,GRU").split(",") if k.strip()]
    max_epochs = int(os.getenv("DL_MAX_EPOCHS", "120"))
    patience = int(os.getenv("DL_PATIENCE", "12"))
    device = torch_device() if TORCH_AVAILABLE else "unavailable"
    print(
        "[deep_learning] torch=%s device=%s lookbacks=%s kinds=%s splits=%s"
        % (TORCH_AVAILABLE, device, lookbacks, kinds, describe_splits(masks)),
        flush=True,
    )
    if not TORCH_AVAILABLE:
        pd.DataFrame().to_csv(RESULTS_DIR / "dl_results.csv", index=False)
        pd.DataFrame().to_csv(RESULTS_DIR / "dl_val_predictions.csv", index=False)
        pd.DataFrame().to_csv(RESULTS_DIR / "dl_test_predictions.csv", index=False)
        return

    rows = []
    val_predictions = []
    test_predictions = []
    for lookback in lookbacks:
        for kind in kinds:
            print("[deep_learning] train %s lookback=%s" % (kind, lookback), flush=True)
            result, _ = train_sequence_model(df, features, lookback, kind, max_epochs=max_epochs, patience=patience)
            rows.extend(result["rows"])
            val_predictions.append(result["val_predictions"])
            test_predictions.append(result["test_predictions"])

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "dl_results.csv", index=False)
    pd.concat(val_predictions, ignore_index=True).to_csv(RESULTS_DIR / "dl_val_predictions.csv", index=False)
    pd.concat(test_predictions, ignore_index=True).to_csv(RESULTS_DIR / "dl_test_predictions.csv", index=False)
    top = results[results["Split"].eq("test")].sort_values(["F1_macro", "Accuracy", "AUC"], ascending=False).head(12)
    print(top[["Model", "Accuracy", "F1_macro", "AUC", "Threshold", "ThresholdMode"]].to_string(index=False))


if __name__ == "__main__":
    main()
