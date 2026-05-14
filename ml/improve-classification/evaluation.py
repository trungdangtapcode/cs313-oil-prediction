#!/usr/bin/env python3
"""Data loading, metrics, and reusable train/validation/test evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import (
    DATA_PATH,
    FINAL_RESULTS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    SORT_COLS,
    TARGET,
    TARGET_DATE_COL,
    TEST_SPLIT_DATE,
    VAL_SPLIT_DATE,
)
from model_zoo import Candidate, make_model, sample_weights


def load_dataset(path: Path = DATA_PATH) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df[TARGET_DATE_COL] = pd.to_datetime(df[TARGET_DATE_COL])
    features = [c for c in df.columns if c not in ["date", TARGET_DATE_COL, TARGET]]
    return df, features


def target_array(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] > 0).astype(int).to_numpy()


def split_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    split_dates = pd.to_datetime(df[TARGET_DATE_COL])
    return {
        "train": (split_dates < VAL_SPLIT_DATE).to_numpy(),
        "val": ((split_dates >= VAL_SPLIT_DATE) & (split_dates < TEST_SPLIT_DATE)).to_numpy(),
        "train_full": (split_dates < TEST_SPLIT_DATE).to_numpy(),
        "test": (split_dates >= TEST_SPLIT_DATE).to_numpy(),
    }


def describe_splits(masks: Dict[str, np.ndarray]) -> Dict[str, int]:
    return {name: int(mask.sum()) for name, mask in masks.items()}


def split_dates(df: pd.DataFrame, masks: Dict[str, np.ndarray], split: str) -> pd.Series:
    return df.loc[masks[split], "date"].reset_index(drop=True)


def scores_from_estimator(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    return model.predict(X).astype(float)


def safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba))


def metric_row(
    model: str,
    split: str,
    y_true: Sequence[int],
    proba: Sequence[float],
    threshold: float,
    extra: Optional[Dict] = None,
) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    row = {
        "Model": model,
        "Split": split,
        "Threshold": float(threshold),
        "Accuracy": accuracy_score(y_true, pred),
        "BalancedAcc": balanced_accuracy_score(y_true, pred),
        "F1_binary": f1_score(y_true, pred, zero_division=0),
        "F1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "Precision_UP": precision_score(y_true, pred, pos_label=1, zero_division=0),
        "Recall_UP": recall_score(y_true, pred, pos_label=1, zero_division=0),
        "Precision_DOWN": precision_score(y_true, pred, pos_label=0, zero_division=0),
        "Recall_DOWN": recall_score(y_true, pred, pos_label=0, zero_division=0),
        "AUC": safe_auc(y_true, proba),
        "MCC": matthews_corrcoef(y_true, pred),
        "LogLoss": log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6)),
        "Brier": brier_score_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6)),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "PosRate": float(pred.mean()),
        "TargetUpRate": float(y_true.mean()),
        "Coverage": 1.0,
        "N": int(len(y_true)),
    }
    if extra:
        row.update(extra)
    return row


def best_threshold(y_true: Sequence[int], proba: Sequence[float], metric: str = "F1_macro") -> float:
    best_t = 0.5
    best_score = -np.inf
    for threshold in np.linspace(0.30, 0.70, 161):
        row = metric_row("tmp", "val", y_true, proba, threshold)
        score = row[metric]
        if score > best_score:
            best_score = score
            best_t = float(threshold)
    return best_t


def selective_rows(model: str, y_true: Sequence[int], proba: Sequence[float]) -> List[Dict]:
    rows = []
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    for margin in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        mask = np.abs(proba - 0.5) >= margin
        if int(mask.sum()) < 30:
            continue
        rows.append(
            metric_row(
                model,
                "test_selective",
                y_true[mask],
                proba[mask],
                0.5,
                {"Margin": margin, "Coverage": float(mask.mean()), "N": int(mask.sum())},
            )
        )
    return rows


def load_final_baselines(final_results_dir: Path = FINAL_RESULTS_DIR) -> pd.DataFrame:
    rows = []
    selected_path = final_results_dir / "step6_results.csv"
    all_path = final_results_dir / "step6_test_all_schemes.csv"

    if selected_path.exists():
        selected = pd.read_csv(selected_path)
        if not selected.empty:
            row = selected.sort_values(SORT_COLS, ascending=False).iloc[0].to_dict()
            row.update({"Source": "final_step6_selected", "Split": "test"})
            rows.append(row)

    if all_path.exists():
        all_schemes = pd.read_csv(all_path)
        if not all_schemes.empty:
            by_f1 = all_schemes.sort_values(SORT_COLS, ascending=False).iloc[0].to_dict()
            by_f1.update({"Source": "final_step6_all_schemes_best_f1", "Split": "test"})
            rows.append(by_f1)

            by_auc = all_schemes.sort_values(["AUC", "F1_macro", "Accuracy"], ascending=False).iloc[0].to_dict()
            by_auc.update({"Source": "final_step6_all_schemes_best_auc", "Split": "test"})
            rows.append(by_auc)

    if rows:
        return pd.DataFrame(rows)

    existing = RESULTS_DIR / "final_baselines.csv"
    if existing.exists():
        return pd.read_csv(existing)
    return pd.DataFrame()


def prediction_frame(
    model: str,
    split: str,
    dates: pd.Series,
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    experiment: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Experiment": experiment,
            "Model": model,
            "Split": split,
            "date": dates.astype(str).to_numpy(),
            "target": y_true.astype(int),
            "proba_up": proba.astype(float),
            "pred_val_threshold": (proba >= threshold).astype(int),
            "pred_05": (proba >= 0.5).astype(int),
            "threshold": float(threshold),
        }
    )


def _fit_model(model, X, y, weights=None):
    if weights is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=weights)
    return model


def evaluate_candidate(
    df: pd.DataFrame,
    features: List[str],
    masks: Dict[str, np.ndarray],
    candidate: Candidate,
    experiment: str,
    use_sample_weight: bool = False,
    add_selective: bool = False,
    save_model: bool = False,
) -> Tuple[List[Dict], pd.DataFrame, pd.DataFrame, Dict]:
    y = target_array(df)
    X_train = df.loc[masks["train"], features]
    X_val = df.loc[masks["val"], features]
    X_train_full = df.loc[masks["train_full"], features]
    X_test = df.loc[masks["test"], features]
    y_train = y[masks["train"]]
    y_val = y[masks["val"]]
    y_train_full = y[masks["train_full"]]
    y_test = y[masks["test"]]

    val_model = make_model(candidate.model_kind)
    val_weights = sample_weights(len(y_train), candidate.scheme) if use_sample_weight else None
    _fit_model(val_model, X_train, y_train, val_weights)
    val_proba = scores_from_estimator(val_model, X_val)
    threshold = best_threshold(y_val, val_proba, metric="F1_macro")

    final_model = make_model(candidate.model_kind)
    final_weights = sample_weights(len(y_train_full), candidate.scheme) if use_sample_weight else None
    _fit_model(final_model, X_train_full, y_train_full, final_weights)
    test_proba = scores_from_estimator(final_model, X_test)

    extra = {
        "Experiment": experiment,
        "ModelType": candidate.model_kind,
        "Scheme": candidate.scheme,
        "FeatureSet": candidate.feature_set,
        "FeatureCount": len(features),
    }
    rows = [
        metric_row(candidate.name, "val", y_val, val_proba, threshold, dict(extra, ThresholdMode="val_f1_macro")),
        metric_row(candidate.name, "test", y_test, test_proba, threshold, dict(extra, ThresholdMode="val_f1_macro")),
        metric_row("%s_th05" % candidate.name, "test", y_test, test_proba, 0.5, dict(extra, ThresholdMode="fixed_0.5")),
    ]
    if add_selective:
        for row in selective_rows(candidate.name, y_test, test_proba):
            row.update(extra)
            row["ThresholdMode"] = "selective_fixed_0.5"
            rows.append(row)

    val_predictions = prediction_frame(candidate.name, "val", split_dates(df, masks, "val"), y_val, val_proba, threshold, experiment)
    test_predictions = prediction_frame(candidate.name, "test", split_dates(df, masks, "test"), y_test, test_proba, threshold, experiment)
    bundle = {
        "model": final_model,
        "features": features,
        "candidate": candidate.__dict__,
        "threshold": threshold,
        "threshold_mode": "val_f1_macro",
        "experiment": experiment,
    }
    if save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODELS_DIR / ("%s.joblib" % candidate.name))

    return rows, val_predictions, test_predictions, bundle


def write_experiment_outputs(
    experiment: str,
    rows: List[Dict],
    val_predictions: List[pd.DataFrame],
    test_predictions: List[pd.DataFrame],
) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / ("%s_results.csv" % experiment), index=False)
    if val_predictions:
        pd.concat(val_predictions, ignore_index=True).to_csv(RESULTS_DIR / ("%s_val_predictions.csv" % experiment), index=False)
    if test_predictions:
        pd.concat(test_predictions, ignore_index=True).to_csv(RESULTS_DIR / ("%s_test_predictions.csv" % experiment), index=False)
    return results
