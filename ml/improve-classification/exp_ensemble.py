#!/usr/bin/env python3
"""Probability-ensemble experiment built from previous experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import RESULTS_DIR, ensure_dirs, set_seed
from evaluation import best_threshold, metric_row, prediction_frame, selective_rows


SOURCE_EXPERIMENTS = ("baseline", "feature_selection", "weight_decay")


def load_prediction_tables(results_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_tables = []
    test_tables = []
    for experiment in SOURCE_EXPERIMENTS:
        val_path = results_dir / ("%s_val_predictions.csv" % experiment)
        test_path = results_dir / ("%s_test_predictions.csv" % experiment)
        if val_path.exists():
            val_tables.append(pd.read_csv(val_path))
        if test_path.exists():
            test_tables.append(pd.read_csv(test_path))
    val = pd.concat(val_tables, ignore_index=True) if val_tables else pd.DataFrame()
    test = pd.concat(test_tables, ignore_index=True) if test_tables else pd.DataFrame()
    return val, test


def pivot_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "target", "Model", "proba_up"}
    if predictions.empty or not required.issubset(predictions.columns):
        return pd.DataFrame()
    return predictions.pivot_table(index=["date", "target"], columns="Model", values="proba_up", aggfunc="mean")


def ensemble_definitions(val_wide: pd.DataFrame, val_scores: pd.DataFrame) -> Dict[str, List[str]]:
    available = set(val_wide.columns)
    definitions = {
        "ENS_FINAL3": ["LGBM_exp100", "GBM_exp100", "XGB_linear03"],
        "ENS_XGB_LGBM_STEP": ["XGB_step50_3", "LGBM_step50_3"],
    }
    if not val_scores.empty and {"Model", "F1_macro"}.issubset(val_scores.columns):
        top_models = (
            val_scores[val_scores["Split"].eq("val")]
            .sort_values(["F1_macro", "Accuracy", "AUC"], ascending=False)["Model"]
            .drop_duplicates()
            .tolist()
        )
        definitions["ENS_TOP3_VAL"] = top_models[:3]
        definitions["ENS_TOP5_VAL"] = top_models[:5]
    return {name: [m for m in members if m in available] for name, members in definitions.items() if len([m for m in members if m in available]) >= 2}


def main() -> None:
    ensure_dirs()
    set_seed()
    val_predictions, test_predictions = load_prediction_tables(RESULTS_DIR)
    score_tables = []
    for experiment in SOURCE_EXPERIMENTS:
        path = RESULTS_DIR / ("%s_results.csv" % experiment)
        if path.exists():
            score_tables.append(pd.read_csv(path))
    val_scores = pd.concat(score_tables, ignore_index=True) if score_tables else pd.DataFrame()
    print("[ensemble] val_predictions=%s test_predictions=%s score_rows=%s" % (len(val_predictions), len(test_predictions), len(val_scores)))

    val_wide = pivot_predictions(val_predictions)
    test_wide = pivot_predictions(test_predictions)
    if val_wide.empty or test_wide.empty:
        print("[ensemble] no valid source predictions found; run baseline/feature_selection/weight_decay first")
        pd.DataFrame().to_csv(RESULTS_DIR / "ensemble_results.csv", index=False)
        return

    rows = []
    val_pred_frames = []
    test_pred_frames = []
    for name, members in ensemble_definitions(val_wide, val_scores).items():
        members = [m for m in members if m in test_wide.columns]
        if len(members) < 2:
            continue
        val_proba = val_wide[members].mean(axis=1).to_numpy()
        test_proba = test_wide[members].mean(axis=1).to_numpy()
        y_val = val_wide.index.get_level_values("target").to_numpy().astype(int)
        y_test = test_wide.index.get_level_values("target").to_numpy().astype(int)
        val_dates = pd.Series(val_wide.index.get_level_values("date").to_numpy())
        test_dates = pd.Series(test_wide.index.get_level_values("date").to_numpy())
        threshold = best_threshold(y_val, val_proba, metric="F1_macro")
        extra = {
            "Experiment": "ensemble",
            "ModelType": "average_proba_ensemble",
            "Scheme": ",".join(members),
            "FeatureSet": "from_members",
            "FeatureCount": np.nan,
        }
        rows.append(metric_row(name, "val", y_val, val_proba, threshold, dict(extra, ThresholdMode="val_f1_macro")))
        rows.append(metric_row(name, "test", y_test, test_proba, threshold, dict(extra, ThresholdMode="val_f1_macro")))
        rows.append(metric_row("%s_th05" % name, "test", y_test, test_proba, 0.5, dict(extra, ThresholdMode="fixed_0.5")))
        for row in selective_rows(name, y_test, test_proba):
            row.update(extra)
            row["ThresholdMode"] = "selective_fixed_0.5"
            rows.append(row)
        val_pred_frames.append(prediction_frame(name, "val", val_dates, y_val, val_proba, threshold, "ensemble"))
        test_pred_frames.append(prediction_frame(name, "test", test_dates, y_test, test_proba, threshold, "ensemble"))

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "ensemble_results.csv", index=False)
    if val_pred_frames:
        pd.concat(val_pred_frames, ignore_index=True).to_csv(RESULTS_DIR / "ensemble_val_predictions.csv", index=False)
    if test_pred_frames:
        pd.concat(test_pred_frames, ignore_index=True).to_csv(RESULTS_DIR / "ensemble_test_predictions.csv", index=False)

    top = results[results["Split"].eq("test")].sort_values(["F1_macro", "Accuracy", "AUC"], ascending=False).head(12)
    print(top[["Model", "Scheme", "Accuracy", "F1_macro", "AUC", "Threshold", "ThresholdMode"]].to_string(index=False))


if __name__ == "__main__":
    main()
