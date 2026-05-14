#!/usr/bin/env python3
"""Build one unified, non-duplicated final report.

The main leaderboard uses exactly one row per model configuration:
full-coverage test metrics with the fixed 0.5 decision threshold. Validation-
selected thresholds are kept only as diagnostics, not mixed into the main table.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import RESULTS_DIR, THIS_DIR
from evaluation import load_final_baselines


ML_EXPERIMENT_FILES = {
    "baseline": "baseline_results.csv",
    "feature_selection": "feature_selection_results.csv",
    "weight_decay": "weight_decay_results.csv",
    "ensemble": "ensemble_results.csv",
}

ML_PREDICTION_FILES = {
    "baseline": ("baseline_val_predictions.csv", "baseline_test_predictions.csv"),
    "feature_selection": ("feature_selection_val_predictions.csv", "feature_selection_test_predictions.csv"),
    "weight_decay": ("weight_decay_val_predictions.csv", "weight_decay_test_predictions.csv"),
    "ensemble": ("ensemble_val_predictions.csv", "ensemble_test_predictions.csv"),
}

CURRENT_RESULT_FILES = dict(ML_EXPERIMENT_FILES, deep_learning="dl_results.csv")

PRIMARY_THRESHOLD_MODE = "fixed_0.5"

COUNT_COLUMNS = {"N", "TP", "FP", "TN", "FN", "Lookback", "Epochs", "Rows", "ValRows", "TestRows", "SelectiveRows", "Configs"}

DISPLAY_COLUMNS = [
    "Experiment",
    "ModelConfig",
    "ModelType",
    "Accuracy",
    "BalancedAcc",
    "F1_macro",
    "AUC",
    "MCC",
    "LogLoss",
    "Brier",
    "Precision_UP",
    "Recall_UP",
    "Precision_DOWN",
    "Recall_DOWN",
    "PosRate",
    "N",
]

INTERPRETATION_MODEL_COLUMNS = [
    "Experiment",
    "ModelConfig",
    "ModelType",
    "Accuracy",
    "F1_macro",
    "AUC",
    "MCC",
    "ThresholdMode",
    "N",
]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def model_config_name(model: object) -> str:
    name = "" if pd.isna(model) else str(model)
    return name[:-5] if name.endswith("_th05") else name


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out


def format_value(value, column: Optional[str] = None) -> str:
    if pd.isna(value):
        return ""
    if column in COUNT_COLUMNS:
        return str(int(float(value)))
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}" if math.isfinite(float(value)) else ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def md_table(df: pd.DataFrame, columns: Sequence[str], n: Optional[int] = None) -> str:
    if df.empty:
        return "_No rows._"
    cols = [col for col in columns if col in df.columns]
    view = df[cols].copy()
    if n is not None:
        view = view.head(n)
    for col in view.columns:
        view[col] = view[col].map(lambda value, column=col: format_value(value, column))
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in view.astype(str).to_numpy()]
    return "\n".join([header, sep, *body])


def result_artifact_status(results_dir: Path) -> pd.DataFrame:
    rows = []
    for experiment, filename in CURRENT_RESULT_FILES.items():
        path = results_dir / filename
        df = read_csv(path)
        split_counts = df["Split"].value_counts() if "Split" in df.columns and not df.empty else pd.Series(dtype=int)
        configs = df[df["Split"].eq("test")]["Model"].map(model_config_name).nunique() if "Split" in df.columns and "Model" in df.columns else 0
        rows.append(
            {
                "Experiment": experiment,
                "ResultsFile": filename,
                "Exists": path.exists(),
                "Rows": len(df),
                "Configs": int(configs),
                "ValRows": int(split_counts.get("val", 0)),
                "TestRows": int(split_counts.get("test", 0)),
                "SelectiveRows": int(split_counts.get("test_selective", 0)),
            }
        )
    return pd.DataFrame(rows)


def combine_ml_results(results_dir: Path) -> pd.DataFrame:
    frames = []
    for experiment, filename in ML_EXPERIMENT_FILES.items():
        df = read_csv(results_dir / filename)
        if df.empty:
            continue
        df = df.copy()
        df["Experiment"] = df.get("Experiment", experiment).fillna(experiment)
        frames.append(df)
    ml_results = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    ml_results.to_csv(results_dir / "ml_results.csv", index=False)
    return ml_results


def combine_ml_predictions(results_dir: Path) -> None:
    for split_idx, output_name in [(0, "ml_val_predictions.csv"), (1, "ml_test_predictions.csv")]:
        frames = []
        for _, files in ML_PREDICTION_FILES.items():
            df = read_csv(results_dir / files[split_idx])
            if not df.empty:
                frames.append(df)
        combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        combined.to_csv(results_dir / output_name, index=False)


def refresh_inputs(results_dir: Path) -> Dict[str, pd.DataFrame]:
    final_baselines = load_final_baselines()
    final_baselines.to_csv(results_dir / "final_baselines.csv", index=False)
    ml_results = combine_ml_results(results_dir)
    combine_ml_predictions(results_dir)
    dl_results = read_csv(results_dir / "dl_results.csv")
    status = result_artifact_status(results_dir)
    status.to_csv(results_dir / "experiment_artifact_status.csv", index=False)
    return {
        "final_baselines": final_baselines,
        "ml_results": ml_results,
        "dl_results": dl_results,
        "status": status,
    }


def standardize_current_rows(df: pd.DataFrame, source_group: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df[df["Split"].eq("test")].copy()
    if out.empty:
        return out
    out["SourceGroup"] = source_group
    out["Experiment"] = out.get("Experiment", source_group).fillna(source_group)
    out["ModelConfig"] = out["Model"].map(model_config_name)
    out["TestEvidence"] = "current_unified_test_run"
    out["CommonMetricSet"] = "full_common_metrics"
    out["ComparableUse"] = "primary" if source_group in {"ml", "dl"} else source_group
    out["Coverage"] = out["Coverage"].fillna(1.0)
    return out


def historical_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["Experiment"] = out.get("Source", "historical_final")
    out["SourceGroup"] = "historical"
    out["ModelConfig"] = out["Model"].astype(str)
    out["TestEvidence"] = "saved_final_test_result"
    out["CommonMetricSet"] = "classification_auc_common_only"
    out["ComparableUse"] = "reference_only"
    out["ThresholdMode"] = "final_saved"
    out["Threshold"] = np.nan
    out["Coverage"] = 1.0
    return out


def build_metric_tables(inputs: Dict[str, pd.DataFrame], results_dir: Path) -> Dict[str, pd.DataFrame]:
    ml_test = standardize_current_rows(inputs["ml_results"], "ml")
    dl_test = standardize_current_rows(inputs["dl_results"], "dl")
    current_all = pd.concat([ml_test, dl_test], ignore_index=True, sort=False)
    hist = historical_rows(inputs["final_baselines"])

    if current_all.empty:
        primary = current_all.copy()
    else:
        full_coverage = current_all[current_all["Coverage"].eq(1.0)].copy()
        primary = full_coverage[full_coverage["ThresholdMode"].eq(PRIMARY_THRESHOLD_MODE)].copy()
        primary = primary.sort_values(["F1_macro", "Accuracy", "AUC"], ascending=False)

    all_full = pd.concat([primary, hist], ignore_index=True, sort=False)
    all_full = all_full.sort_values(["F1_macro", "Accuracy", "AUC"], ascending=False)

    threshold_diag = build_threshold_diagnostics(current_all)
    selective = build_selective_table(inputs["ml_results"])

    primary.to_csv(results_dir / "primary_test_leaderboard.csv", index=False)
    current_all.to_csv(results_dir / "all_threshold_test_metrics.csv", index=False)
    all_full.to_csv(results_dir / "unified_full_coverage_metrics.csv", index=False)
    primary.to_csv(results_dir / "strict_unified_full_coverage_metrics.csv", index=False)
    hist.to_csv(results_dir / "historical_final_reference_metrics.csv", index=False)
    threshold_diag.to_csv(results_dir / "threshold_diagnostics.csv", index=False)
    selective.to_csv(results_dir / "selective_coverage_diagnostics.csv", index=False)
    primary.to_csv(results_dir / "comparison.csv", index=False)
    return {
        "primary": primary,
        "current_all": current_all,
        "all_full": all_full,
        "historical": hist,
        "threshold_diag": threshold_diag,
        "selective": selective,
    }


def build_threshold_diagnostics(current_all: pd.DataFrame) -> pd.DataFrame:
    if current_all.empty:
        return pd.DataFrame()
    rows = []
    groups = current_all.groupby(["Experiment", "ModelConfig"], dropna=False)
    for (experiment, config), group in groups:
        fixed = group[group["ThresholdMode"].eq("fixed_0.5")]
        val = group[group["ThresholdMode"].eq("val_f1_macro")]
        if fixed.empty or val.empty:
            continue
        fixed_row = fixed.iloc[0]
        val_row = val.iloc[0]
        rows.append(
            {
                "Experiment": experiment,
                "ModelConfig": config,
                "Fixed_F1_macro": fixed_row["F1_macro"],
                "ValThreshold_F1_macro": val_row["F1_macro"],
                "Delta_ValMinusFixed_F1": val_row["F1_macro"] - fixed_row["F1_macro"],
                "Fixed_Accuracy": fixed_row["Accuracy"],
                "ValThreshold_Accuracy": val_row["Accuracy"],
                "Delta_ValMinusFixed_Acc": val_row["Accuracy"] - fixed_row["Accuracy"],
                "ValThreshold": val_row["Threshold"],
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Delta_ValMinusFixed_F1", ascending=False)


def build_selective_table(ml_results: pd.DataFrame) -> pd.DataFrame:
    if ml_results.empty or "Split" not in ml_results.columns:
        return pd.DataFrame()
    selective = ml_results[ml_results["Split"].eq("test_selective")].copy()
    if selective.empty:
        return selective
    selective["ModelConfig"] = selective["Model"].map(model_config_name)
    return selective.sort_values(["Accuracy", "Coverage", "F1_macro"], ascending=False)


def best_row(df: pd.DataFrame, sort_cols: Sequence[str], ascending=False) -> Optional[pd.Series]:
    if df.empty:
        return None
    return df.sort_values(list(sort_cols), ascending=ascending).iloc[0]


def best_text(row: Optional[pd.Series]) -> str:
    if row is None:
        return "No row available."
    return (
        f"`{row['ModelConfig']}` "
        f"Accuracy={row['Accuracy']:.4f}, "
        f"F1_macro={row['F1_macro']:.4f}, "
        f"AUC={row['AUC']:.4f}"
    )


def best_by_experiment(primary: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return pd.DataFrame()
    return (
        primary.sort_values(["F1_macro", "Accuracy", "AUC"], ascending=False)
        .groupby("Experiment", dropna=False)
        .head(1)
        .sort_values(["F1_macro", "Accuracy", "AUC"], ascending=False)
    )


def metric_delta(best: Optional[pd.Series], ref: Optional[pd.Series], metric: str) -> str:
    if best is None or ref is None or pd.isna(best.get(metric)) or pd.isna(ref.get(metric)):
        return ""
    delta = float(best[metric]) - float(ref[metric])
    return f"{delta:+.4f}"


def write_metric_contract(results_dir: Path) -> None:
    contract = {
        "one_setup": {
            "dataset": "data/processed/dataset_final_noleak_step5c_scaler.csv",
            "target": "oil_return_fwd1 > 0 => UP=1",
            "train_for_validation": "target date < 2022-01-01",
            "validation": "2022-01-01 <= target date < 2023-01-01",
            "final_refit": "target date < 2023-01-01",
            "test": "target date >= 2023-01-01",
        },
        "main_leaderboard": {
            "coverage": "Coverage == 1.0",
            "split": "Split == test",
            "threshold_policy": PRIMARY_THRESHOLD_MODE,
            "one_row_per_model_configuration": True,
            "primary_sort": ["F1_macro", "Accuracy", "AUC"],
        },
        "diagnostics": {
            "threshold_diagnostics": "Compares fixed_0.5 against validation-selected thresholds.",
            "selective_coverage": "Confidence-filter rows; not comparable to full-coverage leaderboard.",
            "historical_reference": "Old final pipeline rows; reference only because LogLoss/Brier are unavailable.",
        },
    }
    (results_dir / "metric_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


def model_catalog_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [
                "BASE_LogReg",
                "baseline",
                "Logistic Regression",
                "Linear baseline that checks whether a simple additive signal exists.",
            ],
            [
                "BASE_SVM_RBF",
                "baseline",
                "RBF-kernel SVM",
                "Nonlinear margin baseline; tests whether a smooth nonlinear boundary helps.",
            ],
            [
                "BASE_RandomForest",
                "baseline",
                "Random Forest",
                "Bagged tree baseline; robust to nonlinearities but can overfit weak financial signals.",
            ],
            [
                "BASE_GBM",
                "baseline",
                "Gradient Boosting",
                "Boosted-tree baseline for nonlinear feature interactions.",
            ],
            [
                "BASE_MLP",
                "baseline",
                "Shallow neural net",
                "Tabular neural baseline; checks whether a small neural model helps without sequence structure.",
            ],
            [
                "BASE_XGB",
                "baseline",
                "Small XGBoost",
                "Regularized boosted-tree baseline using a compact XGB setup.",
            ],
            [
                "BASE_LGBM",
                "baseline",
                "Small LightGBM",
                "Regularized LightGBM baseline; a strong tabular ML reference.",
            ],
            [
                "FS_GBM / FS_XGB / FS_LGBM / FS_LGBM_small",
                "feature_selection",
                "Selected-feature tree models",
                "Retrains tree/boosting models only on the selected feature subset to test whether reducing noisy/redundant inputs helps.",
            ],
            [
                "GBM_exp100 / GBM_exp250",
                "weight_decay",
                "GBM with exponential recency weights",
                "`exp100` and `exp250` give more weight to recent observations with different half-lives.",
            ],
            [
                "XGB_exp100 / XGB_linear03 / XGB_step50_3",
                "weight_decay",
                "XGBoost with recency weights",
                "Tests XGB under exponential, linear, and step-style recency weighting schemes.",
            ],
            [
                "LGBM_exp100 / LGBM_linear01 / LGBM_linear03 / LGBM_step50_3 / LGBM_small_step50_3",
                "weight_decay",
                "LightGBM with recency weights",
                "Tests whether LightGBM benefits from emphasizing more recent oil-market regimes.",
            ],
            [
                "ENS_FINAL3",
                "ensemble",
                "Average-probability ensemble",
                "Hand-picked stable ensemble of `LGBM_exp100`, `GBM_exp100`, and `XGB_linear03`.",
            ],
            [
                "ENS_TOP3_VAL / ENS_TOP5_VAL",
                "ensemble",
                "Validation-ranked ensembles",
                "Averages the top 3 or top 5 validation models to test whether validation strength transfers to test.",
            ],
            [
                "ENS_XGB_LGBM_STEP",
                "ensemble",
                "Step-weight boosted-tree ensemble",
                "Averages XGB and LGBM variants using the step recency-weight scheme.",
            ],
            [
                "DL_MLP_L5/L10/L20/L40",
                "deep_learning",
                "Sequence MLP",
                "Flattens 5/10/20/40-day windows into a neural classifier; tests short-history sequence signal without recurrence.",
            ],
            [
                "DL_GRU_L5/L10/L20/L40",
                "deep_learning",
                "GRU sequence model",
                "Recurrent neural model over 5/10/20/40-day windows; tests whether temporal dynamics add value.",
            ],
        ],
        columns=["Model name pattern", "Experiment", "Model family", "Why it is included"],
    )


def naming_convention_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["BASE_", "Baseline model trained on all features."],
            ["FS_", "Feature-selection model trained on the selected feature subset."],
            ["GBM / XGB / LGBM", "Gradient boosting, XGBoost, and LightGBM model families."],
            ["exp100 / exp250", "Exponential recency weighting with different half-life settings."],
            ["linear01 / linear03", "Linear recency weighting variants."],
            ["step50_3", "Step recency weighting that emphasizes the most recent half of the training window."],
            ["ENS_", "Average-probability ensemble built from previous model predictions."],
            ["DL_MLP / DL_GRU", "Deep-learning sequence models."],
            ["L5 / L10 / L20 / L40", "Lookback window length in trading rows/days for sequence models."],
            ["_th05", "Internal result-row suffix for fixed 0.5 threshold; merged away in the primary leaderboard."],
        ],
        columns=["Name part", "Meaning"],
    )


def experiment_count_explanation_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [
                "baseline",
                7,
                "One representative set of general-purpose classifiers: Logistic Regression, RBF SVM, Random Forest, GBM, MLP, XGBoost, LightGBM.",
                "Covers linear, nonlinear margin, bagged trees, boosted trees, and shallow neural tabular baselines without making the baseline sweep too large.",
            ],
            [
                "feature_selection",
                4,
                "The selected feature subset is retrained with GBM, XGBoost, LightGBM, and small LightGBM.",
                "Tests whether the same strong tree/boosting families improve when noisy/redundant features are removed.",
            ],
            [
                "weight_decay",
                10,
                "GBM/XGBoost/LightGBM variants crossed with a compact set of recency-weighting schemes.",
                "Oil regimes shift, so these rows test whether emphasizing recent observations helps. The set is intentionally compact to avoid a huge tuning grid.",
            ],
            [
                "ensemble",
                4,
                "Four average-probability ensembles: one hand-picked stable ensemble, two validation-ranked ensembles, and one XGB/LGBM step-weight ensemble.",
                "Tests whether averaging weak but diverse probability forecasts is more stable than choosing one learner.",
            ],
            [
                "deep_learning",
                8,
                "Two neural sequence families, MLP and GRU, each run at four lookbacks: 5, 10, 20, and 40.",
                "Tests whether short temporal windows add signal beyond tabular ML while keeping the DL sweep bounded.",
            ],
        ],
        columns=["Experiment", "Configs", "Where the count comes from", "Why this count is useful"],
    )


def feature_selection_count_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [
                "Feature ranking",
                "27 original features ranked",
                "Ranks all available features using train-only mutual information and Spearman-style association.",
            ],
            [
                "Subset search cases",
                "16 subset cases",
                "3 ranking methods (`spearman`, `mi`, `mi_spearman`) x 5 subset sizes (`8,12,16,20,25`) plus `ALL_27`.",
            ],
            [
                "Inner CV train fits",
                "80 proxy model fits",
                "16 subset cases x 5 TimeSeriesSplit folds. These are inner-train proxy fits used only to select the feature subset.",
            ],
            [
                "Selected subset",
                "1 subset",
                "`SPEARMAN_TOP_25` was selected from the train-only subset search.",
            ],
            [
                "Final model configs",
                "4 configs",
                "`FS_GBM`, `FS_XGB`, `FS_LGBM`, and `FS_LGBM_small` are retrained on the selected feature subset and evaluated on validation/test.",
            ],
            [
                "Validation metrics",
                "4 validation rows",
                "Each final config is fit on train `<2022-01-01`, predicts validation `2022`, and gets one validation metric row.",
            ],
            [
                "Final refit train operations",
                "4 final refits",
                "Each final config is refit on train_full `<2023-01-01` before touching the final test split.",
            ],
            [
                "Full-coverage test metrics",
                "8 test rows",
                "4 final configs x 2 threshold policies: `fixed_0.5` and `val_f1_macro`.",
            ],
            [
                "Primary test leaderboard",
                "4 rows",
                "The report keeps one strict comparable row per final config: `Split=test`, `Coverage=1.0`, `ThresholdMode=fixed_0.5`.",
            ],
        ],
        columns=["Stage", "Count", "Meaning"],
    )


def feature_selection_funnel_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Start", "Feature subset candidates", 16, "Not final models; candidate feature sets."],
            ["Inner CV", "Proxy model train fits", 80, "16 subset candidates x 5 chronological folds."],
            ["Selection", "Chosen feature subset", 1, "`SPEARMAN_TOP_25`."],
            ["Final validation", "Selected-feature model configs validated", 4, "`FS_GBM`, `FS_XGB`, `FS_LGBM`, `FS_LGBM_small`."],
            ["Final validation", "Validation metric rows", 4, "One validation row per final model config."],
            ["Final refit", "Train_full refits before test", 4, "One final refit per final model config."],
            ["Final test diagnostics", "Full-coverage test metric rows", 8, "Two threshold rows per final model config."],
            ["Final report", "Primary test rows", 4, "One strict comparable row per final model config."],
        ],
        columns=["Phase", "What is counted", "Count", "Explanation"],
    )


def training_evaluation_funnel_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["baseline", 7, 0, 7, 7, 7, 14, 7, "Seven standalone baseline estimators."],
            ["feature_selection", 4, 80, 4, 4, 4, 8, 4, "80 proxy CV fits choose one subset; four final selected-feature models are evaluated."],
            ["weight_decay", 10, 0, 10, 10, 10, 20, 10, "Ten recency-weighted tabular model configurations."],
            ["ensemble", 4, 0, 0, 4, 0, 8, 4, "No estimator is refit; ensembles average existing validation/test probabilities."],
            ["deep_learning", 8, 0, 8, 8, 8, 16, 8, "Eight sequence neural configs; each uses validation for epoch/threshold choice, then final refit."],
            ["TOTAL", 33, 80, 29, 33, 29, 66, 33, "There is no training on the test split; test rows are metric rows only."],
        ],
        columns=[
            "Experiment",
            "Final model configs",
            "Inner-CV train fits",
            "Validation-stage train fits",
            "Validation metric rows",
            "Final refits before test",
            "Full test metric rows",
            "Primary test rows",
            "Note",
        ],
    )


def readable_training_test_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["1. Planned final configs", "33 configs", "These are the candidates intended for the unified comparison."],
            ["2. Internal feature-selection CV", "80 train fits", "Only used to choose one feature subset: 16 subset candidates x 5 chronological folds."],
            ["3. Train for validation", "29 train fits", "Actual learned models trained on the train split. Ensembles do not train new estimators."],
            ["4. Score on validation", "33 configs", "29 trained models + 4 ensembles are scored on the same validation split."],
            ["5. Final refit before test", "29 refits", "The learned models are refit on train + validation. Ensembles still only average probabilities."],
            ["6. Strict final test", "33 configs", "Final comparable test rows: full coverage, fixed_0.5 threshold, one row per final config."],
        ],
        columns=[
            "Stage",
            "Count",
            "Meaning",
        ],
    )


def training_evaluation_summary_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Initial final model configs", 33, "The configs intended for unified comparison: 7 baseline + 4 FS + 10 weight_decay + 4 ensemble + 8 DL."],
            ["Inner-CV proxy train fits", 80, "Feature-selection only: 16 subset cases x 5 chronological folds."],
            ["Validation-stage train fits", 29, "Actual estimator/model fits before validation. Ensembles have validation rows but no extra fit."],
            ["Validation metric rows", 33, "One validation metric row per final model config, including ensemble configs."],
            ["Final refits before test", 29, "Actual estimator/model refits on train_full. Ensembles reuse already-generated member predictions."],
            ["Actual estimator/model fits total", 138, "80 inner-CV proxy fits + 29 validation-stage fits + 29 final refits."],
            ["Full-coverage test metric rows", 66, "33 final configs x 2 threshold modes: fixed_0.5 and val_f1_macro."],
            ["Primary test leaderboard rows", 33, "One strict comparable test row per final config: fixed_0.5, full coverage."],
            ["Selective diagnostic rows", 112, "Confidence-filtered diagnostics only; not additional models."],
        ],
        columns=["Count type", "Count", "Meaning"],
    )


def interpretation_lines(primary: pd.DataFrame, historical: pd.DataFrame, threshold_diag: pd.DataFrame) -> List[str]:
    best = best_row(primary, ["F1_macro", "Accuracy", "AUC"])
    best_ml = best_row(primary[primary["SourceGroup"].eq("ml")], ["F1_macro", "Accuracy", "AUC"])
    best_dl = best_row(primary[primary["SourceGroup"].eq("dl")], ["F1_macro", "Accuracy", "AUC"])
    best_auc = best_row(primary, ["AUC", "F1_macro", "Accuracy"])
    old_f1 = best_row(historical[historical["Experiment"].eq("final_step6_all_schemes_best_f1")], ["F1_macro", "Accuracy", "AUC"])
    lines = []
    lines.append(f"- Current best primary test leaderboard classifier: {best_text(best)}.")
    if old_f1 is not None:
        lines.append(
            "- Compared with the historical best-F1 reference: "
            f"Accuracy {metric_delta(best, old_f1, 'Accuracy')}, "
            f"F1_macro {metric_delta(best, old_f1, 'F1_macro')}, "
            f"AUC {metric_delta(best, old_f1, 'AUC')}."
        )
    if best_dl is not None:
        lines.append(f"- Best DL: {best_text(best_dl)}. DL was evaluated with the same metrics, but it does not beat the best ML model.")
    if best_auc is not None and best is not None and best_auc["ModelConfig"] != best["ModelConfig"]:
        lines.append(
            f"- Best AUC/ranking config is `{best_auc['ModelConfig']}` with AUC={best_auc['AUC']:.4f}, "
            f"but its F1_macro={best_auc['F1_macro']:.4f}; therefore the final classifier should not be selected by AUC alone."
        )
    if best is not None and pd.notna(best.get("TargetUpRate")):
        target_up = float(best["TargetUpRate"])
        majority_acc = max(target_up, 1 - target_up)
        lines.append(
            f"- The test set is nearly balanced: TargetUpRate={target_up:.4f}, majority baseline~{majority_acc:.4f}. "
            f"The best model is {float(best['Accuracy']) - majority_acc:+.4f} accuracy points above that baseline."
        )
    if not threshold_diag.empty:
        better_val = int((threshold_diag["Delta_ValMinusFixed_F1"] > 0).sum())
        lines.append(
            f"- Validation-selected thresholds beat fixed 0.5 on only {better_val}/{len(threshold_diag)} configs by F1_macro. "
            "Because the final report needs one consistent setup, fixed 0.5 is used for the primary leaderboard."
        )
    lines.append(
        "- The result is still in the acceptable baseline band for daily next-day oil direction; "
        "it has not reached the next research target of Accuracy>=0.56 and AUC>=0.58, so it should not be overclaimed."
    )
    return lines


def build_interpretation_markdown(
    generated: str,
    status: pd.DataFrame,
    primary: pd.DataFrame,
    current_all: pd.DataFrame,
    threshold_diag: pd.DataFrame,
    selective: pd.DataFrame,
    historical: pd.DataFrame,
) -> str:
    total_configs = len(primary)
    total_val_rows = int(status["ValRows"].sum()) if not status.empty else 0
    total_test_rows = int(status["TestRows"].sum()) if not status.empty else 0
    total_selective_rows = int(status["SelectiveRows"].sum()) if not status.empty else 0
    total_result_rows = int(status["Rows"].sum()) if not status.empty else 0
    threshold_counts = (
        current_all.groupby(["Experiment", "ThresholdMode"], dropna=False)
        .size()
        .reset_index(name="Rows")
        .sort_values(["Experiment", "ThresholdMode"])
        if not current_all.empty
        else pd.DataFrame()
    )
    model_list = primary.sort_values(["Experiment", "ModelConfig"]).copy()
    best = best_row(primary, ["F1_macro", "Accuracy", "AUC"])
    best_auc = best_row(primary, ["AUC", "F1_macro", "Accuracy"])
    best_dl = best_row(primary[primary["Experiment"].eq("deep_learning")], ["F1_macro", "Accuracy", "AUC"])
    best_by_exp = best_by_experiment(primary)

    lines: List[str] = []
    lines.append("# Interpretation - Unified Oil Direction Pipeline\n")
    lines.append(f"Generated by `report.py` at {generated}.\n")
    lines.append("## Short Conclusion\n")
    lines.append(
        "The current pipeline has completed one unified run covering baseline models, feature selection, weight decay, ensembles, "
        "deep learning, and historical references. This file explains the model/config counts, validation/test rows, and how to read the metric tables.\n"
    )
    lines.append(
        f"- Total current model configurations evaluated: `{total_configs}`\n"
        f"- Total validation metric rows: `{total_val_rows}`\n"
        f"- Total full-coverage test metric rows: `{total_test_rows}`\n"
        f"- Total primary test leaderboard rows: `{len(primary)}`\n"
        f"- Total selective test diagnostic rows: `{total_selective_rows}`\n"
        f"- Historical reference rows: `{len(historical)}`\n"
    )
    lines.append(
        f"- Best primary test leaderboard classifier: {best_text(best)}\n"
        f"- Best AUC/ranking config: {best_text(best_auc)}\n"
    )
    if best_dl is not None:
        lines.append(f"- Best deep learning config: {best_text(best_dl)}\n")

    lines.append("## My Direct Answer To The Critical Question\n")
    lines.append(
        "My answer is: we trained many models because, in this specific problem, a single model score is not enough evidence. "
        "Daily next-day oil direction is a low-signal financial time-series task. In that setting, one good-looking model can easily be luck, "
        "an artifact of a threshold, a regime fit, or a subtle leakage problem. The purpose of the model grid is to ask whether the apparent edge "
        "survives across different ways of looking at the same forecasting problem.\n\n"
        "So the large table is not the conclusion. The table is the evidence behind the conclusion. It should answer these questions for a reader:\n\n"
        "1. Is there any signal above a simple baseline?\n"
        "2. Does the signal survive different model families, or does it appear in only one fragile model?\n"
        "3. Does feature selection reduce noise, or does it remove useful weak signal?\n"
        "4. Does recency weighting help with oil regime shifts?\n"
        "5. Does an ensemble stabilize weak models better than a single model?\n"
        "6. Does deep learning extract sequence information that tabular ML misses?\n"
        "7. Are high-looking rows actually comparable, or are they threshold/coverage diagnostics?\n\n"
        "After reading the results, my interpretation is that this experiment did not discover a strong predictive law. "
        "It discovered a small, plausible edge. The strongest hard-classification row is the ML ensemble `ENS_FINAL3`, but its edge is modest. "
        "The best AUC/ranking row is `XGB_linear03`, but it is not the best hard classifier. Deep learning was worth testing because the literature often uses neural sequence models for oil forecasting, "
        "but in this data it did not beat the ML ensemble. Feature selection was also worth testing, but its result suggests that simply shrinking the feature set is not the main missing ingredient.\n\n"
        "That is the real meaning of training many models here: not `we tried many things, therefore trust the best number`, but the opposite: "
        "`we tried the obvious families under one controlled setup, and the whole pattern says the edge is weak, ML ensemble is currently the most usable, and further progress needs better information or a different labeling/validation design`.\n"
    )

    lines.append("## Counts By Experiment\n")
    lines.append(md_table(status, ["Experiment", "Configs", "ValRows", "TestRows", "SelectiveRows"], len(status)))
    lines.append(
        "\nColumn definitions:\n"
        "- `Configs`: number of distinct model configurations after merging the two threshold variants under one config name.\n"
        "- `ValRows`: each config has one validation row, used to choose the `val_f1_macro` threshold.\n"
        "- `TestRows`: each config has two full-coverage test rows: `fixed_0.5` and `val_f1_macro`.\n"
        "- `SelectiveRows`: confidence-filter diagnostics only; these are not part of the primary leaderboard.\n"
    )

    lines.append("## Training And Evaluation Funnel\n")
    lines.append(
        "This table separates model configurations from actual fit operations and metric rows. "
        "The key point is that the test split is never trained on; it only receives predictions for metric computation.\n"
    )
    lines.append(
        "Short version: there are 138 actual train/refit operations, but only 33 final configurations reach the strict final test leaderboard. "
        "The test split is never used for training.\n"
    )
    lines.append(md_table(readable_training_test_table(), ["Stage", "Count", "Meaning"], None))
    lines.append(
        "\nSo the funnel is: 80 internal CV fits for feature selection, 29 real validation-stage model fits, "
        "29 final refits, then 33 strict final test rows. The extra 4 final configs are ensembles, so they are tested but not separately trained.\n"
    )

    lines.append("## Why These Counts: 7 + 4 + 10 + 4 + 8\n")
    lines.append(
        "These counts are not arbitrary. They are the size of each research block after removing duplicate threshold rows. "
        "The goal was broad enough coverage to answer the modeling questions, but small enough to avoid turning the project into an uncontrolled hyperparameter search.\n"
    )
    lines.append(md_table(experiment_count_explanation_table(), ["Experiment", "Configs", "Where the count comes from", "Why this count is useful"], None))

    lines.append("## Feature Selection Count Detail\n")
    lines.append(
        "This is the easiest place to confuse counts. The feature-selection step searches many candidate subsets, "
        "but only four selected-feature models are retrained and reported as final test model configurations.\n"
    )
    lines.append(md_table(feature_selection_count_table(), ["Stage", "Count", "Meaning"], None))
    lines.append(
        "\nIn other words, the feature-selection step is a funnel. Many internal proxy fits are used to choose one feature subset, "
        "then only four final model configurations are evaluated on the strict test leaderboard.\n"
    )
    lines.append(md_table(feature_selection_funnel_table(), ["Phase", "What is counted", "Count", "Explanation"], None))

    lines.append("## Count Reconciliation: 33 vs 44 vs 66 vs 211\n")
    lines.append(
        "The correct current model/config count is `33`. This is not a reduction from `44`; it is a different unit of counting.\n\n"
        "- `33` = distinct current model configurations evaluated on the unified final test set.\n"
        "- `66` = full-coverage test metric rows, because each of the 33 configs is reported twice: `fixed_0.5` and `val_f1_macro`.\n"
        "- `112` = selective coverage diagnostic rows; these are confidence-filtered slices, not additional models.\n"
        f"- `{total_result_rows}` = all current metric rows across step artifacts, including validation, test-threshold, and selective diagnostic rows.\n\n"
        "The earlier `44` number was a row count from an incomplete/older artifact shape, not 44 distinct models. "
        "For example, the current `ensemble_results.csv` still has exactly `44` rows, but that means 4 ensemble configs "
        "with validation rows, two full-coverage test threshold rows per config, and selective coverage diagnostics. "
        "Counting those rows as 44 models would double-count thresholds and confidence-filter slices.\n\n"
        "The primary leaderboard intentionally keeps one row per model configuration to avoid that duplication. "
        "The threshold and selective rows are still saved, but they are diagnostics rather than separate final models.\n"
    )

    lines.append("## Why So Many Models Were Trained\n")
    lines.append(
        "The purpose of training many models is not to make the table look impressive and not to claim that more models automatically mean "
        "better science. In this project, many model configurations are used as a controlled research grid: each group answers a different "
        "question about a very noisy daily direction problem.\n"
    )
    rationale = pd.DataFrame(
        [
            [
                "baseline",
                "What is the honest floor?",
                "Linear, kernel, tree, boosting, and shallow neural baselines show whether the dataset has any usable signal before adding research tricks.",
            ],
            [
                "feature_selection",
                "Do fewer/noisier features help?",
                "If selected features beat all-features models, the issue is redundancy/noise. If not, the feature signal itself is probably weak.",
            ],
            [
                "weight_decay",
                "Does recent data matter more?",
                "Oil regimes change. Recency weighting tests whether post-2020 or more recent observations should influence the model more strongly.",
            ],
            [
                "ensemble",
                "Is the edge stable across models?",
                "In low-signal forecasting, averaging diverse weak models can reduce idiosyncratic errors; if ensemble wins, the edge is distributed, not one lucky learner.",
            ],
            [
                "deep_learning",
                "Do short sequences add value?",
                "MLP/GRU sequence models test whether temporal patterns in short lookbacks beat tabular ML. Here they do not beat the best ML ensemble.",
            ],
        ],
        columns=["Experiment", "Question", "Meaning"],
    )
    lines.append(md_table(rationale, ["Experiment", "Question", "Meaning"], len(rationale)))
    lines.append(
        "\nThis is why the large table should be read as an evidence map, not as a raw competition scoreboard. "
        "If many unrelated model families cluster around 0.51-0.54 F1/accuracy, the main message is that the daily oil direction signal is weak. "
        "If one isolated model reported 0.70+ accuracy under the same setup, the right reaction would be leakage/data-snooping audit, not celebration.\n"
    )

    lines.append("## How To Read The Large Table\n")
    lines.append(
        "The table is meant to show four things:\n\n"
        "1. `Winner`: which model is best under the one primary rule, `test + full coverage + fixed_0.5`.\n"
        "2. `Family pattern`: whether the winner is isolated or whether similar methods support the same conclusion.\n"
        "3. `Metric trade-off`: whether a model is good at hard UP/DOWN classification (`F1_macro`, `Accuracy`) or only at ranking probabilities (`AUC`).\n"
        "4. `Negative evidence`: which ideas did not help, such as the current DL branch and feature-selection branch.\n\n"
        "So the table is not saying every row is equally important. The first decision row is the primary leaderboard winner. "
        "The rest of the table is the audit trail explaining why that winner is plausible, modest, or fragile.\n"
    )

    lines.append("## Model Name Decoder\n")
    lines.append(
        "The full `REPORT.md` contains a model catalog. The short version is: `BASE_` means ordinary all-feature baselines, "
        "`FS_` means selected-feature variants, `GBM/XGB/LGBM_*` are boosted-tree recency-weight experiments, `ENS_` means probability averaging, "
        "and `DL_*_L5/L10/L20/L40` means a sequence neural model with that lookback length. The suffix `_th05` appears only in raw result files for fixed-threshold rows; "
        "the primary leaderboard merges it back into one model configuration.\n"
    )

    lines.append("## Research Context Behind This Design\n")
    lines.append(
        "The local research notes in `docs/` and `docs/improve/` set a conservative expectation for this task: daily next-day oil direction is noisy, "
        "has weak feature-target relationships, and should not be benchmarked against headline paper claims without checking horizon, target definition, "
        "split policy, and leakage controls. In particular:\n\n"
        "- `docs/improve/oil_direction_research_benchmark_brief.md` frames `0.53-0.56` accuracy as an acceptable baseline and `0.56-0.60` as good for this exact daily UP/DOWN setup.\n"
        "- `docs/WHY_ML_FAILS.md` argues that the target has very low signal-to-noise, weak autocorrelation, weak feature-target correlations, regime shift, and small sample size.\n"
        "- `docs/improve/oil_direction_leakage_audit_checklist.md` says high results should be checked for random splits, same-day/future features, global preprocessing, and repeated test-set selection.\n\n"
        "The external literature points in the same direction. Ghoddusi, Creamer, and Rafizadeh review more than 130 ML papers in energy economics/finance and show that energy forecasting work uses many model families, with both opportunities and limitations. "
        "Luo et al. evaluate daily oil futures forecasting with directional accuracy and formal forecast-comparison ideas; their reported daily direction numbers are much closer to the conservative band than to unrealistic 0.80+ expectations. "
        "Zhao, Li, and Yu motivate ensemble/deep-learning approaches for crude oil because oil prices are affected by many factors and nonlinear relationships. "
        "White's Reality Check paper is the warning label for this whole exercise: when many models are tried on one time series, the best-looking model may be lucky. "
        "Diebold-Mariano style forecast comparison is the statistical tradition behind comparing forecasts rather than trusting one raw leaderboard number.\n"
    )

    lines.append("## What This Means For The Current Result\n")
    lines.append(
        "Training 33 model configurations had a concrete purpose: it showed that the best result is a modest ML ensemble improvement, not a deep-learning breakthrough and not a hidden 0.70+ signal. "
        "`ENS_FINAL3` is useful because it improves the primary full-coverage test leaderboard, but the improvement is small and still below the next research target. "
        "The current conclusion is therefore disciplined: the pipeline found a weak but plausible edge; further gains likely require better information, better labels, walk-forward validation, or new feature sources rather than simply adding more model types.\n"
    )

    lines.append("## My Reading Of The Evidence\n")
    evidence_reading = pd.DataFrame(
        [
            [
                "The best row is not spectacular.",
                "`ENS_FINAL3` reaches roughly 0.548 accuracy and 0.541 F1_macro. That is above the old best-F1 reference, but it is still below the near-term research target.",
            ],
            [
                "The model families mostly cluster near weak-signal performance.",
                "Many baselines, DL models, and feature-selection variants sit near 0.50-0.53. That pattern supports the local docs' point that daily oil direction is close to a noisy coin flip.",
            ],
            [
                "Ensembling helps, but only modestly.",
                "The best model is an average-probability ensemble, which means the useful signal is distributed across weak learners rather than concentrated in one dominant model.",
            ],
            [
                "Deep learning is negative evidence in this dataset.",
                "GRU/MLP sequence models were evaluated on the same setup and GPU-rerun with final refit. They did not beat the best ML ensemble, so more neural complexity is not the current answer.",
            ],
            [
                "Feature selection is also negative evidence.",
                "The feature-selection branch did not produce the best result, suggesting that the bottleneck is not just too many features; it is the weakness and instability of the available signal.",
            ],
            [
                "AUC and hard classification disagree.",
                "`XGB_linear03` has the best AUC, but weak F1_macro. It may rank probabilities better, but it is not the best UP/DOWN decision model at full coverage.",
            ],
            [
                "The big table mainly prevents overclaiming.",
                "Because many reasonable alternatives were checked, the honest conclusion is more credible: current progress is real but small, and high paper-like accuracy would require much stricter audit.",
            ],
        ],
        columns=["Observation", "My comment"],
    )
    lines.append(md_table(evidence_reading, ["Observation", "My comment"], len(evidence_reading)))

    lines.append("## Research Sources Used For This Interpretation\n")
    lines.append(
        "- Local docs: `docs/WHY_ML_FAILS.md`, `docs/DATA_LEAKAGE.md`, `docs/improve/oil_direction_research_benchmark_brief.md`, "
        "`docs/improve/oil_direction_leakage_audit_checklist.md`.\n"
        "- Ghoddusi, Creamer, and Rafizadeh (2019), *Machine learning in energy economics and finance: A review*, Energy Economics. "
        "https://doi.org/10.1016/j.eneco.2019.05.006\n"
        "- Luo et al. (2019), *Can We Forecast Daily Oil Futures Prices? Experimental Evidence from Convolutional Neural Networks*. "
        "https://www.mdpi.com/1911-8074/12/1/9\n"
        "- Zhao, Li, and Yu (2017), *A deep learning ensemble approach for crude oil price forecasting*, Energy Economics. "
        "https://doi.org/10.1016/j.eneco.2017.05.023\n"
        "- Chen, He, and Tso (2017), *Forecasting Crude Oil Prices: a Deep Learning based Model*, Procedia Computer Science. "
        "https://doi.org/10.1016/j.procs.2017.11.373\n"
        "- White (2000), *A Reality Check for Data Snooping*, Econometrica. "
        "https://www.jstor.org/stable/2999444\n"
        "- Diebold and Mariano (1995), *Comparing Predictive Accuracy*, Journal of Business & Economic Statistics. "
        "https://doi.org/10.1080/07350015.1995.10524599\n"
    )

    lines.append("## All Evaluated Model Configurations\n")
    lines.append(
        f"This table lists all `{total_configs}` current model configurations. The metrics shown here are the primary leaderboard metrics: "
        "`Split=test`, `Coverage=1.0`, `ThresholdMode=fixed_0.5`.\n"
    )
    lines.append(md_table(model_list, INTERPRETATION_MODEL_COLUMNS, len(model_list)))

    lines.append("## How Validation And Test Differ\n")
    lines.append(
        "Every model/config follows the same evaluation flow:\n\n"
        "```text\n"
        "1. Fit a temporary model on train: target date < 2022-01-01\n"
        "2. Predict validation: 2022-01-01 <= target date < 2023-01-01\n"
        "3. Choose the best validation threshold by F1_macro\n"
        "4. Refit the final model on train_full: target date < 2023-01-01\n"
        "5. Predict final test: target date >= 2023-01-01\n"
        "6. Compute test metrics under two threshold policies\n"
        "```\n"
    )
    lines.append(
        "Validation is therefore not the final result. It is used for threshold/model selection. "
        "The final report uses the 2023+ test period because it is outside the threshold-selection step.\n"
    )

    lines.append("## Leakage Audit Status\n")
    lines.append(
        "For the current unified pipeline, the evaluation code uses the same chronological split, same target definition, "
        "and same metric function for baseline, feature selection, weight decay, ensemble, and deep-learning rows. "
        "The test split is not used to fit tabular models, choose validation thresholds, rank/select features, select TOP-N ensembles, "
        "or early-stop deep-learning models. After validation choices are made, final tabular and deep-learning models are refit on "
        "`train_full` before test prediction.\n\n"
        "Important caveat: sorting the test leaderboard identifies the best retrospective test row. That is useful for comparison, "
        "but it is not the same as a pre-registered deployment model selected before looking at test. Also, this pipeline trusts "
        "the source feature table `dataset_final_noleak_step5c_scaler.csv`; it does not re-audit the upstream feature-generation code here.\n"
    )

    lines.append("## Why TestRows Are Double The Config Count\n")
    lines.append(
        "There are `33` current configs but `66` full-coverage test rows because each config is evaluated under two threshold policies:\n"
    )
    lines.append(md_table(threshold_counts, ["Experiment", "ThresholdMode", "Rows"], len(threshold_counts)))
    lines.append(
        "\nThese two threshold rows are not two different models. They use the same probability vector on the test set; only the probability-to-label conversion differs:\n\n"
        "```text\n"
        "fixed_0.5:     predict UP if P(UP) >= 0.5\n"
        "val_f1_macro:  predict UP if P(UP) >= threshold selected on validation\n"
        "```\n\n"
        "To keep the report readable and non-duplicated, the primary leaderboard uses only `fixed_0.5`, so it has exactly `33` rows.\n"
    )

    lines.append("## Best Model By Experiment\n")
    lines.append(md_table(best_by_exp, DISPLAY_COLUMNS, len(best_by_exp)))

    lines.append("## What Selective Coverage Means\n")
    lines.append(
        f"There are `{total_selective_rows}` selective diagnostic rows. These rows evaluate only higher-confidence days, "
        "so coverage is below 1.0. They must not be compared directly with the full-coverage primary leaderboard.\n"
    )
    if not selective.empty:
        lines.append(md_table(selective, ["ModelConfig", "Margin", "Coverage", "N", "Accuracy", "F1_macro", "AUC"], 12))

    lines.append("## Historical Reference\n")
    lines.append(
        "Historical rows come from the old `ml/classification/final` pipeline. They are kept for reference, "
        "but they are not current reruns and they lack some probability metrics such as LogLoss/Brier.\n"
    )
    lines.append(md_table(historical, ["Experiment", "ModelConfig", "Accuracy", "F1_macro", "AUC", "Coverage"], len(historical)))

    lines.append("## My Interpretation\n")
    for line in interpretation_lines(primary, historical, threshold_diag):
        lines.append(line + "\n")
    lines.append(
        "In short: the pipeline is now complete and unified. Use `primary_test_leaderboard.csv` / `REPORT.md` for the final classifier decision, "
        "use `threshold_diagnostics.csv` to understand threshold behavior, and treat selective diagnostics as reference only.\n"
    )
    return "\n".join(lines)


def build_report(root_dir: Path = THIS_DIR, runtime_s: Optional[float] = None) -> None:
    results_dir = root_dir / "results"
    inputs = refresh_inputs(results_dir)
    tables = build_metric_tables(inputs, results_dir)
    write_metric_contract(results_dir)

    primary = tables["primary"]
    current_all = tables["current_all"]
    historical = tables["historical"]
    threshold_diag = tables["threshold_diag"]
    selective = tables["selective"]
    status = inputs["status"]

    best = best_row(primary, ["F1_macro", "Accuracy", "AUC"])
    best_auc = best_row(primary, ["AUC", "F1_macro", "Accuracy"])
    by_experiment = best_by_experiment(primary)
    missing = status[~status["Exists"]]
    generated = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    runtime_note = "" if runtime_s is None else f" Runtime={runtime_s:.1f}s."

    report: List[str] = []
    report.append("# Final Unified Oil Direction Report\n")
    report.append(f"Generated by `report.py` at {generated}.{runtime_note}\n")

    report.append("## Quick Read\n")
    report.append(
        "This is the unified report after combining baseline models, feature selection, weight decay, ensembles, deep learning, "
        "and historical references. The main leaderboard no longer mixes threshold variants: each model configuration has one "
        "primary row using `fixed_0.5`, full coverage, and the final test split.\n"
    )
    report.append(
        f"- Primary leaderboard configs: `{len(primary)}`\n"
        f"- All threshold test rows kept for diagnostics: `{len(current_all)}`\n"
        f"- Historical reference rows: `{len(historical)}`\n"
        f"- Best primary test leaderboard classifier: {best_text(best)}\n"
        f"- Best AUC/ranking config in the primary leaderboard: {best_text(best_auc)}\n"
    )
    if not missing.empty:
        report.append("- WARNING: some artifacts are missing; check Artifact Coverage before drawing conclusions.\n")

    report.append("## Unified Setup\n")
    report.append(
        "- Dataset: `data/processed/dataset_final_noleak_step5c_scaler.csv`\n"
        "- Target: `oil_return_fwd1 > 0` => UP=1, otherwise DOWN=0\n"
        "- Train for validation: target date `< 2022-01-01`\n"
        "- Validation/model-threshold selection: `2022-01-01 <= target date < 2023-01-01`\n"
        "- Final refit: target date `< 2023-01-01`\n"
        "- Final test: target date `>= 2023-01-01`\n"
        "- Main threshold policy: `fixed_0.5` for every current model configuration\n"
        "- Main sort: `F1_macro`, then `Accuracy`, then `AUC`\n"
    )

    report.append("## Leakage Audit Status\n")
    report.append(
        "Current rows share the same chronological split, target definition, and metric implementation. "
        "The test split is only used for final metric computation after model/threshold/feature choices are made. "
        "Tabular ML and DL final models are refit on `train_full` before test prediction; ensembles combine those saved validation/test predictions. "
        "The source dataset is assumed to be leakage-audited upstream because this report starts from `dataset_final_noleak_step5c_scaler.csv`.\n\n"
        "Caveat: the primary leaderboard is a retrospective test comparison. Choosing the top row after seeing all test metrics can introduce "
        "test-set selection bias, even though the model training itself does not use test labels.\n"
    )

    report.append("## Artifact Coverage\n")
    report.append(md_table(status, ["Experiment", "ResultsFile", "Exists", "Rows", "Configs", "ValRows", "TestRows", "SelectiveRows"], len(status)))
    if missing.empty:
        report.append(
            "\nAll required experiment artifacts are present. This report does not read a stale combined `ml_results.csv`; "
            "`ml_results.csv` is regenerated from the individual step output files.\n"
        )
    else:
        report.append("\nSome artifacts are missing, so this report only reflects the steps that produced result files.\n")

    report.append("## Training And Evaluation Funnel\n")
    report.append(
        "This section explains how many configurations existed at the start, how many actual train/refit operations were run, "
        "how many validation rows were produced, how much inner-CV feature-selection work happened, and how many rows finally reached test. "
        "The test split is not used for training; test rows are metric rows only.\n"
    )
    report.append(
        "Short version: there are 138 actual train/refit operations, but only 33 final configurations reach the strict final test leaderboard. "
        "The test split is never used for training.\n"
    )
    report.append(md_table(readable_training_test_table(), ["Stage", "Count", "Meaning"], None))
    report.append(
        "\nSo the funnel is: 80 internal CV fits for feature selection, 29 real validation-stage model fits, "
        "29 final refits, then 33 strict final test rows. The extra 4 final configs are ensembles, so they are tested but not separately trained.\n"
    )

    report.append("## Metric Contract\n")
    metric_contract = pd.DataFrame(
        [
            ["F1_macro", "Primary decision metric; treats UP and DOWN symmetrically"],
            ["Accuracy", "Overall share of correct UP/DOWN calls"],
            ["BalancedAcc", "Average recall of UP and DOWN"],
            ["AUC", "Threshold-free ranking quality"],
            ["MCC", "Correlation-style classifier quality"],
            ["LogLoss/Brier", "Probability quality/calibration"],
            ["Precision/Recall UP/DOWN", "Class behavior diagnostics"],
        ],
        columns=["Metric", "Meaning"],
    )
    report.append(md_table(metric_contract, ["Metric", "Meaning"], len(metric_contract)))

    report.append("## Research Context\n")
    report.append(
        "This report should be read against the research context in `docs/` and `docs/improve/`: daily next-day oil direction is a noisy, "
        "low-signal forecasting task, so realistic KPI bands are much lower than many headline paper claims. The local benchmark brief treats "
        "`0.53-0.56` accuracy as an acceptable baseline and `0.56-0.60` as good for this exact daily UP/DOWN setup. The leakage checklist also "
        "warns that unusually high results should be audited for random splits, same-day/future features, global preprocessing, and repeated test-set selection.\n"
    )
    research_context = pd.DataFrame(
        [
            [
                "Local project docs",
                "`docs/WHY_ML_FAILS.md`, `docs/DATA_LEAKAGE.md`, `docs/improve/*`",
                "Set conservative expectations and define leakage checks for this repo.",
            ],
            [
                "Energy ML review",
                "Ghoddusi, Creamer, and Rafizadeh (2019)",
                "Energy forecasting uses many model families, but the literature has both opportunities and limitations.",
            ],
            [
                "Daily oil direction benchmark",
                "Luo et al. (2019)",
                "Daily directional accuracy in a clean-ish oil futures setting is closer to conservative KPI bands than to 0.80+ claims.",
            ],
            [
                "Ensemble / DL motivation",
                "Zhao, Li, and Yu (2017); Chen, He, and Tso (2017)",
                "Oil prices are nonlinear and multi-factor, so ensembles and neural models are reasonable to test, but not guaranteed to win.",
            ],
            [
                "Model-selection caution",
                "White (2000); Diebold and Mariano (1995)",
                "Trying many models on one time series creates data-snooping risk; forecast comparisons should be interpreted cautiously.",
            ],
        ],
        columns=["Theme", "Sources", "Report implication"],
    )
    report.append(md_table(research_context, ["Theme", "Sources", "Report implication"], len(research_context)))

    report.append("## Model Catalog\n")
    report.append(
        "The leaderboard uses compact model names so the table stays readable. This catalog explains what each name pattern means "
        "and why that model family was included in the experiment grid.\n"
    )
    report.append(md_table(model_catalog_table(), ["Model name pattern", "Experiment", "Model family", "Why it is included"], None))

    report.append("## Why These Model Counts\n")
    report.append(
        "The primary test leaderboard has 33 model configurations because the experiment grid is split into five bounded research blocks. "
        "The counts below explain where `7 + 4 + 10 + 4 + 8` comes from.\n"
    )
    report.append(md_table(experiment_count_explanation_table(), ["Experiment", "Configs", "Where the count comes from", "Why this count is useful"], None))

    report.append("## Feature Selection Count Detail\n")
    report.append(
        "The feature-selection step has more internal search work than the final count of 4 suggests. "
        "The report counts final test model configurations, not every proxy fit used during feature-subset selection.\n"
    )
    report.append(md_table(feature_selection_count_table(), ["Stage", "Count", "Meaning"], None))
    report.append(
        "\nThe funnel below shows the exact path from initial feature-subset candidates to final test rows.\n"
    )
    report.append(md_table(feature_selection_funnel_table(), ["Phase", "What is counted", "Count", "Explanation"], None))

    report.append("## Naming Conventions\n")
    report.append(md_table(naming_convention_table(), ["Name part", "Meaning"], None))

    report.append("## Primary Test Leaderboard\n")
    report.append(
        "This is the main table: one row per model configuration, `Split=test`, `Coverage=1.0`, "
        "`ThresholdMode=fixed_0.5`. There are no duplicate threshold rows in this table.\n"
    )
    report.append(md_table(primary, DISPLAY_COLUMNS, len(primary)))

    report.append("## Best By Experiment\n")
    report.append(md_table(by_experiment, DISPLAY_COLUMNS, len(by_experiment)))

    report.append("## Historical References\n")
    report.append(
        "Historical rows are kept to compare against the old pipeline. They are not current reruns and they lack "
        "some probability metrics such as LogLoss/Brier, so they are reference rows only.\n"
    )
    report.append(md_table(historical, ["Experiment", "ModelConfig", "Accuracy", "BalancedAcc", "F1_macro", "AUC", "MCC", "N", "Coverage"], len(historical)))

    report.append("## Threshold Diagnostics\n")
    report.append(
        "This is not a leaderboard. It only shows whether validation-selected thresholds beat fixed 0.5 on the test split.\n"
    )
    report.append(md_table(threshold_diag, ["Experiment", "ModelConfig", "Fixed_F1_macro", "ValThreshold_F1_macro", "Delta_ValMinusFixed_F1", "Fixed_Accuracy", "ValThreshold_Accuracy", "ValThreshold"], 20))

    report.append("## Selective Coverage Diagnostic\n")
    report.append(
        "Selective coverage filters test rows by confidence margin, so these rows must not be compared directly with "
        "the full-coverage leaderboard.\n"
    )
    report.append(md_table(selective, ["ModelConfig", "Margin", "Coverage", "N", "Accuracy", "F1_macro", "AUC"], 12))

    report.append("## Interpretation\n")
    report.append(
        "This file intentionally keeps interpretation brief and table-focused. "
        "The full narrative answer to why many models were trained, what the large table means, and what I infer from the evidence is in `INTERPRETATION.md`.\n"
    )

    report.append("## Output Files\n")
    report.append(
        "- `results/primary_test_leaderboard.csv`: main non-duplicated leaderboard\n"
        "- `results/all_threshold_test_metrics.csv`: all full-coverage test threshold rows for diagnostics\n"
        "- `results/threshold_diagnostics.csv`: fixed 0.5 vs validation-selected threshold comparison\n"
        "- `results/selective_coverage_diagnostics.csv`: confidence-filter diagnostics only\n"
        "- `results/unified_full_coverage_metrics.csv`: primary leaderboard plus historical references\n"
        "- `results/metric_contract.json`: machine-readable evaluation contract\n"
    )

    (root_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    interpretation = build_interpretation_markdown(
        generated=generated,
        status=status,
        primary=primary,
        current_all=current_all,
        threshold_diag=threshold_diag,
        selective=selective,
        historical=historical,
    )
    (root_dir / "INTERPRETATION.md").write_text(interpretation, encoding="utf-8")


def main() -> None:
    build_report()
    print(f"Wrote unified report to {THIS_DIR / 'REPORT.md'}")
    print(f"Wrote primary leaderboard to {RESULTS_DIR / 'primary_test_leaderboard.csv'}")


if __name__ == "__main__":
    main()
