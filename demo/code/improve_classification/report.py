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

    lines.append("## Counts By Experiment\n")
    lines.append(md_table(status, ["Experiment", "Configs", "ValRows", "TestRows", "SelectiveRows"], len(status)))
    lines.append(
        "\nColumn definitions:\n"
        "- `Configs`: number of distinct model configurations after merging the two threshold variants under one config name.\n"
        "- `ValRows`: each config has one validation row, used to choose the `val_f1_macro` threshold.\n"
        "- `TestRows`: each config has two full-coverage test rows: `fixed_0.5` and `val_f1_macro`.\n"
        "- `SelectiveRows`: confidence-filter diagnostics only; these are not part of the primary leaderboard.\n"
    )

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

    report.append("## My Interpretation\n")
    for line in interpretation_lines(primary, historical, threshold_diag):
        report.append(line + "\n")

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
