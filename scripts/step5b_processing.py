"""
================================================================================
STEP 5B: CANONICAL LEAKAGE-SAFE FEATURE PROCESSING
================================================================================

This is the canonical processed dataset step after step4b_fix_leakage.py.

It creates a processed variant of dataset_final_noleak.csv using only
deterministic, leakage-safe transforms:

  - cyclical encoding for calendar features
  - log1p for heavy-tailed positive features
  - signed log1p for selected heavy-tailed signed-change features

It intentionally does NOT fit StandardScaler / RobustScaler / PowerTransformer on
the full dataset because those should be fit on the training split only inside the
model pipeline.

Guidance:
  - Treat this file as the final deterministic processing step for the dataset.
  - Do NOT create a separate "step5c" data variant unless the transforms actually
    change.
  - Train-time preprocessing groups live in ml/model_preprocessing.py.

Outputs:
  - data/processed/dataset_final_noleak_processed.csv
  - data/processed/dataset_final_noleak_processing_report.csv
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data/processed"

INPUT_FILE = PROCESSED_DIR / "dataset_final_noleak.csv"
OUTPUT_FILE = PROCESSED_DIR / "dataset_final_noleak_processed.csv"
REPORT_FILE = PROCESSED_DIR / "dataset_final_noleak_processing_report.csv"

TARGET_COLS = ["date", "oil_return_fwd1", "oil_return_fwd1_date"]

# Business-day calendar in this project only runs Monday-Friday -> period 5.
CYCLE_SPECS = {
    "day_of_week": 5,
    "month": 12,
}

LOG1P_COLS = [
    "gdelt_events",
    "conflict_event_count",
    "conflict_intensity_7d",
    "fatalities",
    "fatalities_7d",
    "gdelt_volume_lag1",
    "vix_lag1",
]

SIGNED_LOG1P_COLS = [
    "net_imports_change_pct",
    "production_change_pct",
    "vix_return",
]

RECOMMENDED_SCALERS = {
    "yield_spread": "StandardScaler",
    "inventory_change_pct": "RobustScaler",
    "gdelt_goldstein": "StandardScaler",
    "gdelt_goldstein_7d": "StandardScaler",
    "oil_return": "RobustScaler",
    "usd_return": "StandardScaler",
    "sp500_return": "RobustScaler",
    "inventory_zscore": "already_zscore_or_StandardScaler",
    "oil_return_lag1": "RobustScaler",
    "oil_return_lag2": "RobustScaler",
    "gdelt_tone_7d": "PowerTransformer(Yeo-Johnson)",
    "gdelt_tone_30d": "PowerTransformer(Yeo-Johnson)",
    "gdelt_tone_lag1": "PowerTransformer(Yeo-Johnson)",
}


def signed_log1p(series: pd.Series) -> pd.Series:
    return np.sign(series) * np.log1p(np.abs(series))


def load_data() -> pd.DataFrame:
    print(f"\n⏳ Loading {INPUT_FILE.name} ...")
    df = pd.read_csv(INPUT_FILE)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "oil_return_fwd1_date" in df.columns:
        df["oil_return_fwd1_date"] = pd.to_datetime(df["oil_return_fwd1_date"])
    return df


def process_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    processed = df.copy()
    rows = []
    produced_cols = set()

    # 1) Cyclical encoding for calendar features.
    for col, period in CYCLE_SPECS.items():
        if col not in processed.columns:
            continue
        base = processed[col]
        phase = base if col == "day_of_week" else (base - 1)
        sin_col = f"{col}_sin"
        cos_col = f"{col}_cos"
        processed[sin_col] = np.sin(2 * np.pi * phase / period)
        processed[cos_col] = np.cos(2 * np.pi * phase / period)
        processed.drop(columns=[col], inplace=True)
        produced_cols.update({sin_col, cos_col})
        rows.append(
            {
                "source_feature": col,
                "output_feature": f"{sin_col}, {cos_col}",
                "transform": "cyclical_encoding",
                "skew_before": float(base.skew()),
                "skew_after": np.nan,
                "recommended_model_time_scaler": "none_or_optional_standard",
                "why": f"Calendar feature encoded as sin/cos with period={period}.",
            }
        )

    # 2) log1p for heavy-tailed positive features.
    for col in LOG1P_COLS:
        if col not in processed.columns:
            continue
        base = processed[col]
        out_col = f"{col}_log1p"
        processed[out_col] = np.log1p(base.clip(lower=0))
        processed.drop(columns=[col], inplace=True)
        produced_cols.add(out_col)
        rows.append(
            {
                "source_feature": col,
                "output_feature": out_col,
                "transform": "log1p",
                "skew_before": float(base.skew()),
                "skew_after": float(processed[out_col].skew()),
                "recommended_model_time_scaler": "StandardScaler_or_RobustScaler",
                "why": "Positive heavy-tail feature compressed with log1p.",
            }
        )

    # 3) signed log1p for selected signed change features.
    for col in SIGNED_LOG1P_COLS:
        if col not in processed.columns:
            continue
        base = processed[col]
        out_col = f"{col}_slog1p"
        processed[out_col] = signed_log1p(base)
        processed.drop(columns=[col], inplace=True)
        produced_cols.add(out_col)
        rows.append(
            {
                "source_feature": col,
                "output_feature": out_col,
                "transform": "signed_log1p",
                "skew_before": float(base.skew()),
                "skew_after": float(processed[out_col].skew()),
                "recommended_model_time_scaler": "RobustScaler",
                "why": "Signed heavy-tail feature compressed without losing direction.",
            }
        )

    untouched = [c for c in processed.columns if c not in TARGET_COLS and c not in produced_cols]
    for col in untouched:
        rows.append(
            {
                "source_feature": col,
                "output_feature": col,
                "transform": "keep",
                "skew_before": float(processed[col].skew()) if pd.api.types.is_numeric_dtype(processed[col]) else np.nan,
                "skew_after": float(processed[col].skew()) if pd.api.types.is_numeric_dtype(processed[col]) else np.nan,
                "recommended_model_time_scaler": RECOMMENDED_SCALERS.get(col, "model_dependent_or_none"),
                "why": "Kept as-is in dataset. Any fit-based scaler should be applied inside the model pipeline only.",
            }
        )

    report = pd.DataFrame(rows).sort_values(["transform", "source_feature"]).reset_index(drop=True)

    # Keep target/date columns first for readability.
    leading = [c for c in TARGET_COLS if c in processed.columns]
    trailing = [c for c in processed.columns if c not in leading]
    processed = processed[leading + trailing]
    return processed, report


def main() -> None:
    print("\n" + "=" * 80)
    print("STEP 5B: NO-LEAK FEATURE PROCESSING")
    print("=" * 80)
    print("\nThis step applies only deterministic transforms.")
    print("Fit-based scalers are intentionally deferred to the training pipeline.")

    df = load_data()
    processed, report = process_data(df)

    print(f"\n⏳ Saving processed dataset to {OUTPUT_FILE.name} ...")
    processed.to_csv(OUTPUT_FILE, index=False)

    print(f"⏳ Saving processing report to {REPORT_FILE.name} ...")
    report.to_csv(REPORT_FILE, index=False)

    print(f"\n{'=' * 80}")
    print("📊 STEP 5B REPORT")
    print(f"{'=' * 80}")
    print(f"Input shape        : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Processed shape    : {processed.shape[0]} rows × {processed.shape[1]} cols")
    print(f"Output dataset     : {OUTPUT_FILE}")
    print(f"Processing report  : {REPORT_FILE}")

    changed = report[report["transform"] != "keep"]
    print("\nTransforms applied:")
    for row in changed.itertuples(index=False):
        print(f"  - {row.source_feature} -> {row.output_feature} [{row.transform}]")


if __name__ == "__main__":
    main()
