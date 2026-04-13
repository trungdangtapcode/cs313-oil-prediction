"""
================================================================================
STEP 4B: FIX LEAKAGE IN TRANSFORMED DATA
================================================================================

Purpose:
  1. Shift same-day features that can leak information from day T
  2. Drop rows made invalid by shifting
  3. Export clean no-leakage datasets for downstream modeling

Input:
  - data/processed/dataset_step4_transformed.csv

Outputs:
  - data/processed/dataset_step4_noleak.csv
  - data/processed/dataset_final_noleak.csv

Leakage fixes applied here:
  - Shift same-day market returns by 1 day:
      * usd_return
      * sp500_return
      * vix_return
  - Shift oil_volatility_7d by 1 day
  - Keep target oil_return unchanged

Notes:
  - This script is meant to run after step4_transformation.py
  - It reuses the same reduction logic as step5_reduction.py to create a model-ready
    no-leakage dataset
"""

from pathlib import Path

import pandas as pd

from step5_reduction import create_model_ready_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data/processed"

INPUT_FILE = PROCESSED_DIR / "dataset_step4_transformed.csv"
OUTPUT_STEP4_NOLEAK = PROCESSED_DIR / "dataset_step4_noleak.csv"
OUTPUT_FINAL_NOLEAK = PROCESSED_DIR / "dataset_final_noleak.csv"

SHIFT_COLS = [
    "usd_return",
    "sp500_return",
    "vix_return",
    "oil_volatility_7d",
]


def load_step4_data() -> pd.DataFrame:
    """Load transformed dataset and sort by date."""
    print("\nLoading transformed dataset...")
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  Loaded shape: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def apply_leakage_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """Shift same-day leakage-prone features and drop invalid rows."""
    print("\nApplying no-leakage shifts...")
    df = df.copy()

    applied = []
    for col in SHIFT_COLS:
        if col in df.columns:
            df[col] = df[col].shift(1)
            applied.append(col)

    if applied:
        print(f"  Shifted columns: {applied}")
    else:
        print("  No configured columns found to shift")

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    print(f"  Dropped {dropped} rows after shifting")
    print(f"  New shape: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def save_outputs(df: pd.DataFrame) -> None:
    """Save full no-leak dataset and reduced model-ready dataset."""
    print("\nSaving no-leakage datasets...")
    df.to_csv(OUTPUT_STEP4_NOLEAK, index=False)
    print(f"  Saved full dataset: {OUTPUT_STEP4_NOLEAK}")

    df_model = create_model_ready_dataset(df)
    df_model.to_csv(OUTPUT_FINAL_NOLEAK, index=False)
    print(f"  Saved model-ready dataset: {OUTPUT_FINAL_NOLEAK}")


def main() -> None:
    print("\n" + "=" * 80)
    print("STEP 4B: FIX LEAKAGE IN TRANSFORMED DATA")
    print("=" * 80)

    df = load_step4_data()
    df_fixed = apply_leakage_fixes(df)
    save_outputs(df_fixed)

    print("\nDone.")


if __name__ == "__main__":
    main()
