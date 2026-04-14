"""
================================================================================
STEP 4B: CREATE NO-LEAK DATASET BY DROPPING CONTAMINATED FEATURES
================================================================================

This step creates conservative "no-leak" exports from dataset_step4_transformed.csv.
Instead of trying to re-time monthly releases inside the already-built step4 dataset,
it removes columns that are known to be contaminated by:

  1. Monthly release-timing leakage (FRED monthly series exposed too early)
  2. Derived-feature inheritance from those leaky monthly series
  3. Split leakage from transforms fit on train+validation together
  4. Full-series preprocessing leakage (global winsorization cap)

Outputs:
  - data/processed/dataset_step4_noleak.csv
  - data/processed/dataset_final_noleak.csv
  - data/processed/dataset_step4_noleak_drop_report.csv
"""

from pathlib import Path

import pandas as pd

from step5_reduction import create_model_ready_dataset

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data/processed"

INPUT_STEP4 = PROCESSED_DIR / "dataset_step4_transformed.csv"
OUTPUT_STEP4_NOLEAK = PROCESSED_DIR / "dataset_step4_noleak.csv"
OUTPUT_FINAL_NOLEAK = PROCESSED_DIR / "dataset_final_noleak.csv"
OUTPUT_REPORT = PROCESSED_DIR / "dataset_step4_noleak_drop_report.csv"


LEAKAGE_REASONS = {
    "cpi_lag": {
        "category": "release_timing",
        "why": (
            "Monthly CPI is shifted to day 1 of the next month and forward-filled. "
            "That makes the new CPI value visible before the real CPI release date."
        ),
    },
    "unemployment_lag": {
        "category": "release_timing",
        "why": (
            "Monthly unemployment is shifted to day 1 of the next month and forward-filled. "
            "That exposes the new value before the Employment Situation release date."
        ),
    },
    "fed_funds_rate_lag": {
        "category": "release_timing",
        "why": (
            "Monthly Fed Funds is also shifted to day 1 of the next month and forward-filled. "
            "This is a timing-risk column because availability is assumed too early."
        ),
    },
    "cpi_yoy": {
        "category": "derived_from_release_timing",
        "why": (
            "Built from CPI values that already appear too early in the daily timeline, "
            "so the derived YoY inflation feature inherits the same timestamp leakage."
        ),
    },
    "real_rate": {
        "category": "derived_from_release_timing",
        "why": (
            "Computed from fed_funds_rate_lag and cpi_yoy, so it inherits the early-availability "
            "problem from those monthly macro inputs."
        ),
    },
    "fed_rate_change": {
        "category": "derived_from_release_timing",
        "why": (
            "Diff of fed_funds_rate_lag. If the monthly rate is visible too early, "
            "the change feature is also visible too early."
        ),
    },
    "fed_rate_regime": {
        "category": "derived_from_release_timing",
        "why": (
            "Rule-based label derived from fed_funds_rate_lag. It is not independent and inherits "
            "the same monthly timing problem."
        ),
    },
    "stress_tone": {
        "category": "split_leakage",
        "why": (
            "MinMax scaling for stress features is fit on all rows before 2023-01-01, which includes "
            "the 2022 validation period. That leaks validation distribution into the training-era feature."
        ),
    },
    "stress_volume": {
        "category": "split_leakage",
        "why": (
            "Same issue as stress_tone: scaling is fit on pre-2023 rows, not on the pure training window, "
            "so validation information influences the transformed feature."
        ),
    },
    "stress_goldstein": {
        "category": "split_leakage",
        "why": (
            "Same issue as the other stress components: transformation is fit on train+validation together, "
            "not training only."
        ),
    },
    "geopolitical_stress_index": {
        "category": "split_leakage",
        "why": (
            "Weighted combination of stress_tone, stress_volume, and stress_goldstein after those components "
            "were scaled with train+validation together."
        ),
    },
    "oil_volatility_7d": {
        "category": "global_preprocessing",
        "why": (
            "Winsorized with a 99th-percentile cap computed on the full series. "
            "That means early rows are clipped using future distribution information."
        ),
    },
}


def load_step4() -> pd.DataFrame:
    print(f"\n⏳ Loading {INPUT_STEP4.name} ...")
    df = pd.read_csv(INPUT_STEP4)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "oil_return_fwd1_date" in df.columns:
        df["oil_return_fwd1_date"] = pd.to_datetime(df["oil_return_fwd1_date"])
    return df


def build_drop_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, meta in LEAKAGE_REASONS.items():
        if col in df.columns:
            rows.append(
                {
                    "column": col,
                    "category": meta["category"],
                    "why": meta["why"],
                }
            )
    return pd.DataFrame(rows)


def drop_leakage_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    report = build_drop_report(df)
    drop_cols = report["column"].tolist()

    print("\n🔎 Leakage columns to drop:")
    for row in report.itertuples(index=False):
        print(f"  - {row.column} [{row.category}]")
        print(f"    {row.why}")

    cleaned = df.drop(columns=drop_cols, errors="ignore").copy()
    return cleaned, report


def main() -> None:
    print("\n" + "=" * 80)
    print("STEP 4B: CREATE NO-LEAK DATASET")
    print("=" * 80)
    print("\nThis step removes columns that are already contaminated by leakage.")
    print("It does not attempt to re-time monthly releases inside the existing step4 file.")

    df = load_step4()
    df_noleak, report = drop_leakage_columns(df)

    print(f"\n⏳ Saving step4 no-leak export to {OUTPUT_STEP4_NOLEAK.name} ...")
    df_noleak.to_csv(OUTPUT_STEP4_NOLEAK, index=False)

    print(f"⏳ Saving drop report to {OUTPUT_REPORT.name} ...")
    report.to_csv(OUTPUT_REPORT, index=False)

    print("\n⏳ Building model-ready no-leak export ...")
    df_final_noleak = create_model_ready_dataset(df_noleak)
    df_final_noleak.to_csv(OUTPUT_FINAL_NOLEAK, index=False)

    print(f"\n{'=' * 80}")
    print(" STEP 4B REPORT")
    print(f"{'=' * 80}")
    print(f"Input step4 shape        : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Step4 no-leak shape      : {df_noleak.shape[0]} rows × {df_noleak.shape[1]} cols")
    print(f"Final no-leak shape      : {df_final_noleak.shape[0]} rows × {df_final_noleak.shape[1]} cols")
    print(f"Dropped leakage columns  : {len(report)}")
    print(f"Step4 no-leak file       : {OUTPUT_STEP4_NOLEAK}")
    print(f"Final no-leak file       : {OUTPUT_FINAL_NOLEAK}")
    print(f"Drop report              : {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
