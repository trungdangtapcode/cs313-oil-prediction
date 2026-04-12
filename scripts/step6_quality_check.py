"""
================================================================================
BƯỚC 6: QUALITY CHECK CUỐI (Final Data Quality Verification)
================================================================================

Mục tiêu:
  1. Kiểm tra NaN cuối cùng
  2. Kiểm tra data leakage
  3. In sample data
  4. Shape report

Input: data/processed/dataset_final.csv (từ step 5)
Output: In console (không lưu file)
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data/processed"

MODEL_FILE = PROCESSED_DIR / "dataset_final.csv"


def load_final_data() -> pd.DataFrame:
    """Load final dataset"""
    print("\n⏳ Loading dataset_final.csv ...")
    df = pd.read_csv(MODEL_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def check_nan_and_inf(df: pd.DataFrame) -> None:
    """Kiểm tra NaN và inf"""
    print(f"\n{'='*80}")
    print("🔍 CHECK 1: NaN & INF VALUES")
    print(f"{'='*80}")

    nan_count = df.isnull().sum()
    if nan_count.sum() > 0:
        print(f"\n⚠️  Found {nan_count.sum()} NaN values:")
        for col, count in nan_count[nan_count > 0].items():
            print(f"   {col:40s}: {count:6d} ({count/len(df)*100:5.1f}%)")
    else:
        print(f"\n✓ No NaN values found")

    # Check inf
    inf_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_cols.append((col, inf_count))

    if inf_cols:
        print(f"\n⚠️  Found {len(inf_cols)} columns with INF values:")
        for col, count in inf_cols:
            print(f"   {col:40s}: {count:6d}")
    else:
        print(f"\n✓ No INF values found")


def check_data_leakage(df: pd.DataFrame) -> None:
    """Kiểm tra data leakage"""
    print(f"\n{'='*80}")
    print("🔍 CHECK 2: DATA LEAKAGE DETECTION")
    print(f"{'='*80}")

    leakage_risks = []

    # Rule 1: vix_close, usd_close, sp500_close không được dùng trực tiếp cùng ngày
    future_cols = ["vix_close", "usd_close", "sp500_close"]
    has_future = [col for col in future_cols if col in df.columns]

    if has_future:
        print(f"\n⚠️  Found {len(has_future)} columns with same-day information:")
        for col in has_future:
            print(f"   {col} - MUST use {col}_lag1 instead")
        leakage_risks.extend(has_future)

    # Rule 2: gdet_volume_log không nên dùng trực tiếp nếu chưa lag
    if "gdelt_volume_log" in df.columns and "gdelt_volume_lag1" in df.columns:
        print(f"\n✓ gdelt_volume dùng lag version (gdelt_volume_lag1)")

    if not leakage_risks:
        print(f"\n✓ No obvious data leakage detected")


def check_target_distribution(df: pd.DataFrame) -> None:
    """Kiểm tra phân phối target variable"""
    print(f"\n{'='*80}")
    print("🔍 CHECK 3: TARGET VARIABLE DISTRIBUTION")
    print(f"{'='*80}")

    if "oil_return" not in df.columns:
        print("⚠️  oil_return not found")
        return

    target = df["oil_return"]
    valid = target.dropna()

    print(f"\nTarget: oil_return")
    print(f"  Valid values: {len(valid)}/{len(target)}")
    print(f"  Mean        : {valid.mean():.6f}")
    print(f"  Std         : {valid.std():.6f}")
    print(f"  Min         : {valid.min():.6f}")
    print(f"  Max         : {valid.max():.6f}")
    print(f"  Skewness    : {valid.skew():.3f}")
    print(f"  Kurtosis    : {valid.kurtosis():.3f}")

    # Check for fat tails
    if valid.kurtosis() > 5:
        print(f"  ⚠️  High kurtosis (fat tails) - typical for financial returns")


def print_sample_data(df: pd.DataFrame, n: int = 10) -> None:
    """In sample data"""
    print(f"\n{'='*80}")
    print(f"🔍 CHECK 4: SAMPLE DATA (first {n} rows)")
    print(f"{'='*80}\n")

    sample = df.head(n)
    print(sample.to_string())


def check_train_test_split(df: pd.DataFrame) -> None:
    """Kiểm tra train/test split"""
    print(f"\n{'='*80}")
    print("🔍 CHECK 5: TRAIN/TEST SPLIT")
    print(f"{'='*80}")

    train_date = pd.Timestamp("2023-01-01")
    train = df[df["date"] < train_date]
    test = df[df["date"] >= train_date]

    print(f"\nTrain set (< {train_date.date()}):")
    print(f"  Rows : {len(train)}")
    print(f"  Date range: {train['date'].min().date()} → {train['date'].max().date()}")

    print(f"\nTest set (>= {train_date.date()}):")
    print(f"  Rows : {len(test)}")
    print(f"  Date range: {test['date'].min().date()} → {test['date'].max().date()}")

    print(f"\nTotal: {len(train)} + {len(test)} = {len(df)} rows")
    print(f"Train/Test ratio: {len(train)/len(df)*100:.1f}% / {len(test)/len(df)*100:.1f}%")


def main():
    print("\n" + "="*80)
    print("✅ BƯỚC 6: QUALITY CHECK CUỐI")
    print("="*80)

    # --- Load ---
    df = load_final_data()

    # --- Checks ---
    check_nan_and_inf(df)
    check_data_leakage(df)
    check_target_distribution(df)
    check_train_test_split(df)
    print_sample_data(df, n=10)

    # --- Final report ---
    print(f"\n{'='*80}")
    print("📊 FINAL REPORT")
    print(f"{'='*80}")

    print(f"\nDataset Info:")
    print(f"  Shape : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  Columns: {list(df.columns)}")

    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
    print(f"  Missing: {missing_pct:.2f}%")

    print(f"\n✅ Quality check completed!")
    print(f"\nReady for:")
    print(f"  1. EDA (Exploratory Data Analysis)")
    print(f"  2. Model training")


if __name__ == "__main__":
    main()
