"""
================================================================================
BƯỚC 3: DATA INTEGRATION (Gộp nhiều nguồn)
================================================================================

Mục tiêu:
  1. Re-merge với proper business day index
  2. Tạo daily timeline đầy đủ
  3. Join các nguồn lại
  4. Kiểm tra redundancy (wti_fred vs oil_close)

Tương ứng slide: "Data Integration: merging of data from multiple data stores",
                 "Entity Identification Problem", "Redundancy and Correlation Analysis"

Input: data/processed/dataset_step2_cleaned.csv
Output: data/processed/dataset_step3_integrated.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = BASE_DIR / "data/processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STEP2_FILE = PROCESSED_DIR / "dataset_step2_cleaned.csv"
OUTPUT_FILE = PROCESSED_DIR / "dataset_step3_integrated.csv"

START_DATE = "2015-01-01"
END_DATE = "2026-03-20"


def load_cleaned_data() -> pd.DataFrame:
    """Load dữ liệu đã clean từ step 2"""
    print("\n⏳ Loading dataset_step2_cleaned.csv ...")
    df = pd.read_csv(STEP2_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def create_business_day_index(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Tạo business day index chuẩn để align tất cả dữ liệu"""
    print(f"\n⏳ Creating business day index ({start_date} → {end_date}) ...")
    bday_index = pd.date_range(start=start_date, end=end_date, freq="B")
    print(f"   ✓ {len(bday_index)} business days")
    return bday_index


def reindex_and_fill(df: pd.DataFrame, bday_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Re-index dataframe vào business day index
    Forward fill (limit=3) cho các gaps
    """
    print(f"\n⏳ Re-indexing {len(df)} rows → {len(bday_index)} business days ...")
    
    # Set date as index
    df = df.set_index("date").sort_index()
    
    # Re-index vào business day index
    df = df.reindex(bday_index)
    
    # Forward fill (limit=3) những cột không phải returns
    fill_cols = [col for col in df.columns if col not in ["oil_return", "usd_return", "sp500_return", "vix_return"]]
    df[fill_cols] = df[fill_cols].ffill(limit=3)
    
    # Reset index để date thành cột
    df = df.reset_index()
    df.rename(columns={"index": "date"}, inplace=True)
    
    return df


def check_redundancy(df: pd.DataFrame) -> None:
    """
    Kiểm tra redundancy giữa các cột tương tự
    Ví dụ: wti_fred vs oil_close
    """
    print(f"\n⏳ Checking redundancy...")
    
    # wti_fred vs oil_close
    if "wti_fred" in df.columns and "oil_close" in df.columns:
        valid = df[["wti_fred", "oil_close"]].dropna()
        if len(valid) > 0:
            corr = valid["wti_fred"].corr(valid["oil_close"])
            print(f"\n   Correlation wti_fred vs oil_close: {corr:.4f}")
            if corr > 0.98:
                print(f"   ⚠️  Very high correlation (>0.98) - wti_fred can be dropped")
            else:
                print(f"   ✓ Moderate correlation - keep both")
        else:
            print(f"   (Not enough data to compute correlation)")
    
    # Check FRED lag columns redundancy
    fed_cols = [col for col in df.columns if "fed_funds_rate" in col]
    cpi_cols = [col for col in df.columns if "cpi" in col]
    
    if len(fed_cols) > 1:
        print(f"\n   FRED Fed Funds Rate cols: {fed_cols}")
    if len(cpi_cols) > 1:
        print(f"   FRED CPI cols: {cpi_cols}")


def main():
    print("\n" + "="*80)
    print("🔗 BƯỚC 3: DATA INTEGRATION")
    print("="*80)

    # --- Load cleaned data ---
    df = load_cleaned_data()

    # --- Create business day index ---
    bday_index = create_business_day_index(START_DATE, END_DATE)

    # --- Re-index và fill ---
    df = reindex_and_fill(df, bday_index)

    # --- Check redundancy ---
    check_redundancy(df)

    # --- Save ---
    print(f"\n⏳ Saving output to {OUTPUT_FILE} ...")
    df.to_csv(OUTPUT_FILE, index=False)

    # --- Report ---
    print(f"\n{'='*80}")
    print("📊 REPORT BƯỚC 3")
    print(f"{'='*80}")
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"  Shape    : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")

    missing = df.isnull().sum().sum()
    total = df.shape[0] * df.shape[1]
    pct = missing / total * 100 if total > 0 else 0
    print(f"  Missing  : {missing}/{total} ({pct:.1f}%)")

    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            print(f"  {i:2d}. {col:35s} ({nan_count:6d} NaN)")
        else:
            print(f"  {i:2d}. {col:35s}")

    print(f"\n✅ Step 3 hoàn thành! Dữ liệu hội nhập sẵn cho Bước 4 (Transformation)")


if __name__ == "__main__":
    main()
