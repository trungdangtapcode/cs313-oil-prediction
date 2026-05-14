"""
================================================================================
BƯỚC 1: LOAD & KIỂM TRA DỮ LIỆU THÔ (Load & Inspect Raw Data)
================================================================================

Mục tiêu:
  - Load 5 file CSV từ folder data/raw
  - Parse cột date về datetime
  - In ra statistic mô tả: shape, dtypes, date range, missing values, quantiles
  
Tương ứng slide: "Real data is noisy, incomplete and inconsistent" + Statistical Descriptions

Output:
  - In console (không lưu file)
  - Dùng để hiểu structure của dữ liệu trước khi xử lí
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data/raw"

MARKET_FILE = RAW_DIR / "market_data.csv"
FRED_FILE = RAW_DIR / "fred_data.csv"
EIA_FILE = RAW_DIR / "eia_data.csv"
GDELT_FILE = RAW_DIR / "gdelt_data.csv"
ACLED_FILE = RAW_DIR / "ACLED Data_2026-03-26.csv"


def load_data() -> dict[str, pd.DataFrame]:
    """Load 5 nguồn dữ liệu từ CSV"""
    data = {}
    
    try:
        data["market"] = pd.read_csv(MARKET_FILE)
        print(f"✓ Load market_data.csv")
    except FileNotFoundError:
        print(f"✗ Không tìm thấy {MARKET_FILE}")
    
    try:
        data["fred"] = pd.read_csv(FRED_FILE)
        print(f"✓ Load fred_data.csv")
    except FileNotFoundError:
        print(f"✗ Không tìm thấy {FRED_FILE}")
    
    try:
        data["eia"] = pd.read_csv(EIA_FILE)
        print(f"✓ Load eia_data.csv")
    except FileNotFoundError:
        print(f"✗ Không tìm thấy {EIA_FILE}")
    
    try:
        data["gdelt"] = pd.read_csv(GDELT_FILE)
        print(f"✓ Load gdelt_data.csv")
    except FileNotFoundError:
        print(f"✗ Không tìm thấy {GDELT_FILE}")
    
    try:
        data["acled"] = pd.read_csv(ACLED_FILE)
        print(f"✓ Load ACLED Data_2026-03-26.csv")
    except FileNotFoundError:
        print(f"✗ Không tìm thấy {ACLED_FILE}")
    
    return data


def parse_dates(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Parse cột date về datetime"""
    for name, df in data.items():
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        elif "event_date" in df.columns:
            df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
            df.rename(columns={"event_date": "date"}, inplace=True)
    
    return data


def inspect_source(name: str, df: pd.DataFrame):
    """In chi tiết thống kê của 1 nguồn dữ liệu"""
    print(f"\n{'='*80}")
    print(f"📊 NGUỒN: {name.upper()}")
    print(f"{'='*80}")
    
    # --- Shape & dtypes ---
    print(f"\n1️⃣  SHAPE & TYPES")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Data types:")
    for col, dtype in df.dtypes.items():
        print(f"      {col:30s} : {dtype}")
    
    # --- Date range ---
    if "date" in df.columns:
        date_col = "date"
    else:
        date_col = None
    
    if date_col:
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        print(f"\n2️⃣  DATE RANGE")
        print(f"   Min: {min_date}")
        print(f"   Max: {max_date}")
        print(f"   Span: {(max_date - min_date).days} days")
    
    # --- Missing values ---
    print(f"\n3️⃣  MISSING VALUES")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    
    for col in df.columns:
        if missing[col] > 0:
            print(f"   {col:30s}: {missing[col]:6d} ({missing_pct[col]:5.1f}%)")
    
    if missing.sum() == 0:
        print("   ✓ Du0 an toàn - không có missing values")
    
    # --- Descriptive statistics ---
    print(f"\n4️⃣  DESCRIPTIVE STATISTICS")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("   (No numeric columns)")
    else:
        print(df[numeric_cols].describe().to_string())
    
    # --- Duplicate check ---
    if date_col:
        duplicates = df[date_col].duplicated().sum()
        print(f"\n5️⃣  DUPLICATES")
        if duplicates > 0:
            print(f"   ⚠️  {duplicates} duplicate date(s) found!")
        else:
            print(f"   ✓ Không có date trùng lặp")


def main():
    print("\n" + "="*80)
    print("🔍 BƯỚC 1: LOAD & KIỂM TRA DỮ LIỆU THÔ")
    print("="*80 + "\n")
    
    # --- Load all data ---
    print("⏳ Loading data từ data/raw ...\n")
    data = load_data()
    
    if not data:
        print("\n❌ Không có file nào để load. Kiểm tra lại các file raw.")
        return
    
    # --- Parse dates ---
    print("\n⏳ Parsing date columns ...\n")
    data = parse_dates(data)
    
    # --- Inspect từng source ---
    for name in ["market", "fred", "eia", "gdelt", "acled"]:
        if name in data:
            inspect_source(name, data[name])
    
    # --- Summary ---
    print(f"\n{'='*80}")
    print("📋 TÓM TẮT")
    print(f"{'='*80}")
    
    total_missing = 0
    total_cells = 0
    for name, df in data.items():
        missing = df.isnull().sum().sum()
        total = df.shape[0] * df.shape[1]
        total_missing += missing
        total_cells += total
        pct = missing / total * 100 if total > 0 else 0
        print(f"{name:10s}: {df.shape[0]:6d} rows × {df.shape[1]:3d} cols | Missing: {missing:6d}/{total:7d} ({pct:5.1f}%)")
    
    print(f"\nTOTAL    : Missing: {total_missing}/{total_cells} ({total_missing/total_cells*100:.1f}%)")
    
    print(f"\n✅ Step 1 hoàn thành! Dữ liệu sẵn sàng cho Bước 2 (Cleaning)")


if __name__ == "__main__":
    main()
