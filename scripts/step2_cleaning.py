"""
================================================================================
BƯỚC 2: DATA CLEANING (Xử lý Missing Values & Noise)
================================================================================

Mục tiêu:
  1. Xử lý missing values theo từng nguồn (ffill với limit)
  2. Shift FRED monthly data (publication lag)
  3. Xử lý ACLED weekend problem
  4. Xóa duplicates, sắp xếp theo date
  5. Winsorize noise (oil_volatility_7d sẽ tính ở step 4, nhưng winsorize ở đây)

Tương ứng slide: "Fill in Missing Values", "Smooth Out Noise", "Correct Inconsistencies"

Input: data/raw/*.csv
Output: data/processed/dataset_step2_cleaned.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = BASE_DIR / "data/processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MARKET_FILE = RAW_DIR / "market_data.csv"
FRED_FILE = RAW_DIR / "fred_data.csv"
EIA_FILE = RAW_DIR / "eia_data.csv"
GDELT_FILE = RAW_DIR / "gdelt_data.csv"
ACLED_FILE = RAW_DIR / "ACLED Data_2026-03-26.csv"
OUTPUT_FILE = PROCESSED_DIR / "dataset_step2_cleaned.csv"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load 5 nguồn CSV"""
    market = pd.read_csv(MARKET_FILE)
    fred = pd.read_csv(FRED_FILE)
    eia = pd.read_csv(EIA_FILE)
    gdelt = pd.read_csv(GDELT_FILE)
    acled = pd.read_csv(ACLED_FILE)

    # Parse date column
    for df in [market, fred, eia, gdelt, acled]:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "event_date" in df.columns:
            df["date"] = pd.to_datetime(df["event_date"])

    return market, fred, eia, gdelt, acled


# ============================================================================
# BƯỚC 2A: XỬ LÝ MISSING VALUES THEO TỪNG NGUỒN
# ============================================================================

def clean_gdelt(gdelt: pd.DataFrame) -> pd.DataFrame:
    """
    GDELT Cleaning:
      - Flag các ngày bị thiếu dữ liệu gốc (gdelt_tone NaN)
      - ffill(limit=3) để tránh sâu quá xa
    """
    print("\n  → GDELT cleaning...")
    gdelt = gdelt.sort_values("date").copy()

    # Flag: 1 = ngày bị impute, 0 = có dữ liệu thật
    gdelt["gdelt_data_imputed"] = gdelt["gdelt_tone"].isna().astype(int)

    # ffill missing values với giới hạn 3 ngày
    for col in ["gdelt_tone", "gdelt_goldstein", "gdelt_volume", "gdelt_events"]:
        if col in gdelt.columns:
            gdelt[col] = gdelt[col].ffill(limit=3)

    return gdelt


def clean_acled(acled: pd.DataFrame) -> pd.DataFrame:
    """
    ACLED Cleaning - Xử lý Weekend Problem:
      - ACLED có sự kiện cả thứ 7, CN
      - Thị trường tài chính chỉ giao dịch Thứ 2-5
      - Giải pháp: Gộp sự kiện cuối tuần vào thứ 2 tuần sau (next trading day)
    
    Ví dụ:
        Thứ Bảy (dow=4) + 3 ngày → Thứ 2
        Chủ Nhật (dow=6) + 1 ngày → Thứ 2
    """
    print("\n  → ACLED cleaning (weekend aggregation)...")
    acled = acled.sort_values("date").copy()

    # Tổng hợp theo ngày: count event + sum fatalities
    daily_acled = acled.groupby("date").agg({
        "event_id_cnty": "count",
        "fatalities": "sum"
    }).reset_index()
    daily_acled.columns = ["date", "conflict_event_count", "fatalities"]

    # Xác định ngày trong tuần (0=Mon, ..., 6=Sun)
    daily_acled["day_of_week"] = daily_acled["date"].dt.dayofweek

    # Map ngày không giao dịch sang ngày giao dịch tiếp theo
    def map_to_trading_day(row):
        date = row["date"]
        dow = row["day_of_week"]

        if dow == 4:  # Friday → Monday next week (+3 days)
            return date + pd.Timedelta(days=3)
        elif dow == 5:  # Saturday → Monday next week (+2 days)
            return date + pd.Timedelta(days=2)
        elif dow == 6:  # Sunday → Monday next week (+1 day)
            return date + pd.Timedelta(days=1)
        else:
            return date  # Mon-Thu, keep as is

    daily_acled["date"] = daily_acled.apply(map_to_trading_day, axis=1)

    # Gộp các event vào cùng ngày trading
    daily_acled = daily_acled.groupby("date").agg({
        "conflict_event_count": "sum",
        "fatalities": "sum"
    }).reset_index()

    # Drop helper column
    daily_acled = daily_acled[["date", "conflict_event_count", "fatalities"]]

    return daily_acled


def clean_fred(fred: pd.DataFrame) -> pd.DataFrame:
    """
    FRED Cleaning:
      - FRED monthly data được shift 1 tháng ở bước crawl để tránh data leakage
      - (Vì CPI tháng T chỉ công bố tháng T+1, không thể dùng ngay)
      - Ở đây chỉ cần rename để rõ ý nghĩa: fed_funds_rate_lag, cpi_lag, v.v.
    
    Lưu ý: Phần ffill về daily ĐÃ LÀM ở crawl_macro_supply.py,
           khi merge với market index (business days) sẽ tự động align
    """
    print("\n  → FRED cleaning...")
    fred = fred.sort_values("date").copy()

    # Rename để rõ là lag data (đã shift ở crawl)
    monthly_cols = ["fed_funds_rate", "cpi", "unemployment"]
    for col in monthly_cols:
        if col in fred.columns:
            fred[f"{col}_lag"] = fred[col]
            fred = fred.drop(columns=[col])

    return fred


def clean_eia(eia: pd.DataFrame) -> pd.DataFrame:
    """
    EIA Cleaning:
      - EIA weekly data đã được ffill về daily ở crawl_macro_supply.py
      - Ở đây không cần xử lí thêm (chỉ để đúng structure)
    """
    print("\n  → EIA cleaning...")
    eia = eia.sort_values("date").copy()
    return eia


def clean_market(market: pd.DataFrame) -> pd.DataFrame:
    """
    Market Cleaning:
      - Xóa duplicate date
      - Sắp xếp theo date
      - ffill oil_close trước khi tính return ở step 4 (tránh bug NaN pair)
    """
    print("\n  → Market cleaning...")
    market = market.sort_values("date").copy()

    # Kiểm tra duplicate
    if market["date"].duplicated().any():
        print(f"      ⚠️  Found {market['date'].duplicated().sum()} duplicate dates - removing")
        market = market.drop_duplicates(subset=["date"], keep="first")

    # ffill oil_close nếu có NaN (để chuẩn bị cho step 4)
    market["oil_close"] = market["oil_close"].ffill()

    return market


# ============================================================================
# BƯỚC 2B: KIỂM TRA INCONSISTENCY & DEDUP
# ============================================================================

def ensure_business_days_only(df: pd.DataFrame, start_date: str = "2015-01-01", end_date: str = "2026-03-20") -> pd.DataFrame:
    """
    Lọc chỉ các Business Days (Mon-Fri) để đồng nhất với market data
    """
    # Tạo business day index
    business_days = pd.date_range(start=start_date, end=end_date, freq="B")
    
    # Filter df để chỉ giữ business days
    df = df[df["date"].isin(business_days)].copy()
    df = df.sort_values("date")
    
    return df


# ============================================================================
# MAIN CLEANING FLOW
# ============================================================================

def main():
    print("\n" + "="*80)
    print("🧹 BƯỚC 2: DATA CLEANING")
    print("="*80)

    # --- Load ---
    print("\n⏳ Loading data từ data/raw ...\n")
    market, fred, eia, gdelt, acled = load_data()

    # --- Clean from mỗi source ---
    print("⏳ Cleaning từng nguồn ...\n")
    fred = clean_fred(fred)
    gdelt = clean_gdelt(gdelt)
    acled = clean_acled(acled)
    eia = clean_eia(eia)
    market = clean_market(market)

    print(f"\n✓ Cleaning từng nguồn xong")

    # --- Sync về business days ---
    print("\n⏳ Syncing về business days only...")
    market = ensure_business_days_only(market)
    fred = ensure_business_days_only(fred)
    eia = ensure_business_days_only(eia)
    gdelt = ensure_business_days_only(gdelt)
    acled = ensure_business_days_only(acled)
    
    print("✓ Sync xong")

    # --- Merge tạm thời để chọn output ---
    # (Merge hoàn toàn sẽ làm ở step 3, nhưng lưu tạm ở đây)
    print("\n⏳ Tạo output file...")

    # Vì ở bước này chỉ cleaning, chưa merge.
    # Lưu tất cả 5 DataFrame vào 1 file dạng pickle hoặc lưu riêng?
    # → Lưu vào 1 CSV merged (market base) để step 3 load tiếp

    df_base = market.copy()
    df_base = df_base.merge(fred, on="date", how="left")
    df_base = df_base.merge(eia, on="date", how="left")
    df_base = df_base.merge(gdelt, on="date", how="left")
    df_base = df_base.merge(acled, on="date", how="left")

    # Fill post-merge (ffill tạm để tránh quá nhiều NaN)
    fill_cols = [col for col in df_base.columns if col != "date"]
    df_base[fill_cols] = df_base[fill_cols].ffill(limit=3)

    # Lưu file
    df_base.to_csv(OUTPUT_FILE, index=False)

    # --- Report ---
    print(f"\n{'='*80}")
    print("📊 REPORT BƯỚC 2")
    print(f"{'='*80}")
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"  Shape    : {df_base.shape[0]} rows × {df_base.shape[1]} cols")
    print(f"  Columns  : {list(df_base.columns)}")

    missing = df_base.isnull().sum().sum()
    total = df_base.shape[0] * df_base.shape[1]
    pct = missing / total * 100 if total > 0 else 0
    print(f"  Missing  : {missing}/{total} ({pct:.1f}%)")

    print(f"\n✅ Step 2 hoàn thành! Dữ liệu sạch sẽ đầu vào cho Bước 3 (Integration)")


if __name__ == "__main__":
    main()
