"""
================================================================================
BƯỚC 4: DATA TRANSFORMATION (Tạo Features & Chuẩn hóa)
================================================================================

Mục tiêu:
  1. Tính Returns (oil, usd, sp500, vix)
  2. Rolling Windows aggregation (7d, 30d means)
  3. Derived Features từ FRED, EIA, GDELT, ACLED
  4. Cross-source Geopolitical Stress Index
  5. Lag Features (t-1, t-2)
  6. Time Features (day_of_week, month)
  7. Winsorize noise
  8. Final cleanup

Tương ứng slide: "Data Transformation and Discretization" - Smoothing, Attribute Construction,
                 Aggregation, Normalization

Input: data/processed/dataset_step3_integrated.csv
Output: data/processed/dataset_step4_transformed.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data/processed"
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STEP3_FILE = PROCESSED_DIR / "dataset_step3_integrated.csv"
OUTPUT_FILE = PROCESSED_DIR / "dataset_step4_transformed.csv"


def load_integrated_data() -> pd.DataFrame:
    """Load dữ liệu đã integrate từ step 3"""
    print("\n⏳ Loading dataset_step3_integrated.csv ...")
    df = pd.read_csv(STEP3_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


# ============================================================================
# 4A: COMPUTE RETURNS
# ============================================================================

def compute_market_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính returns cho các chỉ số thị trường
    
    FIX BUG: ffill oil_close TRƯỚC khi tính pct_change để tránh NaN pair
    (Vì yfinance có NaN vào một số ngày lễ Mỹ)
    """
    print("\n⏳ 4A: Computing market returns...")
    df = df.copy()

    # Ffill oil_close first (fix bug NaN pair)
    if "oil_close" in df.columns:
        df["oil_close"] = df["oil_close"].ffill()

    df["oil_return"] = df["oil_close"].pct_change()
    df["usd_return"] = df["usd_close"].pct_change()
    df["sp500_return"] = df["sp500_close"].pct_change()
    df["vix_return"] = df["vix_close"].pct_change()

    print(f"   ✓ Returns computed")
    return df


# ============================================================================
# 4B: AGGREGATION - ROLLING WINDOWS
# ============================================================================

def create_rolling_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo rolling window aggregation cho các tuần kỳ (7d, 30d)
    """
    print("\n⏳ 4B: Creating rolling windows...")
    df = df.copy()

    # GDELT rolling
    if "gdelt_tone" in df.columns:
        df["gdelt_tone_7d"] = df["gdelt_tone"].rolling(7, min_periods=1).mean()
        df["gdelt_tone_30d"] = df["gdelt_tone"].rolling(30, min_periods=1).mean()

    if "gdelt_goldstein" in df.columns:
        df["gdelt_goldstein_7d"] = df["gdelt_goldstein"].rolling(7, min_periods=1).mean()

    if "gdelt_volume" in df.columns:
        df["gdelt_volume_7d"] = df["gdelt_volume"].rolling(7, min_periods=1).mean()

    # Oil volatility (để winsorize ở 7B)
    if "oil_return" in df.columns:
        df["oil_volatility_7d"] = df["oil_return"].rolling(7, min_periods=1).std(ddof=0).fillna(0.0)

    print(f"   ✓ Rolling windows created")
    return df


# ============================================================================
# 4C: DERIVED FEATURES - FRED
# ============================================================================

def build_cpi_yoy_from_raw_fred() -> pd.Series:
    """Tính CPI YoY từ raw FRED data"""
    fred = pd.read_csv(RAW_DIR / "fred_data.csv")
    fred["date"] = pd.to_datetime(fred["date"])
    fred = fred.sort_values("date").set_index("date")

    if "cpi" not in fred.columns:
        print("   ⚠️  CPI column not found in raw FRED data")
        return pd.Series(dtype=float)

    monthly_cpi = fred["cpi"].resample("MS").first()
    cpi_yoy = monthly_cpi.pct_change(12) * 100
    return cpi_yoy


def create_fred_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo features từ FRED data:
      - fed_rate_change: tốc độ thay đổi lãi suất
      - recession_signal: flag yield spread < 0
      - cpi_yoy: lạm phát year-over-year
      - real_rate: lãi suất thực (fed_rate - cpi_yoy)
      - fed_rate_regime: phân loại chế độ lãi suất (0,1,2,3)
    """
    print("\n⏳ 4C: Creating FRED features...")
    df = df.copy()

    # Fed rate change
    if "fed_funds_rate_lag" in df.columns:
        df["fed_rate_change"] = df["fed_funds_rate_lag"].diff()
        df["fed_rate_change"] = df["fed_rate_change"].fillna(0.0)

    # Recession signal
    if "yield_spread" in df.columns:
        df["recession_signal"] = (df["yield_spread"] < 0).astype(int)

    # CPI YoY (tính từ raw FRED)
    cpi_yoy = build_cpi_yoy_from_raw_fred()
    if not cpi_yoy.empty:
        # Reindex cpi_yoy series vào df's date index
        date_index = pd.to_datetime(df["date"])
        df["cpi_yoy"] = cpi_yoy.reindex(date_index, method="ffill").values
    else:
        df["cpi_yoy"] = 0.0

    # Fed rate regime (0=near zero, 1=rising, 2=elevated stable, 3=other)
    regime = []
    for i in range(len(df)):
        if "fed_funds_rate_lag" not in df.columns:
            regime.append(None)
            continue

        rate = df["fed_funds_rate_lag"].iloc[i]
        prev = df["fed_funds_rate_lag"].iloc[i - 1] if i > 0 else rate

        if pd.isna(rate) or pd.isna(prev):
            regime.append(None)
        elif rate < 0.5:
            regime.append(0)  # Near zero
        elif rate - prev > 0.1:
            regime.append(1)  # Rising
        elif rate > 3.0 and abs(rate - prev) < 0.1:
            regime.append(2)  # Elevated stable
        else:
            regime.append(3)  # Other

    df["fed_rate_regime"] = regime

    # Real rate (fed_rate - cpi_yoy)
    if "fed_funds_rate_lag" in df.columns and "cpi_yoy" in df.columns:
        df["real_rate"] = df["fed_funds_rate_lag"] - df["cpi_yoy"]

    print(f"   ✓ FRED features created")
    return df


# ============================================================================
# 4D: DERIVED FEATURES - EIA
# ============================================================================

def create_eia_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo features từ EIA data:
      - inventory_zscore: (inventory - rolling_mean) / rolling_std
      - production_change_pct: % change 5 business days
      - net_imports_change_pct: % change 5 business days
    """
    print("\n⏳ 4D: Creating EIA features...")
    df = df.copy()

    if "crude_inventory_weekly" not in df.columns:
        print("   ⚠️  crude_inventory_weekly not found")
        return df

    # Inventory z-score (standard 252-day window)
    rolling_mean = df["crude_inventory_weekly"].rolling(252, min_periods=30).mean()
    rolling_std = df["crude_inventory_weekly"].rolling(252, min_periods=30).std()
    df["inventory_zscore"] = (df["crude_inventory_weekly"] - rolling_mean) / rolling_std

    # Fallback to expanding z-score for early periods
    exp_mean = df["crude_inventory_weekly"].expanding(min_periods=5).mean()
    exp_std = df["crude_inventory_weekly"].expanding(min_periods=5).std(ddof=0).replace(0, pd.NA)
    exp_zscore = (df["crude_inventory_weekly"] - exp_mean) / exp_std
    df["inventory_zscore"] = df["inventory_zscore"].fillna(exp_zscore).fillna(0.0)

    # Production change
    if "crude_production_weekly" in df.columns:
        df["production_change_pct"] = df["crude_production_weekly"].pct_change(5).fillna(0.0)

    # Net imports change
    if "net_imports_weekly" in df.columns:
        df["net_imports_change_pct"] = df["net_imports_weekly"].pct_change(5).fillna(0.0)

    print(f"   ✓ EIA features created")
    return df


# ============================================================================
# 4E: DERIVED FEATURES - GDELT
# ============================================================================

def create_gdelt_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo features từ GDELT data:
      - gdelt_volume_log: log transformation
      - gdelt_tone_spike: tone < tone_30d - 1
      - media_attention_spike: volume > 95th percentile rolling
    """
    print("\n⏳ 4E: Creating GDELT features...")
    df = df.copy()

    # Volume log transformation
    if "gdelt_volume" in df.columns:
        df["gdelt_volume_log"] = np.log1p(df["gdelt_volume"])

    # Tone spike (extreme negative sentiment)
    if "gdelt_tone" in df.columns and "gdelt_tone_30d" in df.columns:
        df["gdelt_tone_spike"] = (df["gdelt_tone"] < df["gdelt_tone_30d"] - 1).astype(int)

    # Media attention spike (volume outlier)
    if "gdelt_volume_log" in df.columns:
        rolling_q95 = df["gdelt_volume_log"].rolling(90, min_periods=30).quantile(0.95)
        df["media_attention_spike"] = (df["gdelt_volume_log"] > rolling_q95).astype(int)

    print(f"   ✓ GDELT features created")
    return df


# ============================================================================
# 4F: DERIVED FEATURES - ACLED
# ============================================================================

def create_acled_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ACLED features đã được tính ở step 2 (conflict_intensity_7d, fatalities_7d)
    Ở đây chỉ cần verify chúng có ở đây không
    Nếu không, tính lại từ base columns
    """
    print("\n⏳ 4F: Creating/verifying ACLED features...")
    df = df.copy()

    # Nếu chưa có, tính từ base columns
    if "conflict_event_count" in df.columns:
        if "conflict_intensity_7d" not in df.columns:
            df["conflict_intensity_7d"] = df["conflict_event_count"].rolling(7, min_periods=1).sum()

    if "fatalities" in df.columns:
        if "fatalities_7d" not in df.columns:
            df["fatalities_7d"] = df["fatalities"].rolling(7, min_periods=1).sum()

    # Fill missing ACLED with 0
    acled_cols = ["conflict_event_count", "fatalities", "conflict_intensity_7d", "fatalities_7d"]
    for col in acled_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print(f"   ✓ ACLED features verified")
    return df


# ============================================================================
# 4G: CROSS-SOURCE - GEOPOLITICAL STRESS INDEX
# ============================================================================

def create_cross_source_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo Geopolitical Stress Index từ GDELT:
      - stress_tone = -gdelt_tone (âm tone ~ cao stress)
      - stress_volume = gdelt_volume_log (cao volume ~ cao stress)
      - stress_goldstein = -gdelt_goldstein (âm scale ~ cao stress)
    
    MinMax normalize chỉ trên tập TRAIN (< 2023-01-01) để tránh data leakage
    """
    print("\n⏳ 4G: Creating Geopolitical Stress Index...")
    df = df.copy()

    train_mask = df["date"] < pd.Timestamp("2023-01-01")

    components = {
        "stress_tone": -df["gdelt_tone"],
        "stress_volume": df["gdelt_volume_log"],
        "stress_goldstein": -df["gdelt_goldstein"],
    }

    for new_col, series in components.items():
        scaler = MinMaxScaler()
        train_vals = series[train_mask].dropna().values.reshape(-1, 1)

        if len(train_vals) == 0:
            df[new_col] = 0.0
            continue

        scaler.fit(train_vals)

        transformed = series.copy()
        valid_mask = transformed.notna()
        transformed_out = pd.Series(index=series.index, dtype=float)
        transformed_out[valid_mask] = scaler.transform(transformed[valid_mask].values.reshape(-1, 1)).ravel()
        df[new_col] = transformed_out

    # Weighted combination
    df["geopolitical_stress_index"] = (
        df["stress_tone"] * 0.40
        + df["stress_volume"] * 0.35
        + df["stress_goldstein"] * 0.25
    )

    print(f"   ✓ Geopolitical Stress Index created")
    return df


# ============================================================================
# 4H: LAG FEATURES
# ============================================================================

def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo lag features để tránh data leakage
    Chỉ dùng información từ quá khứ (t-1, t-2)
    """
    print("\n⏳ 4H: Creating lag features...")
    df = df.copy()

    if "oil_return" in df.columns:
        df["oil_return_lag1"] = df["oil_return"].shift(1)
        df["oil_return_lag2"] = df["oil_return"].shift(2)

    if "vix_close" in df.columns:
        df["vix_lag1"] = df["vix_close"].shift(1)

    if "gdelt_tone" in df.columns:
        df["gdelt_tone_lag1"] = df["gdelt_tone"].shift(1)

    if "gdelt_volume_log" in df.columns:
        df["gdelt_volume_lag1"] = df["gdelt_volume_log"].shift(1)

    print(f"   ✓ Lag features created")
    return df


# ============================================================================
# 4I: TIME FEATURES
# ============================================================================

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo time-based features cho seasonal patterns
    """
    print("\n⏳ 4I: Creating time features...")
    df = df.copy()

    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon, ..., 6=Sun
    df["month"] = df["date"].dt.month

    print(f"   ✓ Time features created")
    return df


# ============================================================================
# 4J: WINSORIZE NOISE
# ============================================================================

def winsorize_oil_volatility(df: pd.DataFrame, upper_q: float = 0.99) -> pd.DataFrame:
    """
    Winsorize oil_volatility_7d ở 99th percentile
    để cắt các đột biến cực đoan (COVID, sự kiện địa chính trị)
    """
    print("\n⏳ 4J: Winsorizing oil volatility...")
    df = df.copy()

    if "oil_volatility_7d" not in df.columns:
        print("   ⚠️  oil_volatility_7d not found")
        return df

    cap = df["oil_volatility_7d"].quantile(upper_q)
    df["oil_volatility_7d"] = df["oil_volatility_7d"].clip(upper=cap)

    print(f"   ✓ Volatility winsorized at {upper_q*100:.0f}th percentile (cap={cap:.4f})")
    return df


# ============================================================================
# 4K: FINAL CLEANUP & DROPNA
# ============================================================================

def final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleanup cuối:
      1. ffill(limit=3) để fill các tiny gaps
      2. Drop rows có NaN ở oil_return, oil_return_lag1, oil_return_lag2
      3. Reset index
    """
    print("\n⏳ 4K: Final cleanup...")
    df = df.copy()

    # ffill (limit 3) cho tất cả columns
    fill_cols = [col for col in df.columns if col != "date"]
    df[fill_cols] = df[fill_cols].ffill(limit=3)

    # Drop NaN ở target/lag cols
    if "oil_return" in df.columns:
        initial_len = len(df)
        df = df.dropna(subset=["oil_return", "oil_return_lag1", "oil_return_lag2"])
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"   ✓ Dropped {dropped} rows with NaN in oil_return/lag cols")

    df = df.reset_index(drop=True)

    print(f"   ✓ Cleanup done")
    return df


# ============================================================================
# MAIN TRANSFORMATION FLOW
# ============================================================================

def main():
    print("\n" + "="*80)
    print("🔄 BƯỚC 4: DATA TRANSFORMATION")
    print("="*80)

    # --- Load integrated data ---
    df = load_integrated_data()

    # --- 4A: Compute returns ---
    df = compute_market_returns(df)

    # --- 4B: Rolling windows ---
    df = create_rolling_aggregation(df)

    # --- 4C-F: Derived features ---
    df = create_fred_features(df)
    df = create_eia_features(df)
    df = create_gdelt_features(df)
    df = create_acled_features(df)

    # --- 4G: Cross-source features ---
    df = create_cross_source_features(df)

    # --- 4H: Lag features ---
    df = create_lag_features(df)

    # --- 4I: Time features ---
    df = create_time_features(df)

    # --- 4J: Winsorize noise ---
    df = winsorize_oil_volatility(df, upper_q=0.99)

    # --- 4K: Final cleanup ---
    df = final_cleanup(df)

    # --- Save ---
    print(f"\n⏳ Saving output to {OUTPUT_FILE} ...")
    df.to_csv(OUTPUT_FILE, index=False)

    # --- Report ---
    print(f"\n{'='*80}")
    print("📊 REPORT BƯỚC 4")
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
        dtype = df[col].dtype
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            print(f"  {i:2d}. {col:40s} {str(dtype):10s} ({nan_count:6d} NaN)")
        else:
            print(f"  {i:2d}. {col:40s} {str(dtype):10s}")

    print(f"\n✅ Step 4 hoàn thành! Dữ liệu chuyển đổi sẵn cho Bước 5 (Reduction) / EDA")


if __name__ == "__main__":
    main()
