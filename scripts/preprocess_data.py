import numpy as np
import pandas as pd
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
OUTPUT_FILE = PROCESSED_DIR / "dataset_preprocessed.csv"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market = pd.read_csv(MARKET_FILE)
    fred = pd.read_csv(FRED_FILE)
    eia = pd.read_csv(EIA_FILE)
    gdelt = pd.read_csv(GDELT_FILE)
    acled = pd.read_csv(ACLED_FILE)

    for df in [market, fred, eia, gdelt, acled]:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "event_date" in df.columns:
            df["date"] = pd.to_datetime(df["event_date"])

    return market, fred, eia, gdelt, acled


def preprocess_gdelt(gdelt: pd.DataFrame) -> pd.DataFrame:
    gdelt = gdelt.sort_values("date").copy()

    # 1 = ngày thiếu dữ liệu gốc GDELT, sẽ được nội suy bằng ffill ngắn hạn.
    gdelt["gdelt_data_imputed"] = gdelt["gdelt_tone"].isna().astype(int)

    for col in ["gdelt_tone", "gdelt_goldstein", "gdelt_volume", "gdelt_events"]:
        gdelt[col] = gdelt[col].ffill(limit=3)

    gdelt["gdelt_tone_7d"] = gdelt["gdelt_tone"].rolling(7, min_periods=1).mean()
    gdelt["gdelt_tone_30d"] = gdelt["gdelt_tone"].rolling(30, min_periods=1).mean()
    gdelt["gdelt_volume_7d"] = gdelt["gdelt_volume"].rolling(7, min_periods=1).mean()
    gdelt["gdelt_goldstein_7d"] = gdelt["gdelt_goldstein"].rolling(7, min_periods=1).mean()
    gdelt["gdelt_tone_spike"] = (gdelt["gdelt_tone"] < gdelt["gdelt_tone_30d"] - 1).astype(int)

    gdelt["gdelt_volume_log"] = np.log1p(gdelt["gdelt_volume"])
    gdelt["gdelt_volume_7d_log"] = np.log1p(gdelt["gdelt_volume_7d"])

    return gdelt


def preprocess_acled(acled: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý dữ liệu ACLED với giải pháp weekend problem:
    - Dữ liệu ACLED có sự kiện vào Thứ Bảy/Chủ Nhật
    - Dữ liệu thị trường tài chính chỉ có Thứ Hai-Thứ Năm
    - Giải pháp: Gộp sự kiện thứ 6-7 vào thứ 2 tuần tiếp theo (trading day)
    """
    acled = acled.sort_values("date").copy()
    
    # Tổng hợp ACLED theo ngày: đếm sự kiện và tổng fatalities
    daily_acled = acled.groupby("date").agg({
        "event_id_cnty": "count",  # Số sự kiện
        "fatalities": "sum"         # Tổng thương vong
    }).reset_index()
    daily_acled.columns = ["date", "conflict_event_count", "fatalities"]
    
    # Xử lý weekend problem
    daily_acled["day_of_week"] = daily_acled["date"].dt.dayofweek  # 0=Thứ 2, ..., 4=Thứ 6, 5=Thứ 7, 6=CN
    
    # Tạo danh sách ngày giao dịch từ market data (đã được tải)
    # Để xử lý weekend, ta sẽ:
    # - Tìm trading days từ market data
    # - Map các ngày không giao dịch lên trading day tiếp theo
    
    # Với các sự kiện vào thứ 6, 7, CN (day_of_week in [4, 5, 6]):
    # - Thứ 6 (4) → Thứ 2 tuần sau (Thứ Nhật +2 ngày)
    # - Thứ 7 (5) → Thứ 2 tuần sau (Thứ 2 +2 ngày, Thứ 3 +1 ngày)  
    # - CN (6) → Thứ 2 tuần sau (Thứ 2 +1 ngày)
    
    # Tạo mapping date
    daily_acled["date_original"] = daily_acled["date"]
    
    # Nếu là thứ 6, 7 hoặc CN, chuyển sang trading day tiếp theo (thứ 2)
    def map_to_trading_day(row):
        date = row["date"]
        dow = row["day_of_week"]
        
        if dow == 4:  # Thứ 6 -> Thứ 2 tuần sau (+3 ngày)
            return date + pd.Timedelta(days=3)
        elif dow == 5:  # Thứ 7 -> Thứ 2 tuần sau (+2 ngày)
            return date + pd.Timedelta(days=2)
        elif dow == 6:  # CN -> Thứ 2 tuần sau (+1 ngày)
            return date + pd.Timedelta(days=1)
        else:
            return date
    
    daily_acled["date"] = daily_acled.apply(map_to_trading_day, axis=1)
    
    # Gộp các sự kiện đã map vào cùng ngày trading
    daily_acled = daily_acled.groupby("date").agg({
        "conflict_event_count": "sum",
        "fatalities": "sum"
    }).reset_index()
    
    # Tính các rolling features
    daily_acled["conflict_intensity_7d"] = daily_acled["conflict_event_count"].rolling(7, min_periods=1).sum()
    
    # Feature: conflict_intensity (thương vong trung bình trên mỗi vụ)
    daily_acled["conflict_intensity"] = daily_acled["fatalities"] / (daily_acled["conflict_event_count"] + 1)
    
    # Feature: conflict_change_7d (sự bùng phát - so sánh với 7 ngày trước)
    daily_acled["conflict_change_7d"] = (
        daily_acled["conflict_event_count"] - daily_acled["conflict_event_count"].shift(7)
    )
    
    # Feature: fatalities_7d (tổng thương vong 7 ngày)
    daily_acled["fatalities_7d"] = daily_acled["fatalities"].rolling(7, min_periods=1).sum()
    
    # Feature: fatalities_change_7d (sự thay đổi thương vong)
    daily_acled["fatalities_change_7d"] = (
        daily_acled["fatalities"] - daily_acled["fatalities"].shift(7)
    )
    
    # Fill NA
    daily_acled[["conflict_change_7d", "fatalities_change_7d"]] = daily_acled[
        ["conflict_change_7d", "fatalities_change_7d"]
    ].fillna(0)
    
    return daily_acled


def preprocess_fred(fred: pd.DataFrame) -> pd.DataFrame:
    fred = fred.sort_values("date").set_index("date")

    monthly_cols = ["fed_funds_rate", "cpi", "unemployment"]
    for col in monthly_cols:
        # Các series monthly đã được shift ở bước crawl để phản ánh publication lag.
        fred[f"{col}_lag"] = fred[col]

    fred = fred.reset_index()
    return fred


def compute_market_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["oil_return"] = df["oil_close"].pct_change()
    df["usd_return"] = df["usd_close"].pct_change()
    df["sp500_return"] = df["sp500_close"].pct_change()
    df["vix_return"] = df["vix_close"].pct_change()
    return df


def merge_datasets(
    market: pd.DataFrame,
    fred: pd.DataFrame,
    eia: pd.DataFrame,
    gdelt: pd.DataFrame,
    acled: pd.DataFrame,
) -> pd.DataFrame:
    df = market.sort_values("date").copy()

    fred_keep = [
        "date",
        "yield_spread",
        "wti_fred",
        "fed_funds_rate_lag",
        "cpi_lag",
        "unemployment_lag",
    ]

    df = df.merge(fred[fred_keep], on="date", how="left")
    df = df.merge(eia.sort_values("date"), on="date", how="left")
    df = df.merge(gdelt.sort_values("date"), on="date", how="left")
    df = df.merge(acled.sort_values("date"), on="date", how="left")

    return df


def post_merge_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    fill_cols = [col for col in df.columns if col not in ["date", "oil_return", "usd_return", "sp500_return", "vix_return"]]
    df[fill_cols] = df[fill_cols].ffill(limit=3)

    if "gdelt_data_imputed" in df.columns:
        df["gdelt_data_imputed"] = df["gdelt_data_imputed"].fillna(1).astype(int)

    # Fill ACLED features với 0 nếu không có dữ liệu
    acled_cols = ["conflict_event_count", "fatalities", "conflict_intensity_7d", 
                  "conflict_intensity", "conflict_change_7d", "fatalities_7d", "fatalities_change_7d"]
    for col in acled_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def main():
    market, fred, eia, gdelt, acled = load_data()

    fred = preprocess_fred(fred)
    gdelt = preprocess_gdelt(gdelt)
    acled = preprocess_acled(acled)

    df = merge_datasets(market, fred, eia, gdelt, acled)
    df = compute_market_returns(df)
    df = post_merge_fill(df)

    df.to_csv(OUTPUT_FILE, index=False)

    total_missing = int(df.isna().sum().sum())
    print(f"Preprocessing complete. Shape={df.shape}, missing={total_missing}")


if __name__ == "__main__":
    main()