import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data/raw"
PROCESSED_DIR = BASE_DIR / "data/processed"

INPUT_FILE = PROCESSED_DIR / "dataset_preprocessed.csv"
OUTPUT_FULL_FILE = PROCESSED_DIR / "dataset_final_full.csv"
OUTPUT_MODEL_FILE = PROCESSED_DIR / "dataset_final.csv"


def build_cpi_yoy_from_raw_fred() -> pd.Series:
    fred = pd.read_csv(RAW_DIR / "fred_data.csv")
    fred["date"] = pd.to_datetime(fred["date"])
    fred = fred.sort_values("date").set_index("date")

    monthly_cpi = fred["cpi"].resample("MS").first()
    return monthly_cpi.pct_change(12) * 100


def create_fred_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fed_rate_change"] = df["fed_funds_rate_lag"].diff()
    df["recession_signal"] = (df["yield_spread"] < 0).astype(int)

    cpi_yoy = build_cpi_yoy_from_raw_fred()
    df["cpi_yoy"] = cpi_yoy.reindex(df.set_index("date").index, method="ffill").values

    regime = []
    for i in range(len(df)):
        rate = df["fed_funds_rate_lag"].iloc[i]
        prev = df["fed_funds_rate_lag"].iloc[i - 1] if i > 0 else rate
        change = rate - prev
        if pd.isna(rate) or pd.isna(prev):
            regime.append(None)
        elif rate < 0.5:
            regime.append(0)
        elif change > 0.1:
            regime.append(1)
        elif rate > 3.0 and abs(change) < 0.1:
            regime.append(2)
        else:
            regime.append(3)
    df["fed_rate_regime"] = regime

    df["real_rate"] = df["fed_funds_rate_lag"] - df["cpi_yoy"]
    df["macro_pressure_index"] = df["real_rate"] - df["yield_spread"]
    df["fed_rate_change"] = df["fed_rate_change"].fillna(0.0)
    return df


def create_eia_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rolling_mean = df["crude_inventory_weekly"].rolling(252, min_periods=30).mean()
    rolling_std = df["crude_inventory_weekly"].rolling(252, min_periods=30).std()
    df["inventory_zscore"] = (df["crude_inventory_weekly"] - rolling_mean) / rolling_std

    # Với giai đoạn warmup đầu chuỗi, dùng expanding z-score để giữ timeline.
    exp_mean = df["crude_inventory_weekly"].expanding(min_periods=5).mean()
    exp_std = df["crude_inventory_weekly"].expanding(min_periods=5).std(ddof=0).replace(0, pd.NA)
    exp_zscore = (df["crude_inventory_weekly"] - exp_mean) / exp_std
    df["inventory_zscore"] = df["inventory_zscore"].fillna(exp_zscore).fillna(0.0)

    df["production_change_pct"] = df["crude_production_weekly"].pct_change(5).fillna(0.0)
    df["net_imports_change_pct"] = df["net_imports_weekly"].pct_change(5).fillna(0.0)
    return df


def create_gdelt_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rolling_q95 = df["gdelt_volume_log"].rolling(90, min_periods=30).quantile(0.95)
    df["media_attention_spike"] = (df["gdelt_volume_log"] > rolling_q95).astype(int)
    return df


def create_cross_source_features(df: pd.DataFrame) -> pd.DataFrame:
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

    df["geopolitical_stress_index"] = (
        df["stress_tone"] * 0.40
        + df["stress_volume"] * 0.35
        + df["stress_goldstein"] * 0.25
    )
    return df


def create_lag_time_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["oil_volatility_7d"] = df["oil_return"].rolling(7, min_periods=1).std(ddof=0).fillna(0.0)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df


def enforce_lag_integrity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["oil_return_lag1"] = df["oil_return"].shift(1)
    df["oil_return_lag2"] = df["oil_return"].shift(2)
    df["vix_lag1"] = df["vix_close"].shift(1)
    df["gdelt_tone_lag1"] = df["gdelt_tone"].shift(1)
    df["gdelt_volume_lag1"] = df["gdelt_volume_log"].shift(1)
    return df


def winsorize_oil_volatility(df: pd.DataFrame, upper_q: float = 0.99) -> pd.DataFrame:
    df = df.copy()
    cap = df["oil_volatility_7d"].quantile(upper_q)
    df["oil_volatility_7d"] = df["oil_volatility_7d"].clip(upper=cap)
    return df


def create_model_ready_dataset(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ["stress_tone", "stress_volume", "stress_goldstein", "wti_fred"]
    keep_cols = [c for c in df.columns if c not in drop_cols]
    return df[keep_cols].copy()


# =========================
# MAIN
# =========================

def main():

    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df = create_fred_features(df)
    df = create_eia_features(df)
    df = create_gdelt_features(df)
    df = create_cross_source_features(df)
    df = create_lag_time_volatility_features(df)

    df = df.ffill(limit=3)

    # Không dropna toàn bảng để tránh mất đầu chuỗi vì warmup của một vài feature.
    df = enforce_lag_integrity(df)
    df = df.dropna(subset=["oil_return", "oil_return_lag1", "oil_return_lag2"]).reset_index(drop=True)
    df = winsorize_oil_volatility(df, upper_q=0.99)

    full_df = df.copy()
    model_df = create_model_ready_dataset(full_df)

    full_df.to_csv(OUTPUT_FULL_FILE, index=False)
    model_df.to_csv(OUTPUT_MODEL_FILE, index=False)

    total_missing_full = int(full_df.isna().sum().sum())
    total_missing_model = int(model_df.isna().sum().sum())
    print(
        "Feature engineering complete. "
        f"full_shape={full_df.shape}, full_missing={total_missing_full}; "
        f"model_shape={model_df.shape}, model_missing={total_missing_model}"
    )


if __name__ == "__main__":
    main()