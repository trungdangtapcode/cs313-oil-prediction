"""
================================================================================
BƯỚC 5: DATA REDUCTION (Chọn Features Quan Trọng)
================================================================================

Mục tiêu:
    1. Drop cột trung gian/weak/redundant theo chiến lược aggressive từ EDA
    2. Xử lý đa cộng tuyến mạnh ở nhóm Macro, Supply và GDELT
  3. Tạo 2 output files:
     - dataset_final_full.csv: giữ tất cả (cho debug)
     - dataset_final.csv: chỉ features đưa vào model

Tương ứng slide: "Data Reduction: reduce representation yet maintain integrity" - 
                 PCA, Attribute Subset Selection, Sampling

Input: data/processed/dataset_step4_transformed.csv
Output: 
  - data/processed/dataset_final_full.csv
  - data/processed/dataset_final.csv
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data/processed"

STEP4_FILE = PROCESSED_DIR / "dataset_step4_transformed.csv"
OUTPUT_FULL_FILE = PROCESSED_DIR / "dataset_final_full.csv"
OUTPUT_MODEL_FILE = PROCESSED_DIR / "dataset_final.csv"


def load_transformed_data() -> pd.DataFrame:
    """Load dữ liệu đã transform từ step 4"""
    print("\n⏳ Loading dataset_step4_transformed.csv ...")
    df = pd.read_csv(STEP4_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def create_model_ready_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
        Drop cột theo quyết định EDA (aggressive approach):
            - Intermediate: stress_tone, stress_volume, stress_goldstein
            - Weak binary/noise: gdelt_tone_spike, media_attention_spike, recession_signal, gdelt_data_imputed
            - Perfect/high collinearity groups:
                    + Macro: cpi_yoy, fed_funds_rate_lag (giữ real_rate + fed_rate_change)
                    + Supply raw levels: crude_inventory_weekly, crude_production_weekly, net_imports_weekly
                    + GDELT tone/volume: gdelt_tone, gdelt_volume, gdelt_volume_log, gdelt_volume_7d
            - Legacy redundant: wti_fred (nếu tồn tại)
    """
    print("\n⏳ Dropping intermediate/redundant columns...")
    df = df.copy()

    drop_cols = [
                # Redundant/intermediate
                "gdelt_volume",
                "stress_tone",
                "stress_volume",
                "stress_goldstein",

                # Weak binary signals
                "gdelt_tone_spike",
                "media_attention_spike",
                "recession_signal",
                "gdelt_data_imputed",

                # Macro collinearity
                "cpi_yoy",
                "fed_funds_rate_lag",

                # Supply raw level features
                "crude_inventory_weekly",
                "crude_production_weekly",
                "net_imports_weekly",

                # Sentiment volume/tone high-collinearity set
                "gdelt_tone",
                "gdelt_volume_log",
                "gdelt_volume_7d",

                # Legacy redundant column
                "wti_fred",

                # **LEAKAGE**: Same-day market data (use lag1 versions instead)
                "vix_close",
                "usd_close",
                "sp500_close",
                "oil_close",
    ]

    # Remove duplicates while preserving order
    drop_cols = list(dict.fromkeys(drop_cols))

    # Chỉ drop những cột thực tế tồn tại
    drop_cols = [col for col in drop_cols if col in df.columns]

    if drop_cols:
        print(f"   Dropping: {drop_cols}")
        df = df.drop(columns=drop_cols)

    print(f"   ✓ Reduced to {df.shape[1]} columns")
    return df


def main():
    print("\n" + "="*80)
    print("🔪 BƯỚC 5: DATA REDUCTION")
    print("="*80)

    # --- Load ---
    df = load_transformed_data()

    # --- Save full version (debug) ---
    print(f"\n⏳ Saving full version to {OUTPUT_FULL_FILE} ...")
    df.to_csv(OUTPUT_FULL_FILE, index=False)
    print(f"   ✓ {OUTPUT_FULL_FILE}")

    # --- Create model-ready version ---
    df_model = create_model_ready_dataset(df)

    # --- Save model version ---
    print(f"\n⏳ Saving model-ready version to {OUTPUT_MODEL_FILE} ...")
    df_model.to_csv(OUTPUT_MODEL_FILE, index=False)
    print(f"   ✓ {OUTPUT_MODEL_FILE}")

    # --- Report ---
    print(f"\n{'='*80}")
    print("📊 REPORT BƯỚC 5")
    print(f"{'='*80}")

    print(f"\nFull version (debug):")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} cols")

    print(f"\nModel version (production):")
    print(f"  Shape: {df_model.shape[0]} rows × {df_model.shape[1]} cols")
    print(f"  Reduction: {df.shape[1]} → {df_model.shape[1]} cols ({df_model.shape[1]/df.shape[1]*100:.1f}%)")

    missing_full = df.isnull().sum().sum()
    missing_model = df_model.isnull().sum().sum()
    print(f"\nMissing values:")
    print(f"  Full  : {missing_full}/{df.shape[0]*df.shape[1]} ({missing_full/(df.shape[0]*df.shape[1])*100:.1f}%)")
    print(f"  Model : {missing_model}/{df_model.shape[0]*df_model.shape[1]} ({missing_model/(df_model.shape[0]*df_model.shape[1])*100:.1f}%)")

    print(f"\n✅ Step 5 hoàn thành!")


if __name__ == "__main__":
    main()
