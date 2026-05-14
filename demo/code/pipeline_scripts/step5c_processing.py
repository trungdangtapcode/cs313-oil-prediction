"""
================================================================================
STEP 5C: FULL-DATA SCALING EXPORT
================================================================================

This step is intentionally different from step5b:

  - step5b keeps only deterministic transforms and remains leakage-safer
  - step5c takes the step5b-style processed features and bakes imputation +
    scaling into an exported CSV

Design intent:
  - convenience dataset for research / visualization / fixed offline experiments
  - not the strictest setup for honest train/test evaluation, because scaler
    statistics are fit on the full exported dataset

Outputs:
  - data/processed/dataset_final_noleak_step5c.csv
  - data/processed/dataset_final_noleak_step5c_preprocessor.joblib
"""

from pathlib import Path
import sys

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "ml"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_preprocessing import build_model_time_preprocessor, get_preprocessor_feature_names, get_model_time_groups  # noqa: E402
from step5b_processing import TARGET_COLS, load_data, process_data  # noqa: E402


PROCESSED_DIR = BASE_DIR / "data/processed"
OUTPUT_FILE = PROCESSED_DIR / "dataset_final_noleak_step5c.csv"
PREPROCESSOR_FILE = PROCESSED_DIR / "dataset_final_noleak_step5c_preprocessor.joblib"


def main() -> None:
    print("\n" + "=" * 80)
    print("STEP 5C: FULL-DATA SCALING EXPORT")
    print("=" * 80)
    print("\nThis step keeps step5b unchanged.")
    print("It rebuilds the processed features, then fits imputation + scaling on the")
    print("full dataset and exports that scaled matrix as step5c.")
    print("Use this for convenience/research, not for strict leakage-free evaluation.")

    raw_df = load_data()
    processed, report = process_data(raw_df)

    leading_cols = [c for c in TARGET_COLS if c in processed.columns]
    feature_cols = [c for c in processed.columns if c not in leading_cols]

    preprocessor = build_model_time_preprocessor(feature_cols)
    transformed = preprocessor.fit_transform(processed[feature_cols])
    transformed_cols = get_preprocessor_feature_names(preprocessor)
    scaled = pd.DataFrame(transformed, columns=transformed_cols, index=processed.index)

    # Restore the original feature ordering for easier downstream comparison.
    scaled = scaled[feature_cols]
    exported = pd.concat([processed[leading_cols], scaled], axis=1)

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "feature_names": feature_cols,
            "groups": get_model_time_groups(feature_cols),
            "source_dataset": "dataset_final_noleak.csv",
            "source_step": "step5b_processing.py + full-data scaling",
        },
        PREPROCESSOR_FILE,
    )
    exported.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 80}")
    print("STEP 5C REPORT")
    print(f"{'=' * 80}")
    print(f"Input raw shape       : {raw_df.shape[0]} rows × {raw_df.shape[1]} cols")
    print(f"Step5b-like shape     : {processed.shape[0]} rows × {processed.shape[1]} cols")
    print(f"Step5c scaled shape   : {exported.shape[0]} rows × {exported.shape[1]} cols")
    print(f"Output dataset        : {OUTPUT_FILE}")
    print(f"Saved preprocessor    : {PREPROCESSOR_FILE}")
    print(f"Scaled feature count  : {len(feature_cols)}")

    scale_groups = get_model_time_groups(feature_cols)
    print(
        "\nCurated groups used for baked scaling:"
        f"\n  - standard: {len(scale_groups['standard'])}"
        f"\n  - robust: {len(scale_groups['robust'])}"
        f"\n  - power: {len(scale_groups['power'])}"
        f"\n  - passthrough: {len(scale_groups['passthrough']) + len(scale_groups['other'])}"
    )

    changed = report[report["transform"] != "keep"]
    print("\nStep5b transforms preserved before scaling:")
    for row in changed.itertuples(index=False):
        print(f"  - {row.source_feature} -> {row.output_feature} [{row.transform}]")


if __name__ == "__main__":
    main()
