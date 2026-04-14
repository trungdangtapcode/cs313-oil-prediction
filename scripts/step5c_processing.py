"""
================================================================================
STEP 5C: PROCESSING + MODEL-TIME PREPROCESSING MAP
================================================================================

This step rebuilds the deterministic processed dataset from step5b and publishes
it under a clearer step5c name to avoid confusion with the earlier experiment.

It keeps the dataset leakage-safe by only applying deterministic transforms:
  - cyclical encoding for calendar features
  - log1p for heavy-tailed positive features
  - signed log1p for selected signed-change features

Fit-based preprocessing such as StandardScaler / RobustScaler /
PowerTransformer is still deferred to the model pipeline. The train-time group
mapping lives in ml/model_preprocessing.py and is not exported as extra CSV.

Outputs:
  - data/processed/dataset_final_noleak_step5c.csv
"""

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step5b_processing import load_data, process_data  # noqa: E402


PROCESSED_DIR = BASE_DIR / "data/processed"
OUTPUT_FILE = PROCESSED_DIR / "dataset_final_noleak_step5c.csv"


def main() -> None:
    print("\n" + "=" * 80)
    print("STEP 5C: DETERMINISTIC PROCESSING + TRAIN-TIME PREPROCESSING MAP")
    print("=" * 80)
    print("\nThis step republishes the clean processed dataset under the step5c name.")
    print("Fit-based scalers remain in the training pipeline, not in the CSV.")
    print("Model-time preprocessing groups live in ml/model_preprocessing.py.")

    df = load_data()
    processed, report = process_data(df)

    print(f"\n⏳ Saving step5c dataset to {OUTPUT_FILE.name} ...")
    processed.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 80}")
    print("📊 STEP 5C REPORT")
    print(f"{'=' * 80}")
    print(f"Input shape          : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Step5c shape         : {processed.shape[0]} rows × {processed.shape[1]} cols")
    print(f"Output dataset       : {OUTPUT_FILE}")
    print("Train-time groups    : ml/model_preprocessing.py")

    changed = report[report["transform"] != "keep"]
    print("\nDeterministic transforms:")
    for row in changed.itertuples(index=False):
        print(f"  - {row.source_feature} -> {row.output_feature} [{row.transform}]")


if __name__ == "__main__":
    main()
