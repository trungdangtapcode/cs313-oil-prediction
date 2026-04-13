"""
================================================================================
STEP 4B: COMPATIBILITY EXPORT FOR LEGACY "NOLEAK" FILES, BUT DEPRECATED (step4_transformation.py did it for u)
================================================================================

The pipeline now predicts T+1 returns directly from the step4 transformed dataset:
  - features at day T
  - target = oil_return_fwd1

Because that setup no longer needs the old same-day shifting logic, this script now
simply exports compatibility copies for any downstream code that still expects the
"*_noleak.csv" filenames.
"""

from pathlib import Path
import shutil

from step5_reduction import OUTPUT_MODEL_FILE

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data/processed"

INPUT_STEP4 = PROCESSED_DIR / "dataset_step4_transformed.csv"
OUTPUT_STEP4_NOLEAK = PROCESSED_DIR / "dataset_step4_noleak.csv"
INPUT_FINAL = OUTPUT_MODEL_FILE
OUTPUT_FINAL_NOLEAK = PROCESSED_DIR / "dataset_final_noleak.csv"


def copy_with_notice(src: Path, dst: Path) -> None:
    print(f"  {src.name} -> {dst.name}")
    shutil.copy2(src, dst)


def main() -> None:
    print("\n" + "=" * 80)
    print("STEP 4B: COMPATIBILITY EXPORT")
    print("=" * 80)
    print("\nThis step no longer applies extra shifting.")
    print("The T+1 target is now created directly in step4_transformation.py.")

    copy_with_notice(INPUT_STEP4, OUTPUT_STEP4_NOLEAK)
    copy_with_notice(INPUT_FINAL, OUTPUT_FINAL_NOLEAK)

    print("\nDone.")


if __name__ == "__main__":
    main()
