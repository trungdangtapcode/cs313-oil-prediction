"""
Shared preprocessing helpers for classification.

There are now two distinct dataset stages:

  - dataset_final_noleak_processed.csv
    Deterministic transforms only from step5b. Scalers are still fit at model time.

  - dataset_final_noleak_step5c.csv
    Research-oriented dataset where curated scalers are already baked into the
    exported CSV. When this dataset is used, the training pipeline should avoid
    scaling a second time.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler


STEP5C_STANDARD_COLS = [
    "yield_spread",
    "gdelt_goldstein",
    "gdelt_goldstein_7d",
    "usd_return",
    "inventory_zscore",
]

STEP5C_ROBUST_COLS = [
    "inventory_change_pct",
    "oil_return",
    "oil_return_lag1",
    "oil_return_lag2",
    "sp500_return",
    "gdelt_events_log1p",
    "conflict_event_count_log1p",
    "conflict_intensity_7d_log1p",
    "fatalities_log1p",
    "fatalities_7d_log1p",
    "gdelt_volume_lag1_log1p",
    "vix_lag1_log1p",
    "net_imports_change_pct_slog1p",
    "production_change_pct_slog1p",
    "vix_return_slog1p",
]

STEP5C_POWER_COLS = [
    "gdelt_tone_7d",
    "gdelt_tone_30d",
    "gdelt_tone_lag1",
]

STEP5C_PASSTHROUGH_COLS = [
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]


def _ordered_subset(feature_names: Sequence[str], candidates: Sequence[str]) -> List[str]:
    feature_set = set(feature_names)
    return [col for col in candidates if col in feature_set]


def _is_step5c_baked_scaled(data_path: Optional[str]) -> bool:
    if not data_path:
        return False
    try:
        return Path(data_path).name in {
            "dataset_final_noleak_step5c.csv",
            "dataset_final_noleak_step5c_scaler.csv",
        }
    except Exception:
        return False


def get_model_time_groups(feature_names: Sequence[str], data_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Return column groups for model-time preprocessing.

    If the active dataset is the baked-scaled step5c export, treat every feature
    as passthrough. Otherwise, if the feature set matches the step5b/step5c
    processed schema, use the curated Standard / Robust / Power grouping.
    Fall back to a generic all-standard setup for unknown schemas.
    """
    feature_names = list(feature_names)
    feature_set = set(feature_names)

    if _is_step5c_baked_scaled(data_path):
        return {
            "standard": [],
            "robust": [],
            "power": [],
            "passthrough": list(feature_names),
            "other": [],
            "schema": "step5c_baked_scaled_passthrough",
        }

    looks_like_step5c = {
        "day_of_week_sin",
        "gdelt_events_log1p",
        "vix_return_slog1p",
    }.issubset(feature_set)

    if not looks_like_step5c:
        return {
            "standard": list(feature_names),
            "robust": [],
            "power": [],
            "passthrough": [],
            "other": [],
            "schema": "generic_all_standard",
        }

    standard = _ordered_subset(feature_names, STEP5C_STANDARD_COLS)
    robust = _ordered_subset(feature_names, STEP5C_ROBUST_COLS)
    power = _ordered_subset(feature_names, STEP5C_POWER_COLS)
    passthrough = _ordered_subset(feature_names, STEP5C_PASSTHROUGH_COLS)

    assigned = set(standard) | set(robust) | set(power) | set(passthrough)
    other = [col for col in feature_names if col not in assigned]

    return {
        "standard": standard,
        "robust": robust,
        "power": power,
        "passthrough": passthrough,
        "other": other,
        "schema": "step5c_train_curated",
    }


def build_model_time_preprocessor(feature_names: Sequence[str], data_path: Optional[str] = None) -> ColumnTransformer:
    groups = get_model_time_groups(feature_names, data_path=data_path)
    transformers = []

    if groups["standard"]:
        transformers.append(
            (
                "standard",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                groups["standard"],
            )
        )

    if groups["robust"]:
        transformers.append(
            (
                "robust",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                groups["robust"],
            )
        )

    if groups["power"]:
        transformers.append(
            (
                "power",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
                    ]
                ),
                groups["power"],
            )
        )

    passthrough_cols = groups["passthrough"] + groups["other"]
    if passthrough_cols:
        transformers.append(
            (
                "passthrough",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                passthrough_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def get_preprocessor_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    names = list(preprocessor.get_feature_names_out())
    cleaned = []
    for name in names:
        cleaned.append(name.split("__", 1)[1] if "__" in name else name)
    return cleaned


def save_model_bundle(path: str, model, feature_names: Sequence[str], preprocessor=None, data_path: Optional[str] = None) -> str:
    bundle = {
        "model": model,
        "feature_names": list(feature_names),
        "preprocessor": preprocessor,
        "preprocessor_feature_names": get_preprocessor_feature_names(preprocessor) if preprocessor is not None else list(feature_names),
        "groups": get_model_time_groups(feature_names, data_path=data_path),
        "data_path": data_path,
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    return str(out_path)
STEP5B_STANDARD_COLS = STEP5C_STANDARD_COLS
STEP5B_ROBUST_COLS = STEP5C_ROBUST_COLS
STEP5B_POWER_COLS = STEP5C_POWER_COLS
STEP5B_PASSTHROUGH_COLS = STEP5C_PASSTHROUGH_COLS
