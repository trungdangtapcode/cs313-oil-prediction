#!/usr/bin/env python3
"""Project paths, split dates, and small runtime helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]

RESULTS_DIR = THIS_DIR / "results"
LOG_DIR = THIS_DIR / "logs"
MODELS_DIR = THIS_DIR / "models"

DATA_PATH = REPO_ROOT / "data" / "processed" / "dataset_final_noleak_step5c_scaler.csv"
FINAL_RESULTS_DIR = REPO_ROOT / "ml" / "classification" / "final" / "results"

TARGET = "oil_return_fwd1"
TARGET_DATE_COL = "oil_return_fwd1_date"

RANDOM_STATE = 42
VAL_SPLIT_DATE = pd.Timestamp("2022-01-01")
TEST_SPLIT_DATE = pd.Timestamp("2023-01-01")

EXPERIMENTS = [
    "final_baselines",
    "baseline",
    "feature_selection",
    "weight_decay",
    "ensemble",
    "deep_learning",
    "report",
]

SORT_COLS = ["F1_macro", "Accuracy", "AUC"]


def ensure_dirs() -> None:
    for path in [RESULTS_DIR, LOG_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RANDOM_STATE) -> int:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    return seed


def parse_csv_env(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_csv_env(name: str, default: str) -> List[int]:
    values = []
    for item in parse_csv_env(name, default):
        try:
            values.append(int(item))
        except ValueError:
            continue
    return values


def model_n_jobs(default: int = 4) -> int:
    return max(1, int(os.getenv("MODEL_N_JOBS", str(default))))


def search_n_jobs(default: int = 4) -> int:
    return max(1, int(os.getenv("SEARCH_N_JOBS", str(default))))
