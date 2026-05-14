#!/usr/bin/env python3
"""Model factories, experiment candidates, and recency weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from config import RANDOM_STATE, model_n_jobs


@dataclass(frozen=True)
class Candidate:
    name: str
    model_kind: str
    scheme: str = "uniform"
    feature_set: str = "all"
    params: Optional[Dict] = None


def _xgb_classifier():
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise ImportError("xgboost is required for XGB candidates") from exc
    return XGBClassifier


def _lgbm_classifier():
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        raise ImportError("lightgbm is required for LGBM candidates") from exc
    return LGBMClassifier


def has_xgboost() -> bool:
    try:
        _xgb_classifier()
        return True
    except Exception:
        return False


def has_lightgbm() -> bool:
    try:
        _lgbm_classifier()
        return True
    except Exception:
        return False


def make_model(model_kind: str):
    jobs = model_n_jobs()

    if model_kind == "LogisticRegression":
        return LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, C=1.0, solver="lbfgs")
    if model_kind == "SVM_RBF":
        return SVC(random_state=RANDOM_STATE, probability=True, kernel="rbf", C=1.0, gamma="scale")
    if model_kind == "SVM_Linear":
        return SVC(random_state=RANDOM_STATE, probability=True, kernel="linear", C=1.0)
    if model_kind == "RandomForest":
        return RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=jobs,
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
        )
    if model_kind == "GBM":
        return GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            min_samples_leaf=5,
        )
    if model_kind == "GBM_small":
        return GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            n_estimators=150,
            max_depth=3,
            learning_rate=0.04,
            min_samples_leaf=10,
        )
    if model_kind == "XGB":
        XGBClassifier = _xgb_classifier()
        return XGBClassifier(
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=jobs,
            eval_metric="logloss",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
        )
    if model_kind == "XGB_small":
        XGBClassifier = _xgb_classifier()
        return XGBClassifier(
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=jobs,
            eval_metric="logloss",
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
        )
    if model_kind == "LGBM":
        LGBMClassifier = _lgbm_classifier()
        return LGBMClassifier(
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=jobs,
            importance_type="gain",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
        )
    if model_kind == "LGBM_small":
        LGBMClassifier = _lgbm_classifier()
        return LGBMClassifier(
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=jobs,
            importance_type="gain",
            n_estimators=200,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.03,
            min_child_samples=20,
            reg_lambda=1.0,
        )
    if model_kind == "MLP":
        return MLPClassifier(
            random_state=RANDOM_STATE,
            max_iter=500,
            hidden_layer_sizes=(64, 32),
            alpha=0.001,
            learning_rate_init=0.001,
        )

    raise ValueError("Unknown model kind: %s" % model_kind)


def available_candidates(candidates: List[Candidate]) -> List[Candidate]:
    available = []
    for cand in candidates:
        if cand.model_kind.startswith("XGB") and not has_xgboost():
            print("[SKIP] %s needs xgboost" % cand.name)
            continue
        if cand.model_kind.startswith("LGBM") and not has_lightgbm():
            print("[SKIP] %s needs lightgbm" % cand.name)
            continue
        available.append(cand)
    return available


def baseline_candidates() -> List[Candidate]:
    return available_candidates(
        [
            Candidate("BASE_LogReg", "LogisticRegression"),
            Candidate("BASE_SVM_RBF", "SVM_RBF"),
            Candidate("BASE_RandomForest", "RandomForest"),
            Candidate("BASE_GBM", "GBM"),
            Candidate("BASE_MLP", "MLP"),
            Candidate("BASE_XGB", "XGB_small"),
            Candidate("BASE_LGBM", "LGBM_small"),
        ]
    )


def compact_weight_decay_candidates() -> List[Candidate]:
    return available_candidates(
        [
            Candidate("GBM_exp100", "GBM", "exp_hl100"),
            Candidate("GBM_exp250", "GBM", "exp_hl250"),
            Candidate("XGB_step50_3", "XGB", "step_50pct_3x"),
            Candidate("XGB_exp100", "XGB", "exp_hl100"),
            Candidate("XGB_linear03", "XGB", "linear_03"),
            Candidate("LGBM_exp100", "LGBM", "exp_hl100"),
            Candidate("LGBM_step50_3", "LGBM", "step_50pct_3x"),
            Candidate("LGBM_linear01", "LGBM", "linear_01"),
            Candidate("LGBM_linear03", "LGBM", "linear_03"),
            Candidate("LGBM_small_step50_3", "LGBM_small", "step_50pct_3x"),
        ]
    )


def full_weight_decay_candidates() -> List[Candidate]:
    candidates = []
    for scheme in ["uniform", "exp_hl100", "exp_hl250", "linear_01", "linear_03", "step_50pct_3x"]:
        candidates.extend(
            [
                Candidate("GBM_%s" % scheme, "GBM", scheme),
                Candidate("XGB_%s" % scheme, "XGB", scheme),
                Candidate("LGBM_%s" % scheme, "LGBM", scheme),
                Candidate("LGBM_small_%s" % scheme, "LGBM_small", scheme),
            ]
        )
    return available_candidates(candidates)


def normalize_weights(weights: np.ndarray, target_mean: float = 1.0) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    mean_weight = weights.mean()
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError("Cannot normalize weights with non-positive mean")
    return weights * (target_mean / mean_weight)


def exponential_weights(n: int, half_life: float) -> np.ndarray:
    decay = np.log(2) / half_life
    return np.exp(-decay * (n - 1 - np.arange(n)))


def linear_weights(n: int, min_weight: float = 0.1) -> np.ndarray:
    return np.linspace(min_weight, 1.0, n)


def step_weights(n: int, recent_ratio: float = 0.5, recent_weight: float = 2.0) -> np.ndarray:
    weights = np.ones(n)
    cutoff = int(n * (1 - recent_ratio))
    weights[cutoff:] = recent_weight
    return weights


def weight_schemes(n: int) -> Dict[str, np.ndarray]:
    raw = {
        "uniform": np.ones(n),
        "exp_hl100": exponential_weights(n, 100),
        "exp_hl250": exponential_weights(n, 250),
        "exp_hl500": exponential_weights(n, 500),
        "linear_01": linear_weights(n, 0.1),
        "linear_03": linear_weights(n, 0.3),
        "linear_05": linear_weights(n, 0.5),
        "step_50pct_2x": step_weights(n, 0.5, 2.0),
        "step_50pct_3x": step_weights(n, 0.5, 3.0),
        "step_30pct_3x": step_weights(n, 0.3, 3.0),
    }
    return {name: normalize_weights(weights) for name, weights in raw.items()}


def sample_weights(n: int, scheme: str) -> np.ndarray:
    schemes = weight_schemes(n)
    if scheme not in schemes:
        raise ValueError("Unknown weighting scheme: %s" % scheme)
    return schemes[scheme]
