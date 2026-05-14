#!/usr/bin/env python3
"""Feature-selection experiment: rank features, choose a subset, retrain."""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

from config import RESULTS_DIR, RANDOM_STATE, ensure_dirs, parse_int_csv_env, set_seed
from evaluation import describe_splits, evaluate_candidate, load_dataset, split_masks, target_array, write_experiment_outputs
from model_zoo import Candidate, available_candidates, has_lightgbm


def build_rankings(X_train: pd.DataFrame, y_train: np.ndarray, features: List[str]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    mi = mutual_info_classif(X_train.fillna(0), y_train, random_state=RANDOM_STATE, n_neighbors=5)
    sp = X_train.corrwith(pd.Series(y_train, index=X_train.index), method="spearman").abs()

    rank = pd.DataFrame({"feature": features, "MI": mi, "abs_sp": sp.values})
    for col in ["MI", "abs_sp"]:
        mx = rank[col].max()
        rank["%s_n" % col] = rank[col] / mx if mx > 0 else 0
    rank["mix_score"] = (rank["MI_n"] + rank["abs_sp_n"]) / 2

    orders = {
        "spearman": rank.sort_values(["abs_sp", "MI"], ascending=False).reset_index(drop=True),
        "mi": rank.sort_values(["MI", "abs_sp"], ascending=False).reset_index(drop=True),
        "mi_spearman": rank.sort_values(["mix_score", "MI", "abs_sp"], ascending=False).reset_index(drop=True),
    }
    rank["rank_spearman"] = rank["feature"].map({f: i + 1 for i, f in enumerate(orders["spearman"]["feature"])})
    rank["rank_mi"] = rank["feature"].map({f: i + 1 for i, f in enumerate(orders["mi"]["feature"])})
    rank["rank_mi_spearman"] = rank["feature"].map({f: i + 1 for i, f in enumerate(orders["mi_spearman"]["feature"])})
    return rank.sort_values("rank_mi_spearman").reset_index(drop=True), orders


def subset_cases(features: List[str], orders: Dict[str, pd.DataFrame], subset_sizes: List[int]) -> List[Dict]:
    cases = []
    for ranking, ordered in orders.items():
        for n in subset_sizes:
            if n >= len(features):
                continue
            feats = ordered.head(n)["feature"].tolist()
            cases.append(
                {
                    "Case": "%s_TOP_%s" % (ranking.upper(), n),
                    "Ranking": ranking,
                    "Subset": "TOP_%s" % n,
                    "N": n,
                    "Features": feats,
                }
            )
    cases.append({"Case": "ALL_%s" % len(features), "Ranking": "all", "Subset": "ALL", "N": len(features), "Features": list(features)})
    return cases


def proxy_model():
    if has_lightgbm():
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=1,
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
        )
    return GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=120, max_depth=3, learning_rate=0.05)


def compare_subsets(X_train: pd.DataFrame, y_train: np.ndarray, cases: List[Dict]) -> pd.DataFrame:
    proxy = proxy_model()
    rows = []
    cv = TimeSeriesSplit(n_splits=5)
    for idx, case in enumerate(cases, 1):
        feats = case["Features"]
        scores = []
        print("[feature_selection] compare subset %s/%s %s features=%s" % (idx, len(cases), case["Case"], len(feats)), flush=True)
        for train_idx, val_idx in cv.split(X_train):
            model = clone(proxy)
            model.fit(X_train.iloc[train_idx][feats], y_train[train_idx])
            pred = model.predict(X_train.iloc[val_idx][feats])
            scores.append(f1_score(y_train[val_idx], pred, average="macro", zero_division=0))
        scores = np.asarray(scores, dtype=float)
        rows.append(
            {
                "Case": case["Case"],
                "Ranking": case["Ranking"],
                "Subset": case["Subset"],
                "N": len(feats),
                "CV_F1_macro": scores.mean(),
                "CV_F1_macro_std": scores.std(),
                "Features": "|".join(feats),
            }
        )
    return pd.DataFrame(rows).sort_values(["CV_F1_macro", "N"], ascending=[False, True])


def main() -> None:
    ensure_dirs()
    seed = set_seed()
    df, features = load_dataset()
    masks = split_masks(df)
    y = target_array(df)
    X_train = df.loc[masks["train"], features]
    y_train = y[masks["train"]]
    subset_sizes = [n for n in parse_int_csv_env("FEATURE_SUBSET_SIZES", "8,12,16,20,25") if n < len(features)]
    print("[feature_selection] seed=%s features=%s subset_sizes=%s splits=%s" % (seed, len(features), subset_sizes, describe_splits(masks)))

    rank, orders = build_rankings(X_train, y_train, features)
    rank.to_csv(RESULTS_DIR / "feature_ranking.csv", index=False)

    subset_df = compare_subsets(X_train, y_train, subset_cases(features, orders, subset_sizes))
    subset_df.to_csv(RESULTS_DIR / "feature_selection_subset_results.csv", index=False)
    selected = subset_df.iloc[0]
    selected_features = selected["Features"].split("|")
    (RESULTS_DIR / "selected_features.json").write_text(
        json.dumps(
            {
                "selected_case": selected["Case"],
                "selected_ranking": selected["Ranking"],
                "selected_n": int(selected["N"]),
                "selected_features": selected_features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[feature_selection] selected %s with %s features" % (selected["Case"], len(selected_features)))

    candidates = available_candidates(
        [
            Candidate("FS_GBM", "GBM", feature_set=selected["Case"]),
            Candidate("FS_XGB", "XGB", feature_set=selected["Case"]),
            Candidate("FS_LGBM", "LGBM", feature_set=selected["Case"]),
            Candidate("FS_LGBM_small", "LGBM_small", feature_set=selected["Case"]),
        ]
    )

    rows = []
    val_predictions = []
    test_predictions = []
    for candidate in candidates:
        print("[feature_selection] train %s" % candidate.name)
        cand_rows, val_pred, test_pred, _ = evaluate_candidate(
            df,
            selected_features,
            masks,
            candidate,
            experiment="feature_selection",
            use_sample_weight=False,
            add_selective=False,
            save_model=False,
        )
        rows.extend(cand_rows)
        val_predictions.append(val_pred)
        test_predictions.append(test_pred)

    results = write_experiment_outputs("feature_selection", rows, val_predictions, test_predictions)
    top = results[results["Split"].eq("test")].sort_values(["F1_macro", "Accuracy", "AUC"], ascending=False).head(10)
    print(top[["Model", "Accuracy", "F1_macro", "AUC", "Threshold", "ThresholdMode"]].to_string(index=False))


if __name__ == "__main__":
    main()
