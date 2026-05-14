#!/usr/bin/env python3
"""Weight-decay experiment: train ML candidates with recency sample weights."""

from __future__ import annotations

import os

from config import RESULTS_DIR, ensure_dirs, set_seed
from evaluation import describe_splits, evaluate_candidate, load_dataset, split_masks, write_experiment_outputs
from model_zoo import compact_weight_decay_candidates, full_weight_decay_candidates


def main() -> None:
    ensure_dirs()
    seed = set_seed()
    df, features = load_dataset()
    masks = split_masks(df)
    mode = os.getenv("WEIGHT_DECAY_MODE", "compact").strip().lower()
    candidates = full_weight_decay_candidates() if mode == "full" else compact_weight_decay_candidates()
    print(
        "[weight_decay] seed=%s mode=%s candidates=%s features=%s splits=%s"
        % (seed, mode, len(candidates), len(features), describe_splits(masks))
    )

    rows = []
    val_predictions = []
    test_predictions = []
    for candidate in candidates:
        print("[weight_decay] train %s scheme=%s" % (candidate.name, candidate.scheme))
        cand_rows, val_pred, test_pred, _ = evaluate_candidate(
            df,
            features,
            masks,
            candidate,
            experiment="weight_decay",
            use_sample_weight=True,
            add_selective=True,
            save_model=False,
        )
        rows.extend(cand_rows)
        val_predictions.append(val_pred)
        test_predictions.append(test_pred)

    results = write_experiment_outputs("weight_decay", rows, val_predictions, test_predictions)
    results[results["Split"].eq("test")].sort_values(["F1_macro", "Accuracy", "AUC"], ascending=False).to_csv(
        RESULTS_DIR / "weight_decay_full_coverage.csv",
        index=False,
    )
    top = results[results["Split"].eq("test")].sort_values(["F1_macro", "Accuracy", "AUC"], ascending=False).head(12)
    print(top[["Model", "Scheme", "Accuracy", "F1_macro", "AUC", "Threshold", "ThresholdMode"]].to_string(index=False))


if __name__ == "__main__":
    main()
