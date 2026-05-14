#!/usr/bin/env python3
"""Baseline ML experiment: common model families on all features."""

from __future__ import annotations

import json

from config import RESULTS_DIR, ensure_dirs, set_seed
from evaluation import (
    describe_splits,
    evaluate_candidate,
    load_dataset,
    split_masks,
    write_experiment_outputs,
)
from model_zoo import baseline_candidates


def main() -> None:
    ensure_dirs()
    seed = set_seed()
    df, features = load_dataset()
    masks = split_masks(df)
    print("[baseline] seed=%s features=%s splits=%s" % (seed, len(features), describe_splits(masks)))

    rows = []
    val_predictions = []
    test_predictions = []
    thresholds = {}
    for candidate in baseline_candidates():
        print("[baseline] train %s" % candidate.name)
        cand_rows, val_pred, test_pred, bundle = evaluate_candidate(
            df,
            features,
            masks,
            candidate,
            experiment="baseline",
            use_sample_weight=False,
            add_selective=False,
            save_model=False,
        )
        rows.extend(cand_rows)
        val_predictions.append(val_pred)
        test_predictions.append(test_pred)
        thresholds[candidate.name] = bundle["threshold"]

    results = write_experiment_outputs("baseline", rows, val_predictions, test_predictions)
    (RESULTS_DIR / "baseline_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    top = results[results["Split"].eq("test")].sort_values(["F1_macro", "Accuracy", "AUC"], ascending=False).head(10)
    print(top[["Model", "Accuracy", "F1_macro", "AUC", "Threshold", "ThresholdMode"]].to_string(index=False))


if __name__ == "__main__":
    main()
