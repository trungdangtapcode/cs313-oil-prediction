# Improve Classification

Goal: compare baseline ML, feature selection, weight decay, ensemble, and deep
learning under one chronological setup and one metric contract.

## One Setup

- Dataset: `data/processed/dataset_final_noleak_step5c_scaler.csv`
- Target: `oil_return_fwd1 > 0` means UP=1, otherwise DOWN=0
- Train for validation: target date `< 2022-01-01`
- Validation/model-threshold selection: `2022-01-01 <= target date < 2023-01-01`
- Final refit: target date `< 2023-01-01`
- Final test: target date `>= 2023-01-01`
- Main comparison: full coverage only, `Split=test`, `Coverage=1.0`

## File Map

The code is organized by responsibility:

| File | Meaning |
| --- | --- |
| `config.py` | Paths, split dates, constants, runtime helpers |
| `evaluation.py` | Data loading, target/split logic, metric calculation, reusable train/evaluate helper |
| `model_zoo.py` | Model factories, candidate lists, sample-weight schemes |
| `exp_*.py` | One experiment per file |
| `run.py` | Runs one or more experiments |
| `report.py` | Builds one final `REPORT.md` and unified metric tables |
| `REPORT.md` | Final report, explanation, interpretation, and leaderboard |
| `INTERPRETATION.md` | Full explanation of every evaluated model/config, val/test row counts, and how to read metrics |

Experiments:

| Experiment | Command | What It Tests |
| --- | --- | --- |
| `final_baselines` | `python ml/improve-classification/run.py --steps final_baselines` | Historical final results copied as reference |
| `baseline` | `python ml/improve-classification/run.py --steps baseline` | Basic model families on all features |
| `feature_selection` | `python ml/improve-classification/run.py --steps feature_selection` | MI/Spearman feature ranking, subset selection, retraining |
| `weight_decay` | `python ml/improve-classification/run.py --steps weight_decay` | Recency sample-weight schemes |
| `ensemble` | `python ml/improve-classification/run.py --steps ensemble` | Average-probability ensembles from previous experiment predictions |
| `deep_learning` | `python ml/improve-classification/run.py --steps deep_learning` | PyTorch MLP/GRU sequence models |
| `report` | `python ml/improve-classification/run.py --steps report` | Final report and unified metric tables |

## Useful Commands

Run report only:

```bash
python ml/improve-classification/run.py --steps report
```

Run ML experiments without DL:

```bash
python ml/improve-classification/run.py --steps final_baselines,baseline,feature_selection,weight_decay,ensemble,report
```

Run all experiments:

```bash
.ml-venv/bin/python ml/improve-classification/run.py --steps all
```

Run all except DL:

```bash
python ml/improve-classification/run.py --steps all --skip-dl
```

## Outputs

Generated outputs are ignored by Git:

- `logs/`
- `models/`
- `results/`

Important generated tables:

- `results/ml_results.csv`: combined ML experiment rows
- `results/dl_results.csv`: DL rows
- `results/primary_test_leaderboard.csv`: main non-duplicated leaderboard, one row per model configuration
- `results/all_threshold_test_metrics.csv`: validation-threshold and fixed-threshold diagnostics
- `results/unified_full_coverage_metrics.csv`: primary leaderboard plus historical reference rows
- `results/strict_unified_full_coverage_metrics.csv`: same primary current leaderboard kept for compatibility
- `results/metric_contract.json`: machine-readable metric and comparability rules

`REPORT.md` is kept in Git because it is the human-readable final summary.
`INTERPRETATION.md` is kept in Git because it explains all model/config counts and val/test evaluation rows.

## How To Read The Report

- Start with `INTERPRETATION.md` if you want the plain-language explanation.
- `current_unified_test_run`: actually evaluated by this folder on final test.
- `saved_final_test_result`: historical final baseline, included only as reference.
- Main leaderboard uses `Split=test`, `Coverage=1.0`, and `ThresholdMode=fixed_0.5`.
- Validation-selected thresholds are kept only in `threshold_diagnostics.csv`.
- Historical rows are useful context, but not a full one-to-one metric match.
