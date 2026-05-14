# Oil Direction Data-Mining + ML Demo Workspace

This folder is a staged workspace for an impressive, honest demo around next-day
oil direction classification.

## What Is Staged Here

| Path | Purpose |
| --- | --- |
| `data/processed/` | Current processed datasets, including no-leak and scaled exports |
| `artifacts/improve_classification/` | Current unified classification reports, results, predictions, and saved models |
| `assets/eda_current/` | Forward-target-aware EDA figures and tables from the current pipeline |
| `assets/eda_legacy_presentation/` | Older EDA presentation assets, useful for contrast and storyboarding |
| `assets/ml_legacy_step5c/` | Earlier Step5C ML visuals/results, useful for historical comparison |
| `code/` | Source snapshots for preprocessing, EDA, and improve-classification logic |
| `docs/` | Copied project docs and summaries used to design the demo |
| `research/` | Benchmark and leakage-audit references |
| `DEMO_PLAN.md` | Recommended production website, CI/CD, and Capacitor mobile plan |
| `DEPLOYMENT.md` | DevOps/MLOps runbook for local, GitHub Actions, GCP, and APK operation |
| `scripts/` | Data generation, validation, and GCP/GitHub deploy identity setup |
| `backend/` | Express API service serving the generated ML artifact bundle |
| `web/` | React/Vite website plus Capacitor Android project |

## Core Demo Message

Use the current pipeline as the authoritative forecasting story:

```text
raw/integrated data -> feature engineering -> leakage cleanup -> deterministic processing
-> unified ML tournament -> decision/audit dashboard
```

The legacy `other_eda_preprocess` assets are useful as presentation material, but
they are not the final modeling contract because they use same-day framing and
carry known leakage/contamination risks documented in `docs/source_docs/`.

## Planned Product Shape

The updated plan targets a full production-style website and mobile app:

```text
React + Vite + TypeScript website
  -> GitHub Actions CI/CD
  -> Vercel or equivalent web deployment
  -> Capacitor Android/iOS wrappers from the same Vite build
```

The core app should use generated static JSON from the staged ML artifacts, so
the deployed website and mobile apps do not require Python or model training at
runtime.

## Implemented Product Stack

Current implementation status:

```text
Shared ML data bundle: demo/scripts/build_demo_data.py
MLOps validator: demo/scripts/validate_demo_data.py
Backend API: demo/backend
Frontend website: demo/web
Android app shell: demo/web/android
CI/CD workflows: .github/workflows
GCP deploy identity setup: demo/scripts/setup_gcp_github_actions.sh
Deployment runbook: demo/DEPLOYMENT.md
```

Local Android APK output after `npm run android:debug`:

```text
demo/web/android/app/build/outputs/apk/debug/app-debug.apk
```

## Current CI/CD Control Status

Local CLI access is ready for setup and operation:

```text
GitHub CLI account: trungdangtapcode
Repository: trungdangtapcode/cs313-oil-prediction
Repository permission: ADMIN
GCP CLI account: tcuong1000@gmail.com
GCP active project: husky-car
GCP active configuration: husky-car
GCP Workload Identity: configured for GitHub Actions
Artifact Registry: oil-signal-mine in us-central1
```

The remaining CI/CD work is not an auth-permission problem. It is setup work:

```text
1. Add .github/workflows/*.yml.
2. Add web/mobile build scripts.
3. Configure GitHub Actions deploy identity for GCP or the chosen host.
4. Trigger and watch runs with gh workflow run / gh run watch.
```

GitHub Actions does not inherit the local browser `gcloud` login. If a workflow
deploys to GCP, it needs Workload Identity Federation or a temporary service
account key stored as a GitHub secret.

## Best Current Result To Show

Use `artifacts/improve_classification/results/primary_test_leaderboard.csv`.

Latest staged ML report source:

```text
ml/improve-classification/REPORT.md
Generated at 2026-05-14 12:08:11 UTC
```

Current best primary classifier:

```text
ENS_FINAL3
Accuracy  = 0.5476
F1_macro  = 0.5406
AUC       = 0.5351
Test rows = 840
```

This should be framed as a realistic, above-no-skill result for a noisy daily
financial direction problem, not as a trading guarantee.
