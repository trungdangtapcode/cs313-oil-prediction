# Best Demo Plan: Oil Direction Signal Mine

## 1. Product Concept

Build **Oil Direction Signal Mine** as a production-quality web app plus mobile
app shell.

The theme is data-mining and machine learning: the app shows how noisy market,
macro, supply, and geopolitical/news signals are mined, cleaned, filtered,
audited, modeled, and finally turned into a next-day UP/DOWN probability.

The strongest message:

```text
Most daily oil direction signal is weak.
The impressive part is not fake 80% accuracy.
The impressive part is a polished, auditable product that mines weak signal
without leaking future information.
```

## 2. Target Output

This should become a real deployable product, not a notebook-style dashboard.

| Surface | Target |
| --- | --- |
| Website | React/Vite static SPA deployed by CI/CD |
| Mobile | Capacitor Android/iOS apps wrapping the same web build |
| Data | Static generated JSON derived from latest ML artifacts |
| CI/CD | GitHub Actions for test, web deploy, Android build, iOS build |
| Core mode | Offline historical demo from staged artifacts |
| Live mode | Backend-connected historical replay and edited-field scorer |
| Trading mode | Walk-forward strategy research with lagged execution, costs, threshold profit sweep, and Buy & Hold benchmark |

## 3. Authoritative Local Findings

Use these facts throughout the demo:

| Topic | Demo Fact |
| --- | --- |
| Problem | Predict `oil_return_fwd1 > 0` for next trading day direction |
| Split | train `< 2022-01-01`, validation `2022`, final test `>= 2023-01-01` |
| Data size | `2,922` rows, `27` final features, `840` final test rows |
| Latest staged report | `ml/improve-classification/REPORT.md`, generated `2026-05-14 12:08:11 UTC` |
| Current best model | `ENS_FINAL3`, average-probability ensemble |
| Best metrics | Accuracy `0.5476`, F1_macro `0.5406`, AUC `0.5351` |
| Best AUC config | `XGB_linear03`, AUC `0.5596`, but weaker F1 |
| EDA result | Signal exists but is weak; no feature has large class-separation effect |
| Risk framing | Results above `0.60-0.63` should trigger strong leakage audit |

### Current CI/CD Control Status

As of the latest local check, the machine has enough local authority to set up
and operate CI/CD, but the repository still needs workflow files and a GitHub
Actions deploy identity.

Current local control:

| Area | Status |
| --- | --- |
| GitHub CLI | Authenticated as `trungdangtapcode` |
| Repository | `trungdangtapcode/cs313-oil-prediction` |
| Repo permission | `ADMIN` |
| Git protocol | SSH configured by `gh` |
| GCP CLI account | `tcuong1000@gmail.com` |
| GCP active project | `husky-car` |
| GCP active config | `husky-car` |

Current gaps before real CI/CD can run:

- No GitHub Actions workflows are currently listed for this repository.
- GitHub-hosted runners do not inherit the local `gcloud` browser login.
- The workflow needs its own GCP deploy identity, preferably Workload Identity
  Federation, or a service account key stored as a GitHub Actions secret.
- Local Application Default Credentials may still need separate reauthentication
  if Python/Vertex/local SDK tooling must use `husky-car`.

Practical meaning:

- Local deploys can use the current `gcloud` login.
- I can create workflow files, push branches, set GitHub secrets/variables,
  trigger workflows, watch logs, and fix failures with the current `gh` access.
- GitHub Actions GCP deployment starts working only after the deploy identity is
  wired into the workflow.

## 4. Stack Choice

Use a single **React + Vite + TypeScript** app as the source of truth.

Use **Capacitor** to package that same app for iOS and Android. Capacitor is a
web-focused native runtime for building mobile apps from modern web tooling,
while still allowing native APIs when needed.

Recommended libraries:

| Area | Choice |
| --- | --- |
| Web framework | React + Vite + TypeScript |
| Mobile wrapper | Capacitor |
| Styling | Tailwind CSS with local components |
| Charts | Plotly.js or Apache ECharts |
| Tables | TanStack Table |
| State | Zustand or plain React context |
| Tests | Vitest, React Testing Library, Playwright |
| Data generation | Python script converts CSV/JSON/PNG references into web-ready JSON |
| Web hosting | Vercel first choice; Cloudflare Pages is also fine |
| CI/CD | GitHub Actions |

Why not Streamlit now:

- Streamlit is good for fast internal dashboards.
- It is not the right base for a real website plus Capacitor mobile app.
- A static React app can be hosted cheaply, tested normally, and wrapped by Capacitor.

## 5. Proposed Directory Layout

```text
demo/
  DEMO_PLAN.md
  MANIFEST.md
  README.md
  artifacts/
    improve_classification/
      results/
      models/
  assets/
    eda_current/
    eda_legacy_presentation/
    ml_legacy_step5c/
  data/
    processed/
  scripts/
    build_demo_data.py
    validate_demo_data.py
  web/
    package.json
    package-lock.json
    vite.config.ts
    tsconfig.json
    capacitor.config.ts
    index.html
    public/
      data/
        demo_manifest.json
        leaderboard.json
        prediction_wide_test.json
        feature_ranking.json
        threshold_diagnostics.json
        confidence_curve.json
      assets/
        eda/
        ml/
    src/
      app/
      components/
      charts/
      data/
      pages/
      theme/
      utils/
    android/
    ios/
  .github/
    workflows/
      ci.yml
      deploy-web.yml
      mobile-android.yml
      mobile-ios.yml
      release.yml
```

Native Capacitor folders (`android/`, `ios/`) should be committed after
initialization because they carry app config, permissions, signing settings, and
native customizations.

## 6. Data Strategy

Do not load large CSVs directly in the browser for the first production version.
Generate compact JSON files at build time.

Source artifacts:

- `artifacts/improve_classification/results/primary_test_leaderboard.csv`
- `artifacts/improve_classification/results/ml_test_predictions.csv`
- `artifacts/improve_classification/results/dl_test_predictions.csv`
- `artifacts/improve_classification/results/ensemble_test_predictions.csv`
- `artifacts/improve_classification/results/feature_ranking.csv`
- `artifacts/improve_classification/results/selected_features.json`
- `artifacts/improve_classification/results/threshold_diagnostics.csv`
- `artifacts/improve_classification/results/selective_coverage_diagnostics.csv`
- `assets/eda_current/tables/*.csv`
- `data/processed/dataset_final_noleak_step5c_scaler.csv`

Generated web data:

| File | Purpose |
| --- | --- |
| `demo_manifest.json` | Report timestamp, source checksums, row counts, app data version |
| `leaderboard.json` | Primary model leaderboard and best-by-experiment summary |
| `prediction_wide_test.json` | One row per date with probability columns per model |
| `ens_final3_decision_log.json` | Date, target, proba, pred, correct, confidence margin |
| `feature_ranking.json` | Feature ranking, groups, transforms, selected-feature flags |
| `dataset_evolution.json` | Row/column counts for processing stages |
| `leakage_audit_table.json` | Risk, affected columns, action, rationale |
| `threshold_curve.json` | Threshold vs accuracy/F1/pos-rate for selected models |
| `confidence_curve.json` | Confidence margin vs coverage/metrics |
| `asset_index.json` | PNG paths and captions used by the UI |
| `trading_strategy_summary.json` | XGB/RF/Ridge/Buy & Hold backtest with 1-day execution lag, 0.15% turnover cost, Sharpe/Sortino/MDD, yearly returns, and trading-threshold profit sweep |

The web/mobile app should read only `web/public/data/*.json` and static assets.
Model loading remains optional because saved model dependencies can vary by
machine. The validated historical demo should run without Python.

## 7. App Navigation

Use a dense but polished product interface with left navigation:

1. `Mission Control`
2. `Data Mine`
3. `Signal Refinery`
4. `Model Arena`
5. `Decision Microscope`
6. `Confidence Lab`
7. `Leakage Audit`
8. `Demo Script`

For mobile, this becomes a bottom tab bar for the first five pages plus a
secondary menu for audit/script pages.

## 8. Page Designs

### Page 1: Mission Control

Purpose: explain the whole story in 30 seconds.

Show:

- KPI cards: best model, accuracy, F1_macro, AUC, test rows, target UP rate.
- Horizontal pipeline:
  `Raw Sources -> Cleaning -> Integration -> Feature Engineering -> Leakage Cleanup -> Processing -> ML Tournament -> Decision`.
- Compact top-5 leaderboard.
- "Do not overclaim" card: realistic above-no-skill result, not a trading guarantee.
- Data version card from `demo_manifest.json`.

Impressive element:

- Clickable pipeline stages that reveal inputs, outputs, row/feature counts, and leakage controls.

### Page 2: Data Mine

Purpose: show the data-mining foundation.

Show:

- Source groups: market, FRED/macro, EIA/supply, GDELT/news tone, ACLED/conflict, calendar.
- Dataset evolution table:
  `step4_transformed -> step4_noleak -> final_noleak -> step5b processed -> step5c scaled`.
- Current EDA images:
  `step5_upgraded_00_data_quality_overview.png`
  and `step5_upgraded_03_target_over_time.png`.
- Toggle: `Legacy EDA view` vs `Forecasting-safe current view`.

Impressive element:

- Feature-group matrix with columns `raw`, `engineered`, `kept`, `dropped`, `risk`.

### Page 3: Signal Refinery

Purpose: show how noisy features are ranked and selected.

Show:

- Feature ranking from `feature_ranking.json`.
- Current EDA signal chart `step5_upgraded_06_signal_scores.png`.
- Selected feature set from `selected_features.json`.
- Top groups: calendar, supply, conflict, GDELT, market.

Interactions:

- Top-N slider.
- Ranking selector: MI, Spearman, mixed score, research score.
- Feature detail drawer with dictionary text and transform/risk status.

Impressive element:

- Visual feature refinery: raw features are filtered by leakage, stability, and signal score.

### Page 4: Model Arena

Purpose: make the model tournament visually compelling.

Show:

- Primary leaderboard.
- Scatter plot: `AUC` vs `F1_macro`, size by `Accuracy`, color by experiment.
- Best-by-experiment cards:
  baseline, feature selection, weight decay, ensemble, deep learning.
- Comparison of `ENS_FINAL3`, `LGBM_exp100`, `DL_GRU_L40`, `XGB_linear03`.

Interactions:

- Filter by experiment family.
- Sort by F1, Accuracy, AUC, MCC, Brier.
- Click a model to open confusion stats and probability distribution.

Impressive element:

- Rank movement when switching metric priority.

### Page 5: Decision Microscope

Purpose: inspect one prediction date.

Show:

- Date slider over the 2023+ test period.
- Actual target direction and selected-model probability.
- Correct/incorrect state, UP/DOWN prediction, confidence margin.
- Model agreement strip across top models.
- Top feature values for the date vs historical percentiles.

Interactions:

- Model selector.
- Date selector.
- Threshold control that shows when a prediction flips.
- Replay mode over the test period.

Impressive element:

- Timeline replay of probability, actual direction, correctness, and confidence.

### Page 6: Confidence Lab

Purpose: explain threshold and coverage tradeoffs honestly.

Show:

- Threshold diagnostics.
- Selective coverage diagnostics.
- Coverage vs accuracy/F1 chart.
- Correct vs incorrect probability histograms.

Key lesson:

Validation-selected thresholds helped only a minority of configs, so fixed `0.5`
is the clean primary leaderboard policy.

Impressive element:

- Confidence margin slider showing covered days and metric movement.

### Page 7: Leakage Audit

Purpose: prove engineering rigor.

Show:

- What was dropped and why.
- Legacy EDA vs current forecasting-safe comparison.
- High-risk historical columns:
  `cpi_lag`, `unemployment_lag`, `real_rate`, `fed_rate_change`,
  `fed_rate_regime`, `geopolitical_stress_index`, `oil_volatility_7d`.
- Metric contract from `metric_contract.json`.

Key lesson:

The older EDA branch is useful for presentation and descriptive analysis, but
the current demo uses the forward-target no-leak pipeline for modeling.

Impressive element:

- Leakage checklist with each risk marked as dropped, transformed, deferred to
model-time preprocessing, or explicitly caveated.

### Page 8: Demo Script

Purpose: make the live presentation easy.

Show:

- 5-minute path.
- 12-minute path.
- Q&A answers:
  why accuracy is not 80%;
  why ensemble wins F1;
  why AUC winner is not final classifier;
  why threshold-selected rows are diagnostics only;
  why legacy EDA is not the final training contract.

## 9. Mobile App Plan

Capacitor should wrap the exact same Vite build:

```text
npm run build
npx cap sync
```

Capacitor config:

```ts
const config = {
  appId: "ai.oildirection.signalmine",
  appName: "Oil Direction Signal Mine",
  webDir: "dist",
  server: {
    androidScheme: "https"
  }
}
```

Mobile-specific work:

- Responsive layouts for phone and tablet.
- Bottom navigation on small screens.
- Safe-area handling for iOS.
- Dark mode status bar.
- Offline packaged data from `public/data`.
- External links open in system browser.
- Disable unsupported live-data features unless network mode is explicitly added.

Native feature ideas for later:

- Save/share a one-page model report.
- Push notification for new model run availability.
- Local file export for selected date explanation.

Do not add native features until the web product is stable.

## 10. CI/CD Plan

The CI/CD plan should be implemented as an operator-ready setup, not only as
YAML files. The local machine has `gh` and `gcloud` authenticated, so the setup
can be driven from this environment:

```bash
gh auth status
gh repo view --json nameWithOwner,viewerPermission,defaultBranchRef
gcloud config list
gcloud projects describe husky-car
```

The correct model is:

```text
Local auth lets us create and operate CI/CD.
GitHub Actions needs its own GCP deploy identity.
```

### GCP Deploy Identity

Preferred production approach: Workload Identity Federation.

Why:

- No long-lived service account JSON key in GitHub secrets.
- GitHub Actions receives short-lived credentials.
- Access can be restricted to this repository, branch, environment, and service
  account.

Target setup:

```text
GCP project: husky-car
Service account: github-actions-deployer@husky-car.iam.gserviceaccount.com
GitHub repo: trungdangtapcode/cs313-oil-prediction
GitHub environment: production
```

Minimum roles should depend on the chosen deploy target:

| Target | Likely roles |
| --- | --- |
| Static web on Vercel | No GCP role needed for web deploy; Vercel secrets only |
| GCS static hosting | `roles/storage.admin` scoped to the website bucket if possible |
| Cloud Run | `roles/run.admin`, `roles/iam.serviceAccountUser`, artifact registry permissions |
| Firebase Hosting | Firebase hosting deploy permissions |
| Vertex/ML artifact publishing | Narrow storage/artifact permissions only |

Fallback approach: service account key as a GitHub secret.

Use only if speed matters more than key hygiene:

```bash
gh secret set GCP_PROJECT_ID --body "husky-car"
gh secret set GCP_SA_KEY < key.json
```

If this fallback is used, rotate/delete the key after the demo or replace it
with Workload Identity Federation.

### GitHub Actions Operations

After workflows exist, I can operate CI/CD with:

```bash
gh workflow list
gh workflow run ci.yml
gh workflow run deploy-web.yml
gh run list
gh run watch
gh run view --log
gh secret list
gh variable list
```

For repo-level setup:

```bash
gh secret set VERCEL_TOKEN
gh secret set VERCEL_ORG_ID
gh secret set VERCEL_PROJECT_ID
gh variable set GCP_PROJECT_ID --body "husky-car"
```

For branch and environment protection, use GitHub settings or `gh api` once the
exact policy is chosen.

### `ci.yml`

Runs on every PR and push.

Jobs:

- Checkout.
- Setup Node.
- `npm ci`.
- `npm run data:build`.
- `npm run lint`.
- `npm run typecheck`.
- `npm test`.
- `npm run build`.
- `npx playwright install --with-deps chromium`.
- `npm run test:e2e`.
- Upload built `dist/` and Playwright report artifacts.

Required gate:

- PR cannot merge unless CI passes.

### `deploy-web.yml`

Runs on `main` and manual dispatch.

Recommended target: Vercel.

Jobs:

- Reuse build/test steps or depend on `ci.yml`.
- Deploy preview for PRs if desired.
- Deploy production for `main`.

Secrets:

- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

Alternative:

- Cloudflare Pages with `CLOUDFLARE_API_TOKEN` and account/project secrets.
- GitHub Pages if static-only and no preview requirements.

### `mobile-android.yml`

Runs on `main`, version tags, and manual dispatch.

Jobs:

- Setup Node.
- Setup Java.
- `npm ci`.
- `npm run data:build`.
- `npm run build`.
- `npx cap sync android`.
- Gradle lint/test.
- Build debug APK for PR smoke artifacts.
- Build release AAB for tags/manual release.
- Sign release with encrypted keystore secrets.
- Upload APK/AAB artifacts.

Secrets:

- `ANDROID_KEYSTORE_BASE64`
- `ANDROID_KEYSTORE_PASSWORD`
- `ANDROID_KEY_ALIAS`
- `ANDROID_KEY_PASSWORD`

### `mobile-ios.yml`

Runs on version tags and manual dispatch.

Jobs:

- macOS runner.
- Setup Node.
- Setup Xcode.
- `npm ci`.
- `npm run data:build`.
- `npm run build`.
- `npx cap sync ios`.
- Install CocoaPods if needed.
- Xcode archive.
- Export signed IPA.
- Upload IPA artifact or submit to TestFlight.

Secrets depend on signing strategy, but usually include:

- `APPLE_TEAM_ID`
- `ASC_KEY_ID`
- `ASC_ISSUER_ID`
- `ASC_API_KEY_P8`
- certificate/provisioning profile secrets, or a managed signing tool setup

iOS automation is feasible but requires Apple Developer account setup and careful
secret management.

### `release.yml`

Runs on tags like `demo-v1.0.0`.

Actions:

- Build and deploy production web.
- Build Android release AAB.
- Build iOS archive/IPA.
- Generate release notes from `demo_manifest.json`.
- Attach artifacts to GitHub Release.

## 11. Package Scripts

Target `demo/web/package.json` scripts:

```json
{
  "scripts": {
    "dev": "vite",
    "data:build": "python ../scripts/build_demo_data.py",
    "data:validate": "python ../scripts/validate_demo_data.py",
    "lint": "eslint .",
    "typecheck": "tsc --noEmit",
    "test": "vitest run",
    "test:e2e": "playwright test",
    "build": "vite build",
    "preview": "vite preview",
    "cap:init": "cap init",
    "cap:sync": "npm run build && cap sync",
    "android": "npm run cap:sync && cap open android",
    "ios": "npm run cap:sync && cap open ios"
  }
}
```

## 12. Implementation Phases

### Phase 0: Data Version Lock

Deliverable:

- `demo/scripts/build_demo_data.py`
- `demo/scripts/validate_demo_data.py`
- `demo/web/public/data/demo_manifest.json`

Tasks:

- Convert staged CSVs to compact JSON.
- Store source file checksums and report timestamp.
- Validate best model/metric values match latest report.
- Fail CI if required artifacts are missing or stale.

### Phase 1: Web App MVP

Deliverable:

- `demo/web` Vite app with all eight pages.

Tasks:

- Build layout/nav/theme.
- Build JSON data client.
- Build reusable KPI, chart, table, image, and audit components.
- Implement core pages using generated data.
- Add responsive desktop/tablet/mobile behavior.

Success criteria:

- `npm run build` succeeds.
- App works without Python server.
- All charts load from `public/data`.
- Demo can be presented in 5 minutes.

### Phase 2: Web CI/CD

Deliverable:

- GitHub Actions CI, deploy identity, and web deployment.

Tasks:

- Keep `husky-car` as the active GCP project for setup:
  `gcloud config configurations activate husky-car`.
- Create `.github/workflows/ci.yml`.
- Add lint/typecheck/unit/build/Playwright workflows.
- Add Vercel deployment workflow.
- Add `deploy-web.yml` with manual dispatch and `main` deployment.
- Configure GitHub repository variables/secrets with `gh secret` and
  `gh variable`.
- Configure GCP deploy identity for GitHub Actions if the deploy target uses
  GCP. Prefer Workload Identity Federation; use a service account key only as a
  temporary fallback.
- Protect production deployment with GitHub environments.
- Trigger and watch the first run with `gh workflow run` and `gh run watch`.

Success criteria:

- `gh workflow list` shows CI and deploy workflows.
- Every PR gets CI.
- Main branch deploys the website.
- GitHub Actions can authenticate to any required deploy platform without using
  the local browser login.
- Deployment shows the latest staged ML report timestamp.

### Phase 3: Capacitor Mobile Shells

Deliverable:

- Android and iOS projects generated under `demo/web`.

Tasks:

- Install Capacitor.
- Add Android and iOS platforms.
- Tune safe areas and mobile navigation.
- Verify offline packaged data works.
- Add Android CI build.
- Add iOS CI archive once Apple signing exists.

Success criteria:

- Android APK builds in CI.
- iOS archive builds on macOS runner with signing configured.
- Web and mobile use the same data/version manifest.

### Phase 4: Explainability Layer

Deliverable:

- Local decision explanations that do not require runtime model loading.

Tasks:

- Use feature ranks, percentiles, model disagreement, and optional precomputed SHAP/permutation artifacts.
- Add explanation panel to Decision Microscope.
- Label explanations as diagnostic, not causal.

Success criteria:

- Each selected date has a clear reason/context panel.
- Slow explanation work is precomputed.

### Phase 5: Optional Live/As-Of Prototype

Deliverable:

- Explicitly labeled live-data mode.

Tasks:

- Add optional official-source connectors.
- Cache API calls.
- Keep live data separate from validated historical leaderboard.
- Show warnings when live preprocessing cannot exactly match historical artifacts.

Use official sources:

- EIA Open Data API.
- FRED API.
- GDELT API.
- Cboe VIX historical data if needed.

## 13. Visual Style

Use a restrained operational interface:

- Dark charcoal background.
- Oil amber for commodity identity.
- Green/red only for UP/DOWN state.
- Cyan/teal for data-mining accents.
- Dense but readable dashboards.
- No decorative bokeh/orb backgrounds.
- Cards only for repeated items, KPI tiles, dialogs, and framed tools.

Charts:

- Leaderboard bars.
- Metric scatter.
- Timeline bands.
- Feature group heatmaps.
- Probability river.
- Coverage tradeoff line.
- Confusion matrix tiles.

## 14. Core Technical Risks

| Risk | Mitigation |
| --- | --- |
| Demo assets drift from latest ML results | `validate_demo_data.py` checks report timestamp and expected metrics |
| Browser loading huge CSVs is slow | Generate compact JSON and sample where appropriate |
| Saved model dependencies differ by environment | First app uses saved predictions, not runtime model inference |
| Local GCP auth is confused with CI auth | Document that GitHub Actions needs a separate deploy identity |
| Service account key leaks | Prefer Workload Identity Federation; rotate/delete any temporary key |
| Mobile package size grows too large | Keep data compact; do not package raw model files unless needed |
| iOS CI signing is hard | Make Android CI first; add iOS after Apple credentials exist |
| Legacy EDA visuals imply same-day target | Label as legacy/presentation-only |
| Users overinterpret 54.8% accuracy | Keep benchmark and leakage-audit cards visible |
| Live APIs break demo | Core app remains offline historical mode |

## 15. Acceptance Criteria

The plan is complete when:

- Website builds with `npm run build`.
- Website deploys automatically from `main`.
- PRs run lint, typecheck, unit tests, build, and Playwright smoke tests.
- `gh workflow list` shows runnable CI/CD workflows.
- `gh run watch` can monitor workflow execution from this machine.
- GitHub Actions deploy auth is configured separately from local `gcloud` auth.
- GCP-targeted workflows use `husky-car` as the project.
- Android APK/AAB builds in CI.
- iOS archive workflow exists and is ready for signing secrets.
- Web and mobile share the same Vite source and generated JSON data.
- The app displays latest ML report timestamp `2026-05-14 12:08:11 UTC`.
- The app never claims performance beyond the validated historical report.

## 16. Suggested Commands After Implementation

```bash
cd demo/web
npm install
npm run data:build
npm run dev
npm run build
npm run cap:sync
```

For Android after Capacitor setup:

```bash
cd demo/web
npm run android
```

For iOS after Capacitor setup on macOS:

```bash
cd demo/web
npm run ios
```

For CI/CD operation after workflows are committed:

```bash
gh workflow list
gh workflow run ci.yml
gh run watch
gh run view --log
```

For fast GCP project/account switching:

```bash
gcloud config configurations activate husky-car
gcloud config configurations list
```

To switch back to the previous saved profile later, after `paul@igot.ai` is
reauthenticated if needed:

```bash
gcloud config configurations activate igot-studio
```

## 17. References

- Capacitor docs: https://capacitorjs.com/docs
- GitHub Actions workflow syntax: https://docs.github.com/actions/reference/workflows-and-actions/workflow-syntax
- GitHub Actions deployment docs: https://docs.github.com/en/actions/how-tos/deploy
- Vercel deployment docs: https://vercel.com/docs/deployments/deployment-methods
- Vercel GitHub docs: https://vercel.com/docs/deployments/git/vercel-for-github
- Plotly Python/docs reference for chart design: https://plotly.com/python/
- scikit-learn permutation importance: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
- SHAP beeswarm explanation docs: https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html
- EIA Open Data: https://www.eia.gov/opendata/
- FRED API docs: https://fred.stlouisfed.org/docs/api/fred/
- GDELT Cloud API docs: https://docs.gdeltcloud.com/api-reference/v2
