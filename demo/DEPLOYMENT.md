# Oil Signal Mine Deployment Runbook

## Current Control Plane

Local setup has been completed for the `husky-car` GCP project.

```text
GitHub repo: trungdangtapcode/cs313-oil-prediction
GCP project: husky-car
GCP region: us-central1
Artifact Registry repo: oil-signal-mine
Deploy service account: github-actions-deployer@husky-car.iam.gserviceaccount.com
Workload Identity provider: projects/433856687458/locations/global/workloadIdentityPools/github-actions-pool/providers/github-provider
```

GitHub repository variables have been configured:

```text
GCP_PROJECT_ID=husky-car
GCP_REGION=us-central1
GAR_REPOSITORY=oil-signal-mine
GCP_SERVICE_ACCOUNT=github-actions-deployer@husky-car.iam.gserviceaccount.com
GCP_WORKLOAD_IDENTITY_PROVIDER=projects/433856687458/locations/global/workloadIdentityPools/github-actions-pool/providers/github-provider
```

## Local Verification

From the repo root:

```bash
python3 demo/scripts/build_demo_data.py
python3 demo/scripts/validate_demo_data.py

cd demo/backend
npm ci
npm run typecheck
npm test
npm run build

cd ../web
npm ci
npm run typecheck
npm test
npm run build
npm run test:e2e
```

Build the local Android debug APK:

```bash
cd demo/web
ANDROID_HOME=/home/tcuong1000/android-sdk ANDROID_SDK_ROOT=/home/tcuong1000/android-sdk npm run android:debug
```

APK output:

```text
demo/web/android/app/build/outputs/apk/debug/app-debug.apk
```

## GitHub Actions

Workflows:

```text
.github/workflows/ci.yml
.github/workflows/deploy-gcp.yml
.github/workflows/android-apk.yml
.github/workflows/mlops.yml
.github/workflows/release.yml
```

After pushing these files to GitHub:

```bash
gh workflow list --repo trungdangtapcode/cs313-oil-prediction
gh workflow run ci.yml --repo trungdangtapcode/cs313-oil-prediction
gh workflow run android-apk.yml --repo trungdangtapcode/cs313-oil-prediction
gh workflow run deploy-gcp.yml --repo trungdangtapcode/cs313-oil-prediction
gh run watch --repo trungdangtapcode/cs313-oil-prediction
```

## GCP Deployment

`deploy-gcp.yml` builds two containers and deploys both to Cloud Run:

```text
oil-signal-mine-backend
oil-signal-mine-web
```

Current deployed services:

```text
Frontend: https://oil-signal-mine-web-oeo4vhl7ya-uc.a.run.app
Backend: https://oil-signal-mine-backend-oeo4vhl7ya-uc.a.run.app
Backend health: https://oil-signal-mine-backend-oeo4vhl7ya-uc.a.run.app/health
```

Deploy flow:

```text
Build/validate ML data
Authenticate through GitHub OIDC -> GCP Workload Identity
Build backend Docker image
Push backend image to Artifact Registry
Deploy backend to Cloud Run
Read backend URL
Build frontend Docker image with VITE_API_BASE_URL
Push frontend image to Artifact Registry
Deploy frontend to Cloud Run
```

## MLOps Gates

The generated JSON data bundle is treated as a versioned ML artifact.

Required checks:

```text
Latest report timestamp: 2026-05-14 12:08:11 UTC
Best primary model: ENS_FINAL3_th05
Accuracy: 0.5476190476190477
F1 macro: 0.5405766396462786
AUC: 0.535119843987392
Test rows: 840
```

The validation script fails CI if those values drift unexpectedly or if source
hashes no longer match the staged artifacts.

## Fast Account Switching

Stay on Husky for deploy work:

```bash
gcloud config configurations activate husky-car
gcloud config list
```

Switch back later:

```bash
gcloud config configurations activate igot-studio
```

If `paul@igot.ai` needs reauth:

```bash
gcloud auth login paul@igot.ai --launch-browser
gcloud config configurations activate igot-studio
gcloud config set project igot-studio
```
