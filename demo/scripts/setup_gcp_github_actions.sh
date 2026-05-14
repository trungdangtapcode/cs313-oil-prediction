#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-husky-car}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-trungdangtapcode/cs313-oil-prediction}"
POOL_ID="${POOL_ID:-github-actions-pool}"
PROVIDER_ID="${PROVIDER_ID:-github-provider}"
SERVICE_ACCOUNT_ID="${SERVICE_ACCOUNT_ID:-github-actions-deployer}"
GAR_REPOSITORY="${GAR_REPOSITORY:-oil-signal-mine}"

echo "Using project: ${PROJECT_ID}"
echo "Using region: ${REGION}"
echo "Using repo: ${REPO}"

PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_ID}@${PROJECT_ID}.iam.gserviceaccount.com"
POOL_RESOURCE="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}"
PROVIDER_RESOURCE="${POOL_RESOURCE}/providers/${PROVIDER_ID}"

gcloud services enable \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  serviceusage.googleapis.com \
  --project "${PROJECT_ID}"

if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SERVICE_ACCOUNT_ID}" \
    --project "${PROJECT_ID}" \
    --display-name "GitHub Actions deployer for Oil Signal Mine"
fi

for role in \
  roles/run.admin \
  roles/artifactregistry.admin \
  roles/iam.serviceAccountUser \
  roles/serviceusage.serviceUsageAdmin
do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role "${role}" \
    --quiet >/dev/null
done

if ! gcloud iam workload-identity-pools describe "${POOL_ID}" \
  --project "${PROJECT_ID}" \
  --location global >/dev/null 2>&1; then
  gcloud iam workload-identity-pools create "${POOL_ID}" \
    --project "${PROJECT_ID}" \
    --location global \
    --display-name "GitHub Actions Pool"
fi

if ! gcloud iam workload-identity-pools providers describe "${PROVIDER_ID}" \
  --project "${PROJECT_ID}" \
  --location global \
  --workload-identity-pool "${POOL_ID}" >/dev/null 2>&1; then
  gcloud iam workload-identity-pools providers create-oidc "${PROVIDER_ID}" \
    --project "${PROJECT_ID}" \
    --location global \
    --workload-identity-pool "${POOL_ID}" \
    --display-name "GitHub OIDC Provider" \
    --issuer-uri "https://token.actions.githubusercontent.com" \
    --attribute-mapping "google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
    --attribute-condition "assertion.repository == '${REPO}'"
fi

gcloud iam service-accounts add-iam-policy-binding "${SERVICE_ACCOUNT_EMAIL}" \
  --project "${PROJECT_ID}" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/${POOL_RESOURCE}/attribute.repository/${REPO}" \
  --quiet >/dev/null

if ! gcloud artifacts repositories describe "${GAR_REPOSITORY}" \
  --location "${REGION}" \
  --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${GAR_REPOSITORY}" \
    --repository-format docker \
    --location "${REGION}" \
    --description "Oil Signal Mine demo containers" \
    --project "${PROJECT_ID}"
fi

gh variable set GCP_PROJECT_ID --repo "${REPO}" --body "${PROJECT_ID}"
gh variable set GCP_REGION --repo "${REPO}" --body "${REGION}"
gh variable set GAR_REPOSITORY --repo "${REPO}" --body "${GAR_REPOSITORY}"
gh variable set GCP_WORKLOAD_IDENTITY_PROVIDER --repo "${REPO}" --body "${PROVIDER_RESOURCE}"
gh variable set GCP_SERVICE_ACCOUNT --repo "${REPO}" --body "${SERVICE_ACCOUNT_EMAIL}"

cat <<EOF

GCP/GitHub Actions deploy identity is configured.

Project: ${PROJECT_ID}
Region: ${REGION}
Artifact Registry: ${GAR_REPOSITORY}
Provider: ${PROVIDER_RESOURCE}
Service account: ${SERVICE_ACCOUNT_EMAIL}

Next commands:
  gh workflow run deploy-gcp.yml --repo ${REPO}
  gh run watch --repo ${REPO}
EOF
