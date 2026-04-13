#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PY="${ROOT_DIR}/.venv/bin/python"
LOG_DIR="${ROOT_DIR}/ml/classification/results/logs"

mkdir -p "${LOG_DIR}" /tmp/mpl

if [[ ! -x "${VENV_PY}" ]]; then
  echo "Missing virtualenv python at ${VENV_PY}" >&2
  exit 1
fi

steps=(
  "step1_train_baseline.py"
  "step2_finetune_ensemble.py"
  "step3_technical_improve.py"
  "step4_select_and_train.py"
  "step5_smart_selection.py"
  "step6_weight_decay.py"
  "step7_xgb_vs_gbm.py"
)

for step in "${steps[@]}"; do
  log_file="${LOG_DIR}/${step%.py}.log"
  echo "========================================================================"
  echo "Running ${step}"
  echo "Log: ${log_file}"
  echo "========================================================================"
  MPLCONFIGDIR=/tmp/mpl PYTHONUNBUFFERED=1 "${VENV_PY}" "${ROOT_DIR}/ml/classification/${step}" \
    | tee "${log_file}"
done

