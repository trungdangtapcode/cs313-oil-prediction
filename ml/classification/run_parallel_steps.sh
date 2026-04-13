#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLASSIFICATION_DIR="$ROOT_DIR/ml/classification"
RESULTS_DIR="$CLASSIFICATION_DIR/results"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
MAX_JOBS="${MAX_JOBS:-7}"
THREADS_PER_JOB="${THREADS_PER_JOB:-1}"
SEED="${SEED:-42}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$RESULTS_DIR/logs/$TIMESTAMP}"

declare -A STEP_FILES=(
  [1]="step1_train_baseline.py"
  [2]="step2_finetune_ensemble.py"
  [3]="step3_technical_improve.py"
  [4]="step4_select_and_train.py"
  [5]="step5_smart_selection.py"
  [6]="step6_weight_decay.py"
  [7]="step7_xgb_vs_gbm.py"
)

usage() {
  cat <<'EOF'
Usage:
  bash ml/classification/run_parallel_steps.sh
  bash ml/classification/run_parallel_steps.sh 1 4 6

Optional environment variables:
  MAX_JOBS=7            Number of scripts to run at the same time
  THREADS_PER_JOB=1     BLAS/OpenMP threads per Python process
  SEED=42               PYTHONHASHSEED exported to child processes
  PYTHON_BIN=...        Python interpreter to use
  LOG_DIR=...           Directory for per-step logs

Examples:
  MAX_JOBS=3 bash ml/classification/run_parallel_steps.sh
  MAX_JOBS=2 bash ml/classification/run_parallel_steps.sh 4 5 6
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python binary not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS="$THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$THREADS_PER_JOB"
export MKL_NUM_THREADS="$THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$THREADS_PER_JOB"
export VECLIB_MAXIMUM_THREADS="$THREADS_PER_JOB"
export PYTHONHASHSEED="$SEED"

selected_steps=()
if (( $# == 0 )); then
  selected_steps=(1 2 3 4 5 6 7)
else
  for arg in "$@"; do
    if [[ -z "${STEP_FILES[$arg]:-}" ]]; then
      echo "Unknown step: $arg" >&2
      usage >&2
      exit 1
    fi
    selected_steps+=("$arg")
  done
fi

declare -a pids=()
declare -A PID_TO_STEP=()
declare -A PID_TO_LOG=()

active_jobs() {
  jobs -rp | wc -l | tr -d ' '
}

wait_for_slot() {
  while (( $(active_jobs) >= MAX_JOBS )); do
    wait -n || true
  done
}

start_step() {
  local step="$1"
  local script_name="${STEP_FILES[$step]}"
  local step_label="step${step}"
  local log_file="$LOG_DIR/${step_label}.log"

  (
    cd "$ROOT_DIR" || exit 1
    echo "[$(date '+%F %T')] START ${step_label} (${script_name})"
    echo "[$(date '+%F %T')] PYTHON_BIN=$PYTHON_BIN"
    echo "[$(date '+%F %T')] MAX_JOBS=$MAX_JOBS THREADS_PER_JOB=$THREADS_PER_JOB SEED=$SEED"
    "$PYTHON_BIN" -u "$CLASSIFICATION_DIR/$script_name"
  ) >"$log_file" 2>&1 &

  local pid=$!
  pids+=("$pid")
  PID_TO_STEP["$pid"]="$step_label"
  PID_TO_LOG["$pid"]="$log_file"
  echo "[START] ${step_label} -> $log_file"
}

echo "ROOT_DIR=$ROOT_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "MAX_JOBS=$MAX_JOBS"
echo "THREADS_PER_JOB=$THREADS_PER_JOB"
echo "SEED=$SEED"
echo "STEPS=${selected_steps[*]}"

for step in "${selected_steps[@]}"; do
  wait_for_slot
  start_step "$step"
done

overall_status=0
summary_file="$LOG_DIR/summary.tsv"
{
  echo -e "step\tstatus\tlog"
  for pid in "${pids[@]}"; do
    step_label="${PID_TO_STEP[$pid]}"
    log_file="${PID_TO_LOG[$pid]}"
    if wait "$pid"; then
      echo "[ OK ] ${step_label} -> $log_file"
      echo -e "${step_label}\tOK\t${log_file}"
    else
      rc=$?
      echo "[FAIL:${rc}] ${step_label} -> $log_file" >&2
      echo -e "${step_label}\tFAIL(${rc})\t${log_file}"
      overall_status=1
    fi
  done
} | tee "$summary_file"

echo "Summary saved to: $summary_file"
exit "$overall_status"
