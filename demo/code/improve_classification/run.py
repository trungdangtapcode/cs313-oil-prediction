#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from config import EXPERIMENTS, LOG_DIR, THIS_DIR, ensure_dirs


STEP_SCRIPTS = {
    "final_baselines": "exp_final_baselines.py",
    "baseline": "exp_baseline.py",
    "feature_selection": "exp_feature_selection.py",
    "weight_decay": "exp_weight_decay.py",
    "ensemble": "exp_ensemble.py",
    "deep_learning": "exp_deep_learning.py",
    "report": "report.py",
}

TRAINING_STEPS = [step for step in EXPERIMENTS if step != "report"]


def parse_steps(raw: str):
    if raw.strip().lower() == "all":
        return list(EXPERIMENTS)
    return [step.strip() for step in raw.split(",") if step.strip()]


def clean_generated_results() -> None:
    """Remove stale result tables so a run cannot mix old and new artifacts."""
    ensure_dirs()
    results_dir = THIS_DIR / "results"
    removed = 0
    for path in results_dir.iterdir():
        if path.is_file():
            path.unlink()
            removed += 1
    if removed:
        print("[clean] removed %s old result files from %s" % (removed, results_dir))


def run_step(step: str) -> None:
    if step not in STEP_SCRIPTS:
        raise SystemExit("Unknown step %r. Available: %s" % (step, ", ".join(STEP_SCRIPTS)))
    ensure_dirs()
    script = THIS_DIR / STEP_SCRIPTS[step]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / ("%s_%s.log" % (step, timestamp))
    print("\n%s\nRUN STEP: %s\nscript: %s\nlog: %s\n%s" % ("=" * 90, step, script, log_path, "=" * 90))
    with log_path.open("w", encoding="utf-8") as log_file:
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(THIS_DIR / ".mplconfig"))
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=str(THIS_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        code = proc.wait()
    if code != 0:
        raise SystemExit("Step %s failed with exit code %s. See %s" % (step, code, log_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run improve-classification experiments.")
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma-separated steps or 'all'. Available: %s" % ", ".join(STEP_SCRIPTS),
    )
    parser.add_argument("--skip-dl", action="store_true", help="Skip deep_learning even when running all.")
    parser.add_argument("--no-clean", action="store_true", help="Do not remove old result files before training steps.")
    args = parser.parse_args()

    steps = parse_steps(args.steps)
    if args.skip_dl:
        steps = [step for step in steps if step != "deep_learning"]
    if not args.no_clean and any(step in TRAINING_STEPS for step in steps):
        clean_generated_results()
    for step in steps:
        run_step(step)
    print("\nDone. Final report: %s" % (THIS_DIR / "REPORT.md"))


if __name__ == "__main__":
    main()
