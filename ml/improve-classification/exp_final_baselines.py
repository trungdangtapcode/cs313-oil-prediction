#!/usr/bin/env python3
"""Copy saved historical final baselines into this workspace."""

from __future__ import annotations

from config import RESULTS_DIR, ensure_dirs
from evaluation import load_final_baselines


def main() -> None:
    ensure_dirs()
    final_baselines = load_final_baselines()
    out = RESULTS_DIR / "final_baselines.csv"
    final_baselines.to_csv(out, index=False)
    print("[final_baselines] saved %s rows to %s" % (len(final_baselines), out))


if __name__ == "__main__":
    main()
