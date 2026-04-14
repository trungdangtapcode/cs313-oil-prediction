This folder preserves the validation-based classification workflow.

Scope:
- `step1` to `step7` here follow the older `train / validation / test` pattern.
- The top-level `ml/classification/` scripts are now the newer
  `train + inner-CV + test` versions.

Why this exists:
- to keep the earlier validation-based experiment path available
- to avoid losing the old selection logic while the main scripts evolve

Practical rule:
- use `with_val/` if you want the old outer-validation workflow
- use the top-level scripts if you want the newer direct-test workflow
