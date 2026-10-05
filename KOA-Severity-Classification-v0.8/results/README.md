# Results policy

Generated metrics and checkpoints are intentionally not committed by default. Each reproducible run may create a directory under `results/runs/<run-name>/` containing `run.json`, metrics, logs, and references to checkpoints.

Use `scripts/summarize_runs.py` to build `results/experiment_summary.csv` from recorded run metadata.
