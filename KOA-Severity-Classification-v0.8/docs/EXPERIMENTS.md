# Experiment management

The repository distinguishes **historical experiments** from **reproducible runs**.

- `archive/raw_notebooks/` preserves original research artifacts.
- `notebooks/` contains curated analysis notebooks.
- `configs/` contains explicit experiment settings.
- `results/runs/<name>/run.json` records the config used for a run.
- `checkpoints/` is for local model artifacts and is ignored by Git by default.

Example:

```bash
python scripts/run_experiment.py --config configs/resnet18_baseline.yaml
python scripts/summarize_runs.py
```

This metadata layer does not claim that historical notebook results were produced with the current training script. Historical results remain attributed to their source notebooks until independently reproduced.
