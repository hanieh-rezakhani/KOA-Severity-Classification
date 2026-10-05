# KOA Severity Classification

Research repository for **5-class knee osteoarthritis (KOA) severity classification** from knee X-ray images.

> **Repository release:** `v0.8.0`  
> This repository contains the reusable code, curated notebooks, experiment configurations, tests, and documentation for the project. Historical research artifacts are preserved separately from the refactored execution layer.

## Important reproducibility note

**This repository does not claim that the historical notebook results have been independently reproduced by the refactored pipeline.** The original notebooks and their historical results are preserved for provenance. A run created through the current `scripts/` + `configs/` pipeline is considered a new run and should be reported separately.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the exact status and procedure.

## Repository contents

```text
KOA-Severity-Classification/
├── README.md
├── VERSION
├── pyproject.toml
├── requirements.txt
├── configs/                         # explicit experiment configurations
├── src/koa_severity/                # reusable research code
│   ├── config.py
│   ├── data.py
│   ├── datasets.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   ├── metrics.py
│   ├── experiments.py
│   └── reproducibility.py
├── scripts/                         # command-line entry points and validation
├── notebooks/                       # 9 curated research notebooks
├── tests/                           # automated tests
├── docs/
│   ├── REPRODUCIBILITY.md
│   ├── EXPERIMENT_MAP.md
│   └── EXPERIMENTS.md
├── results/                         # generated run metadata/results location
├── checkpoints/                     # checkpoint download instructions; weights hosted externally
└── archive/raw_notebooks/            # original notebooks preserved for provenance
```

## Setup

Python **3.10 or newer** is recommended.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

The project declares the same runtime dependencies in `pyproject.toml`; `requirements.txt` is also provided for straightforward environment installation.

## Dataset

The research dataset is **not included** in this repository. Set `KOA_DATASET_PATH` to the local dataset root.

### Windows PowerShell

```powershell
$env:KOA_DATASET_PATH = "C:\path\to\knee-osteoarthritis-dataset-with-severity"
```

### macOS/Linux

```bash
export KOA_DATASET_PATH="/path/to/knee-osteoarthritis-dataset-with-severity"
```

The expected directory is compatible with an ImageFolder-style layout, typically containing class/split directories such as `train/`, `val/`, and `test/`. The exact layout used by a historical notebook may differ; check that notebook before treating its result as a reproduction target.

## Validate the repository

Run the automated tests and repository validator before committing changes:

```bash
pytest
python scripts/validate_repo.py
```

The validator checks the required repository structure and all 9 curated notebooks for saved outputs, Colab-only paths/install commands, and Python syntax errors.

## Run an experiment metadata record

Experiment configurations live under `configs/`.

```bash
python scripts/run_experiment.py --config configs/resnet18_baseline.yaml
python scripts/summarize_runs.py
```

The run-management layer records the selected configuration, seed, and run metadata under `results/runs/`. Generated run files are ignored by Git by default.

For an inexpensive pipeline smoke test:

```bash
python scripts/run_experiment.py --config configs/resnet18_no_pretrained_smoke.yaml
```

The smoke configuration is intended to test the execution/provenance layer; it is **not** a scientific benchmark.

## Training and evaluation entry points

The reusable execution layer includes:

- `scripts/train.py` — model training entry point.
- `scripts/evaluate.py` — checkpoint evaluation entry point.
- `scripts/run_experiment.py` — creates a recorded experiment run from YAML configuration.
- `scripts/summarize_runs.py` — summarizes recorded runs.
- `scripts/validate_repo.py` — repository integrity checks.

The reusable package under `src/koa_severity/` contains dataset handling, models, training/evaluation helpers, metrics, checkpoint validation, configuration, experiment metadata, and reproducibility utilities.

## Notebook policy

The 9 notebooks under `notebooks/` are **curated copies** of the research workflows. They are cleaned for portability and should not silently download data or depend on Google Colab paths.

The original notebooks remain under `archive/raw_notebooks/` and are preserved as research provenance. They should not be treated as identical to the refactored pipeline.

## Reproducibility status

There are three distinct categories in this repository:

1. **Historical notebook results** — results produced in the original research notebooks and preserved for provenance.
2. **Curated notebook workflows** — cleaned/portable notebook versions; their presence does not mean their historical outputs were independently reproduced.
3. **Refactored pipeline runs** — new runs produced with the current `src/`, `scripts/`, and `configs/` code and recorded under `results/runs/`.

Only category 3 should be described as a run of the current refactored pipeline. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Research scope

The project includes work on:

- ResNet18-based KOA severity classification.
- Ordinal/structured prediction and related metrics.
- Interpretability workflows.
- ConvNeXt and deep-feature experiments.
- Classical ML experiments using extracted features.
- ROC-AUC and other evaluation/benchmarking analyses.
- An auxiliary KNEEL knee X-ray annotation/localization workflow.

See [`docs/EXPERIMENT_MAP.md`](docs/EXPERIMENT_MAP.md) for the relationship between notebooks and experiment areas.

## Model checkpoints

The trained model checkpoints are hosted separately because their combined size is too large for the GitHub repository.

**[Download the model checkpoints from Google Drive](https://drive.google.com/drive/folders/1FphXghZ4p0TEZl6UjrPUbVbpQjWQ6njW?usp=sharing)**

After downloading the required `.pth` files, place them in the local `checkpoints/` directory:

```text
KOA-Severity-Classification/
└── checkpoints/
    ├── best_ordinal_convnext.pth
    └── ...
```

The checkpoint files are intentionally excluded from Git via `.gitignore`. The external checkpoint folder should be treated as the source for the released model weights.

For each checkpoint used in a reported result, keep its filename, architecture, preprocessing, class mapping, training configuration, seed, and SHA-256 hash documented alongside the corresponding experiment.

## Data and generated artifacts

The research dataset, generated reports, caches, credentials, and other machine-local artifacts remain outside Git unless they are intentionally released. `.gitignore` excludes common large/model/local files.

## Version history

- **v0.1–v0.3:** initial organization, reusable utilities, tests, documentation, and portable curated notebooks.
- **v0.4:** reusable supervised training/evaluation pipeline.
- **v0.5:** explicit YAML experiment configurations and run metadata.
- **v0.8:** repository-wide version alignment, clearer reproducibility claims, dependency/package consistency, strengthened validation documentation, and a clean release structure.
