# Experiment Map

This document maps the research material preserved in the project. It is a documentation layer; it does not claim that every historical experiment is part of one final model.

## Core CNN line

### ResNet18
- baseline classification
- augmentation variants
- class weighting / weighted sampling
- preprocessing and resolution studies
- reproducibility / seed-controlled runs
- hybrid and ROI-related experiments

Sources:
- `notebooks/01_dataset_and_resnet_history.ipynb`
- `notebooks/02_resnet18_hybrid_and_reproducibility.ipynb`
- `notebooks/03_resnet18_ordinal_and_interpretability.ipynb`

## Ordinal / structured prediction

The advanced ResNet18 notebook contains experiments around ordinal formulations including CORAL/CORN-related work and hierarchical/cascade approaches.

Source:
- `notebooks/03_resnet18_ordinal_and_interpretability.ipynb`

## Interpretability

The advanced workflow includes Grad-CAM and feature-space analyses.

Source:
- `notebooks/03_resnet18_ordinal_and_interpretability.ipynb`

## Feature-based classical ML

Deep feature experiments include:
- RBF SVM
- PCA + SVM
- LDA + SVM
- SVM tuning
- EfficientNet-B0 comparisons

Source:
- `notebooks/05_deep_features_svm_and_efficientnet.ipynb`

## ConvNeXt branch

ConvNeXt experiments include feature extraction, PCA, classical classifiers, ensemble/hierarchical approaches, fine-tuning and checkpoint/reporting utilities.

Sources:
- `notebooks/04_convnext_pca_svm_and_ensembles.ipynb`
- `notebooks/09_convnext_checkpoint_and_reporting.ipynb`

## Evaluation / benchmarking

- `notebooks/06_final_evaluation_metrics.ipynb`
- `notebooks/07_roc_auc_benchmarks.ipynb`

These contain evaluation and historical benchmarking workflows.

## Auxiliary KNEEL pipeline

`notebooks/08_kneel_annotation_pipeline.ipynb` is kept separate because it addresses annotation/localization rather than the core severity classifier.

## Refactoring policy

The reusable execution path should gradually move:
- data/path handling → `src/koa_severity/data.py`
- reproducibility → `src/koa_severity/reproducibility.py`
- evaluation → `src/koa_severity/metrics.py`
- checkpoint/model loading → `src/koa_severity/models.py`

Historical experiment-specific code remains in notebooks until it is independently verified after refactoring.
