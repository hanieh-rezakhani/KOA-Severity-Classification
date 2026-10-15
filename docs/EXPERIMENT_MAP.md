# Experiment map

## Canonical pipeline

- `notebooks/KOA_OrdinalConvNeXt_PCA_LinearSVC_Final.ipynb`
  - Current manuscript-aligned pipeline: ordinal ConvNeXt-Small checkpoint, 768-dimensional feature extraction, Train-only PCA, Validation-based LinearSVC search, final Test evaluation, and paired filename-derived participant-cluster bootstrap.

## Historical exploratory notebooks

| File | Topic | Status |
|---|---|---|
| `01_dataset_and_resnet_history.ipynb` | Dataset inspection, patient-ID split audit, ResNet18 experiments | Historical; multiple exploratory stages |
| `02_resnet18_hybrid_and_reproducibility.ipynb` | ResNet18, CLAHE, class weights, sampling, seed experiments | Historical; checkpoint/configuration dependent |
| `03_resnet18_ordinal_and_interpretability.ipynb` | ROI, CORAL/CORN, hybrid CNN, interpretability | Historical; multiple configurations |
| `04_convnext_pca_svm_and_ensembles.ipynb` | ConvNeXt, PCA/SVM, boosting and ensembles | Historical; some Test-label tuning was recorded |
| `05_deep_features_svm_and_efficientnet.ipynb` | Deep features, SVM/PCA/LDA and EfficientNet-B0 | Historical; cross-validation does not independently assess the already-trained feature extractor |
| `06_historical_evaluation_metrics.ipynb` | Multiple evaluation/ROC-AUC attempts | Historical; contains failed and inconsistent attempts |
| `07_roc_auc_historical_benchmarks.ipynb` | Historical ROC-AUC experiments | Historical; not a canonical metric source |
| `08_kneel_landmark_localization_experiment.ipynb` | KNEEL landmark localization attempt | Exploratory; no output used in the final classifier |
| `09_convnext_small_training_experiments.ipynb` | ConvNeXt-Small focal and ordinal training runs | Historical; checkpoint metadata needs care |

Saved outputs were removed from the historical notebooks to avoid displaying stale outputs as if they came from the current code. Scientific source cells were not rewritten during archival cleanup. These notebooks are retained for provenance and should not be assumed to run top-to-bottom without recovering their original data, checkpoints, dependencies, and intermediate files.
