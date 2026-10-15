# Canonical notebook reproduction checklist

## Before running

1. Use `notebooks/KOA_OrdinalConvNeXt_PCA_LinearSVC_Final.ipynb`. The nine notebooks under `notebooks/historical/` are exploratory archives, not the canonical end-to-end pipeline.
2. In Google Colab, connect the Google Drive containing `best_ordinal_convnext.pth` at `/content/drive/MyDrive/best_ordinal_convnext.pth`.
3. The notebook writes to the dedicated `KOA_OrdinalConvNeXt_PCA_SVM_v1_REPRO_TEST` folder to avoid overwriting older Drive artifacts. Change `OUTPUT_DIR` if you prefer a different destination.
4. Run every cell from top to bottom in a fresh Colab runtime. Do not use Test results to tune or select a model.

## Expected checkpoints

- Dataset split sizes: Train 5778, Validation 826, Test 1656. `auto_test` must not be loaded or evaluated.
- Class mapping: `{'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}`.
- Feature arrays: Train `(5778, 768)`, Validation `(826, 768)`.
- PCA: 59 components; explained variance approximately 0.950057.
- Validation-selected SVM: `C=0.1`, `class_weight='balanced'`; Validation metrics approximately Accuracy 0.7046, Macro F1 0.7198, Balanced Accuracy 0.7201, QWK 0.8485.
- Validation Macro ROC-AUC: direct baseline 0.9122, proposed hybrid 0.9078. Direct-model threshold-order violations in the AUC score construction: 92.
- Test classification metrics should reproduce `results/final_test_metrics.csv` (small numerical differences can occur across library/runtime versions).
- Test Macro ROC-AUC: direct baseline 0.9279, proposed hybrid 0.9237. Direct-model threshold-order violations in the AUC score construction: 217.
- AUC paired cluster bootstrap: 2000 valid replicates, 828 filename-derived clusters, hybrid-minus-direct Macro ROC-AUC difference -0.0042, 95% CI [-0.0080, -0.0007].
- Classification-metric paired cluster bootstrap: 2000 replicates and 828 filename-derived clusters; see `results/test_bootstrap_patient_level.csv`.

## If results differ

Stop and investigate the dataset version, checkpoint, preprocessing, class order, package versions, prediction ordering, and output paths. Do not adjust settings based on Test metrics. Do not overwrite manuscript results until the cause of a discrepancy is understood.

## Scope of validation

The user-provided notebook containing the current metric cells was executed in Google Colab on a Tesla T4 and produced the outputs summarized in `results/`. Packaging edits in this release add documentation, reorder the AUC cells into a Validation-before-Test workflow, and extend the final artifact checks. Those packaging edits have not been rerun end-to-end in a fresh Colab runtime. Static checks performed during packaging do not replace a fresh complete rerun.
