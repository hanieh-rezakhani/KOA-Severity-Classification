# Reported result artifacts

These small aggregate files are based on the user's completed Google Colab run. They are reference outputs, not outputs regenerated during repository packaging.

- `final_test_metrics.csv`: direct baseline and proposed hybrid Test classification metrics plus Macro ROC-AUC.
- `validation_roc_auc_per_class.csv` / `validation_roc_auc_summary.csv`: per-class and macro one-vs-rest ROC-AUC on Validation.
- `test_roc_auc_per_class.csv` / `test_roc_auc_summary.csv`: per-class and macro one-vs-rest ROC-AUC on Test.
- `test_roc_auc_patient_bootstrap_ci.csv`: 2,000-replicate paired filename-derived cluster bootstrap for the Test Macro ROC-AUC estimates and their difference.
- `test_bootstrap_patient_level.csv`: 2,000-replicate paired participant-cluster bootstrap for Accuracy, Macro F1, Balanced Accuracy, and QWK.
- `test_confusion_matrix_hybrid.csv` / `.png`: proposed Hybrid confusion matrix on Test.

## ROC-AUC scoring notes

- Macro ROC-AUC is the unweighted mean of one-vs-rest AUC values for KL0–KL4.
- The proposed hybrid uses `LinearSVC.decision_function` scores.
- For the direct ordinal baseline only, four threshold logits are converted to cumulative probabilities. Isotonic regression is applied to the samples with threshold-order violations before deriving class scores for AUC. This AUC-score post-processing does not change the saved hard-label predictions.
- Reported direct-model threshold-order violations: 92 Validation samples and 217 Test samples.
- ROC-AUC is complementary and was not used to tune or select the SVM configuration.

The cluster bootstrap intervals are conditional on the saved predictions and filename-derived grouping. Some historical exploratory experiments used Test labels for selection of other configurations; see `docs/NOTEBOOK_CLEANUP_REPORT.md` before interpreting the project history as a completely untouched benchmark. External validity is not established by these intervals.
