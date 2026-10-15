# Notebook cleanup and validation report

## What was changed

- Added one canonical, manuscript-aligned notebook for the current pipeline.

- Pinned the canonical notebook to Kaggle dataset version 1 to reduce the risk that a future dataset update changes the split contents.
- Added section headings and explanations that make the execution order explicit.
- Removed redundant checkpoint-search/debugging cells from the canonical notebook.
- Kept the current model architecture and core evaluation logic: ConvNeXt-Small ordinal checkpoint, fixed 384×384 preprocessing, 768-dimensional features, PCA fitted on Train only, LinearSVC hyperparameter selection on Validation, and final Test evaluation.
- Added patient-ID/laterality validation and a paired cluster-bootstrap section that saves the filename-derived participant ID alongside each Test prediction.
- Added artifact saving and metadata that records the historical evaluation caveat.
- Added nine historical notebooks with original source cells preserved, but cleared stored outputs/execution counts and added archival notes. This reduces size and prevents old outputs from appearing as current results.
- Added summary results from the validated run to `results/`.

## Static validation performed

The package notebook JSON and structure are checked during build. The main notebook is syntax-checked after converting IPython magics to Python-compatible form. Historical notebooks are archival snapshots and are not asserted to be fully executable end-to-end. No model was trained and no dataset/checkpoint was downloaded as part of packaging.

## Important scientific caveats

- The current notebook selects PCA/SVM settings using Validation before the final Test section. It does not fit PCA or LinearSVC on Test.
- The current project has a historical caveat: some exploratory experiments in older notebooks used Test labels to tune other configurations. Re-running the current notebook does not erase that history.
- Participant clusters are derived from the documented filename convention. The bootstrap confidence intervals are conditional on this grouping and on the saved predictions.
- The package does not include the dataset or model checkpoint.

## Not claimed

- The historical notebooks have not all been re-executed after cleanup.
- The package was not tested end-to-end on a fresh Google Colab runtime because the checkpoint is not included.
- The repository is not a clinically validated diagnostic tool.


## ROC-AUC update (v1.1.0)

- Integrated the user's completed Validation/Test Macro ROC-AUC cells into the canonical notebook and organized them so Validation AUC is reported before final Test evaluation.
- Added per-class and macro ROC-AUC CSV summaries and the paired cluster-bootstrap confidence interval for the Test AUC difference.
- Documented that ROC-AUC is a complementary metric and is not used to select the SVM; hyperparameters continue to be selected using Validation Macro F1 with QWK as a tie-breaker.
- Documented the direct ordinal baseline's isotonic-regression post-processing for AUC score construction: 92 Validation and 217 Test samples had threshold-order violations. This post-processing does not alter hard-label predictions.
- The user ran the underlying complete pipeline and shared its outputs. After that run, this packaging update reorganized headings/cells and added final-artifact checks; a new clean-runtime end-to-end run has not been performed after these documentation/manifest edits.
