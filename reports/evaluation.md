# Evaluation and provenance

Source: Albert Shan's private Kaggle draft `albertshan/notebook629aebe781`, downloaded on October 27, 2026. The separately named `titanic-albert-study` draft contained only the Kaggle starter cell and an empty cell, so it was not used as the project implementation.

## Historical outputs verified in the source notebook

| Experiment | Accuracy | Source cell (zero-based) |
|---|---:|---:|
| Logistic regression, without Title | 0.8044692737430168 | 51 |
| Logistic regression, with Title | 0.8491620111731844 | 59 |
| Random forest, 200 trees | 0.8100558659217877 | 61 |
| Tuned random forest | 0.8156424581005587 | 64 |
| XGBoost | 0.8044692737430168 | 65 |

Holdout: 712 training rows and 179 validation rows; stratified 80/20 split, random_state=42. Title improves accuracy by 4.47 percentage points on this split. All tree experiments use Title. The boosting implementation is XGBClassifier, not sklearn's GradientBoostingClassifier.

Five-fold logistic regression accuracy (source cell 81): `[0.82122905, 0.80898876, 0.80898876, 0.80337079, 0.86516854]`; mean **0.8215491808423827**, population standard deviation **0.022577568861997445**. `cv=5` uses stratified folds without shuffling.

## Interpretation

These are saved historical outputs, not new execution results. The holdout was reused for feature and model selection, so it may be optimistic. Cross-validation is exploratory because the same dataset informed the project design. No Kaggle leaderboard score has been verified. A single split does not establish that logistic regression is generally superior.

## Curation changes

- Preserve every original cell's source in `notebooks/original_kaggle.ipynb`; remove all outputs, execution counts, and cell metadata to avoid publishing data previews or stale rich outputs.
- The original draft depends on interactive execution order: a model references `preprocessor` before its definition. Later saved outputs even mention an absent Deck feature. Treat the archive as history, not an executable workflow.
- Provide an ordered notebook and reusable functions with the same family/title transformations, imputers, one-hot encoding, classifier settings, split and CV configuration. Remove duplicate fits and unused manual preprocessing from the curated workflow.
- Give each model its own preprocessing instance. Keep the original unscaled logistic regression and max_iter=1000; convergence warnings should be reviewed rather than hidden.
- Exclude datasets, credentials, generated submissions and model artifacts. Package versions were not fully recorded in the original; exact numerical replication depends on the environment.

## Local verification on October 27, 2026

The command-line workflow completed using the existing local Titanic CSVs (891 training and 418 test passengers), Python 3.13.15, NumPy 2.5.4, pandas 2.3.3, scikit-learn 1.6.1, SciPy 1.18.1, and XGBoost 3.4.1. The local OpenMP library bundled with scikit-learn was supplied to XGBoost through DYLD_LIBRARY_PATH. New results are recorded separately in `local_metrics.json`.

| Experiment | Historical accuracy | Local rerun accuracy |
|---|---:|---:|
| Baseline logistic regression | 0.804469 | 0.804469 |
| Logistic regression with Title | 0.849162 | 0.849162 |
| Random forest | 0.810056 | 0.826816 |
| Tuned random forest | 0.815642 | 0.832402 |
| XGBoost | 0.804469 | 0.826816 |
| Logistic CV mean | 0.821549 | 0.820426 |

The primary holdout result reproduces exactly. Tree results and one CV fold differ. The original package versions and complete interactive state were not recorded, so the cause cannot be isolated; do not replace the historical results with these rerun values. The local SciPy version also emits an unknown `iprint` solver-option warning with scikit-learn 1.6.1; the run still completes. No hyperparameters were changed to match the historical values.

The generated submission has 418 rows, unique PassengerId values, and binary Survived predictions. Datasets and generated CSVs are not published.

All curated notebook code cells were also executed in order from a fresh namespace, using the local data path override and a noninteractive plotting backend. The submission schema checks passed.
