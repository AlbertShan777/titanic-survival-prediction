# Titanic Survival Prediction

A machine learning study by **Albert Shan**, developed on Kaggle and organized into a reproducible Python project. The study explores passenger characteristics, engineers family and name-title features, and compares logistic regression, random forests, and XGBoost.

**Best recorded holdout accuracy: 84.92%** with logistic regression and Title features. Five-fold cross-validation mean: **82.15% ± 2.26 percentage points** (standard deviation). These are verified saved notebook outputs; see [evaluation notes](reports/evaluation.md) for provenance and limitations.

## Approach

1. Explore missing values, age/fare distributions, and survival rates.
2. Engineer `FamilySize`, `IsAlone`, and normalized `Title` from passenger names.
3. Impute numerical features with medians and categorical features with the most frequent value; one-hot encode categories inside a scikit-learn pipeline.
4. Compare models on the same stratified 80/20 split (`random_state=42`).
5. Evaluate logistic regression with five folds, refit on all labeled rows, and export Kaggle predictions.

## Recorded results

| Model | Validation accuracy |
|---|---:|
| Logistic regression without Title | 80.45% |
| **Logistic regression with Title** | **84.92%** |
| Random forest (200 trees) | 81.01% |
| Tuned random forest | 81.56% |
| XGBoost | 80.45% |

Validation uses 179 passengers; training uses 712. This holdout was reused during model selection. Results are exploratory and are not Kaggle leaderboard scores.

## Repository layout

```text
notebooks/
  titanic_study.ipynb       # Ordered presentation and executable study
  original_kaggle.ipynb     # Original cell sources; outputs removed
src/
  titanic.py               # Feature engineering and model factories
  train.py                 # Evaluation and submission command
reports/
  evaluation.md            # Verified historical results and curation notes
requirements.txt
```

## Data

Obtain `train.csv` and `test.csv` from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/data). Sign in and complete any required competition access steps yourself. Place the files in a local `data/` directory. Data and generated submissions are excluded from Git.

Expected input: 891 labeled training rows and 418 test rows. `Survived` is the training target; `PassengerId` identifies submission rows. Keep the original CSV columns, including `Name`, which is needed for Title extraction.

## Run locally

Use Python 3.11 or 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m src.train --data-dir data --output-dir artifacts
```

The command prints holdout and cross-validation metrics and writes `artifacts/metrics.json` and `artifacts/submission.csv`. It does not submit to Kaggle. XGBoost on macOS may require the OpenMP runtime if its import reports a missing `libomp` library.

To explore the notebook:

```bash
jupyter lab notebooks/titanic_study.ipynb
```

Run all cells from a fresh kernel. The curated notebook expects the same `data/` folder; on Kaggle, adjust `DATA_DIR` to the attached Titanic dataset path and make `src/` importable. The archived notebook preserves the original Kaggle paths and interactive execution history.

## Reproducibility and scope

The original notebook used Python 3.12.13 and scikit-learn 1.6 documentation links. This project pins scikit-learn 1.6.1 as a reproducibility choice; the full original environment was not recorded. Other dependencies have bounded ranges. Compare new metrics with the historical values rather than assuming exact agreement. No scaling, additional features, or model settings were added to improve the reported scores.

See [evaluation and provenance](reports/evaluation.md) for known notebook ordering issues, metric evidence, and limitations.

## Verification

The command-line workflow was run locally on the existing Titanic CSV files. The 84.92% logistic regression holdout result reproduced exactly; tree model scores and one cross-validation fold differed from the saved Kaggle outputs. See [local metrics](reports/local_metrics.json) and [evaluation notes](reports/evaluation.md). Historical results above remain labeled as recorded results.
