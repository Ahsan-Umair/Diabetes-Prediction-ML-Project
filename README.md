# Diabetes Prediction ML Project

A binary-classification notebook that predicts diabetes outcomes from common clinical measurements. It walks through data inspection, preprocessing, baseline training, model comparison, and evaluation with both threshold-based and ranking metrics.

> This repository is an educational demonstration, not medical advice or a clinically validated diagnostic system.

## Workflow

- Explore distributions, missing or implausible zero values, and feature relationships.
- Separate the `Outcome` target from the diagnostic measurements.
- Create reproducible training and test sets.
- Standardize features for logistic regression.
- Evaluate with accuracy, precision, recall, F1, confusion matrix, classification report, and ROC AUC.
- Compare logistic regression, a bounded-depth decision tree, and random forest.

## Notebook walkthrough

1. Read the clinical table, check missing values, and count zero entries in glucose, blood pressure, skin thickness, insulin, and BMI.
2. Treat zero in those five fields as missing and fill each field with its median. This imputation is performed on the full table before the train/test split, so the reported test scores are an exploratory estimate rather than a leakage-free validation.
3. Separate `Outcome`, make an 80/20 split with `random_state=42`, and fit `StandardScaler` on the training features. All three classifiers in the notebook use the scaled arrays.
4. Fit logistic regression and inspect the first class predictions and probabilities. Report accuracy, precision, recall, F1, confusion matrix, classification report, and ROC AUC.
5. Fit a decision tree limited to depth 5 and a 100-tree random forest, then compare their predictions with each other and with the logistic baseline.

The final display cell currently prints the decision-tree metrics a second time (`dt_pred`/`dt_prob`) even though it follows the random-forest fit. Use `rf_pred`/`rf_prob` to evaluate the forest separately. The `.ipynb_checkpoints` copy is an editor checkpoint, not a separate modeling workflow.

## Dataset

`diabetes_dataset.csv` contains 767 patient records with the following predictors: pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age. `Outcome` is the binary target.

## Repository contents

| File | Purpose |
| --- | --- |
| `Diabetes_Predictor.ipynb` | Complete exploratory and modeling workflow |
| `diabetes_dataset.csv` | Source dataset |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib seaborn scikit-learn
jupyter lab Diabetes_Predictor.ipynb
```

Run the notebook cells sequentially so that cleaning and preprocessing are applied before model evaluation.
