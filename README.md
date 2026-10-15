# Breast Cancer Tumor Classification

Binary classifier predicting whether a breast tumor is malignant or benign from physical measurements, built for CS-437 (Machine Learning). **F1 score of 0.9508** on held-out test data.

The assignment deliberately left key decisions unspecified, including whether to treat the problem as classification or regression and which scoring metric to use. Those choices, and the reasoning behind them, are the substance of this project.

## The dataset

The [Kaggle breast cancer dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset), derived from the Wisconsin diagnostic data. Each row is one tumor described by physical measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry and fractal dimension, each reported as a mean, standard error, and worst-case value.

569 samples, split 70/30 into 398 training and 171 test rows.

## Why classification

The target variable, `diagnosis`, is categorical (M for malignant, B for benign), which makes this a classification problem rather than regression. The labels were mapped to 1 and 0 so they could be used numerically.

The features are physical characteristics of the tumor, and size, shape and texture correlate directly with diagnosis, so the measurement columns are a reasonable basis for prediction.

## Approach

A scikit-learn `Pipeline` with two stages:

**1. Scaling.** `MinMaxScaler` normalizes every feature to a common range. This matters because the raw measurements span very different scales, with area in the hundreds or thousands while smoothness sits below 1. Without scaling, logistic regression's coefficients are distorted by the units rather than the signal.

**2. Classification.** `LogisticRegression` with `max_iter=1000`. For a binary target with roughly linearly separable features after scaling, logistic regression is a well-matched choice: it is interpretable, fast, and outputs calibrated probabilities rather than bare labels.

Wrapping both steps in a `Pipeline` keeps the scaler fitted on training data only, which prevents test-set information leaking into preprocessing.

## Why F1 score

This was the most consequential decision in the project, since the assignment did not specify a metric.

**Accuracy would have been the wrong choice.** In a medical diagnostic setting, the two kinds of error carry very different costs, and accuracy treats them as equivalent. A false negative means telling a patient their malignant tumor is benign. A false positive means subjecting someone to unnecessary alarm and further testing. Both are harmful, and an accuracy score can look strong while quietly failing on one of them.

**F1 is the harmonic mean of precision and recall**, so it only scores well when the model is both catching malignant tumors (recall) and not over-flagging benign ones (precision). A model that hedges in either direction gets penalized. That balance is what the problem actually calls for, which is why F1 was the right fit over accuracy or over optimizing recall alone.

## Results

| Metric | Score |
|---|---|
| F1 score (test set) | **0.9508** |

An F1 of 0.95 means the model achieved high precision and high recall simultaneously rather than trading one for the other, which is the behavior the metric was chosen to measure.

## Tech stack

- **Python**
- **scikit-learn** — `Pipeline`, `MinMaxScaler`, `LogisticRegression`, `train_test_split`, `f1_score`
- **pandas / NumPy** — data loading, cleaning, and label encoding
- **Jupyter Notebook**

## Repository contents

| File | Description |
|---|---|
| `MidtermProject.ipynb` | Full analysis: data loading, preprocessing, model training, evaluation, and written reflection |
| `breast-cancer.csv` | The dataset |

## Running it

Install the dependencies and open the notebook:

```bash
pip install pandas numpy scikit-learn jupyter
jupyter notebook
```

The notebook expects `breast-cancer.csv` in the same directory.
