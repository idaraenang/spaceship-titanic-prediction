# Spaceship Titanic Prediction

A machine learning pipeline that predicts whether passengers aboard the Spaceship Titanic were transported to an alternate dimension using a Random Forest classifier.

## Overview

This project builds an end-to-end ML pipeline on the Kaggle Spaceship Titanic competition dataset, covering data preprocessing, feature encoding, scaling, model training, and Kaggle submission generation.

## Dataset

- **File:** `train.csv`
- **Rows:** 8,693
- **Target:** `Transported` (True = transported, False = not transported)
- **Class Distribution:** ~50/50 — well balanced

## Pipeline Steps

1. **Load & Explore** — null checks, class balance, `.describe()`
2. **Drop Irrelevant Columns** — `PassengerId`, `Name`, `Cabin`
3. **Imputation** — categorical → most frequent; numerical → median
4. **Encoding** — all categorical columns via `get_dummies`
5. **Scaling** — `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` with `StandardScaler`
6. **Train/Test Split** — 80/20, `random_state=42`
7. **Model Training** — `RandomForestClassifier` (100 estimators, max depth 5)
8. **Evaluation** — Accuracy score
9. **Kaggle Submission** — Generates `submission.csv` with predictions on test set

## Model

```python
RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
```

## Requirements

```
pandas
numpy
scikit-learn
```

## Usage

```bash
pip install -r requirements.txt
jupyter notebook Spacetitanic.ipynb
```
