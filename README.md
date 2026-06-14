# Machine Learning-Based Prediction of Nusselt Number in Forced Convection

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange)

## Overview

This project investigates whether Machine Learning models can predict the **Nusselt number (Nu)**
more accurately than classical empirical correlations (Dittus–Boelter) in forced convection
inside a pipe.

### Core Question
> Can ML models outperform classical heat transfer correlations?

---

## Physics Background

For forced convection inside a pipe:

$$Nu = f(Re, Pr)$$

**Dittus–Boelter Correlation:**

$$Nu = 0.023 \cdot Re^{0.8} \cdot Pr^{0.4}$$

| Symbol | Parameter       | Range Used       |
|--------|----------------|------------------|
| $Re$   | Reynolds Number | 5,000 – 100,000  |
| $Pr$   | Prandtl Number  | 0.7 – 100        |
| $Nu$   | Nusselt Number  | Target Variable  |
 
---

## Models Compared

| Model             | Library              |
|-------------------|----------------------|
| Linear Regression | scikit-learn         |
| Random Forest     | scikit-learn         |
| XGBoost           | xgboost              |
| Neural Network    | scikit-learn (MLP)   |

---

## Results

| Model             | RMSE | MAE | R²   |
|-------------------|------|-----|------|
| Linear Regression | —    | —   | —    |
| Random Forest     | —    | —   | —    |
| XGBoost           | —    | —   | —    |
| Neural Network    | —    | —   | —    |

> Results populate automatically after running `main.py`

---

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/nusselt-ml.git
cd nusselt-ml
pip install -r requirements.txt
```
## Run

bash
python main.py

---

## Key Finding

Feature importance analysis reveals whether $Re$ or $Pr$ dominates
the prediction of $Nu$, providing physical insight alongside ML performance.

---
