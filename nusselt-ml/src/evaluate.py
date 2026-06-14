"""
Model training, evaluation, and comparison table.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict


def train_and_evaluate(
    models: dict,
    X_train, X_test,
    y_train, y_test
) -> pd.DataFrame:
    """
    Train all models and compute RMSE, MAE, R².

    Returns:
        results_df : DataFrame with metrics per model
        predictions: dict of {model_name: y_pred}
    """
    results = []
    predictions = {}

    for name, model in models.items():
        print(f"  Training: {name} ...", end=" ")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "RMSE":  round(rmse, 4),
            "MAE":   round(mae,  4),
            "R²":    round(r2,   6)
        })
        predictions[name] = y_pred
        print(f"R²={r2:.4f}")

    results_df = pd.DataFrame(results).sort_values("R²", ascending=False)
    return results_df, predictions


def feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importance from tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return None
    imp = model.feature_importances_
    df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": imp
    }).sort_values("Importance", ascending=False)
    return df
