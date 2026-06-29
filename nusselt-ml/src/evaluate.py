# src/evaluate.py
"""
Model training, evaluation, and comparison table.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Optional, Tuple


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


def cross_validate_models(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scoring = {
        "r2":   "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae":  "neg_mean_absolute_error",
    }

    rows = []
    print(f"\n  [{n_splits}-Fold Cross-Validation on Training Set]\n")
    print(f"  {'Model':<22} {'R² mean':>9} {'R² std':>8} {'RMSE mean':>10} {'RMSE std':>9} {'MAE mean':>9}")
    print("  " + "─" * 72)

    for name, model in models.items():
        cv_result = cross_validate(
            clone(model), X_train, y_train,
            cv=kf,
            scoring=scoring,
            n_jobs=-1,
        )

        r2_mean   =  cv_result["test_r2"].mean()
        r2_std    =  cv_result["test_r2"].std()
        rmse_mean = -cv_result["test_rmse"].mean()
        rmse_std  =  cv_result["test_rmse"].std()
        mae_mean  = -cv_result["test_mae"].mean()

        # overfitting heuristic: واریانس بالا نسبت به میانگین RMSE
        high_var = (rmse_std > 0.15 * rmse_mean) if rmse_mean > 0 else False
        flag = "⚠ high variance" if high_var else "✓"

        print(
            f"  {name:<22} {r2_mean:>9.4f} {r2_std:>8.4f} "
            f"{rmse_mean:>10.4f} {rmse_std:>9.4f} {mae_mean:>9.4f}  {flag}"
        )

        rows.append({
            "Model":        name,
            "CV R² mean":   round(r2_mean,   4),
            "CV R² std":    round(r2_std,    4),
            "CV RMSE mean": round(rmse_mean, 4),
            "CV RMSE std":  round(rmse_std,  4),
            "CV MAE mean":  round(mae_mean,  4),
            "High Variance": high_var,
        })

    print()
    cv_df = pd.DataFrame(rows).sort_values("CV R² mean", ascending=False).reset_index(drop=True)
    return cv_df


def detect_overfitting(
    cv_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    threshold: float = 0.05,
) -> pd.DataFrame:
    
    merged = cv_df[["Model", "CV R² mean", "CV R² std"]].merge(
        holdout_df[["Model", "R²"]].rename(columns={"R²": "Hold-out R²"}),
        on="Model",
        how="inner",
    )
    merged["Gap (CV−HO)"] = (merged["CV R² mean"] - merged["Hold-out R²"]).round(4)
    merged["Overfit?"]    = merged["Gap (CV−HO)"] > threshold

    print(f"\n  [Overfitting Check]  threshold = {threshold}\n")
    print(f"  {'Model':<22} {'CV R²':>8} {'Hold-out R²':>12} {'Gap':>7}  Flag")
    print("  " + "─" * 60)
    for _, row in merged.iterrows():
        flag = "⚠ OVERFIT" if row["Overfit?"] else "✓ OK"
        print(
            f"  {row['Model']:<22} {row['CV R² mean']:>8.4f} "
            f"{row['Hold-out R²']:>12.4f} {row['Gap (CV−HO)']:>7.4f}  {flag}"
        )
    print()
    return merged


def feature_importance(model, feature_names: list) -> Optional[pd.DataFrame]:
    """Extract feature importance from tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return None
    imp = model.feature_importances_
    df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": imp
    }).sort_values("Importance", ascending=False)
    return df

