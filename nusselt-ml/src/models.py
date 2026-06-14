"""
All ML models used in the study.
Each function returns a fitted model.
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[!] XGBoost not installed. Skipping XGB model.")


def get_linear_regression() -> LinearRegression:
    return LinearRegression()


def get_random_forest(seed: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        n_jobs=-1,
        random_state=seed
    )


def get_xgboost(seed: int = 42):
    if not XGBOOST_AVAILABLE:
        return None
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=0
    )


def get_mlp(seed: int = 42) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed
    )


def get_all_models(seed: int = 42) -> dict:
    """Return all models as a dictionary."""
    models = {
        "Linear Regression": get_linear_regression(),
        "Random Forest":     get_random_forest(seed),
        "Neural Network":    get_mlp(seed),
    }
    xgb = get_xgboost(seed)
    if xgb is not None:
        models["XGBoost"] = xgb
    return models
