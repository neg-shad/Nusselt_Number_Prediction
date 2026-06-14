"""
Data loading, validation, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def load_data(path: str = "data/nusselt_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {"Re", "Pr", "Nu"}.issubset(df.columns), "Missing required columns."
    print(f"[✓] Loaded dataset  |  Shape: {df.shape}")
    return df


def split_data(
    df: pd.DataFrame,
    target: str = "Nu",
    test_size: float = 0.2,
    seed: int = 42,
    scale: bool = True
) -> Tuple:
    """
    Split into train/test sets and optionally scale features.

    Returns:
        X_train, X_test, y_train, y_test, scaler (or None)
    """
    features = [col for col in df.columns if col != target]
    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    print(f"[✓] Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler
