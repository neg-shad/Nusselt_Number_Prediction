"""
Data loading, validation, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
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
    scale: bool = True,
    log_transform: bool = False
) -> Tuple:
    

    df_processed = df.copy()

    #  Log transformation
    if log_transform:
        cols_to_log = ['Re', 'Pr', target]

        for col in cols_to_log:
            if col in df_processed.columns:
                if (df_processed[col] <= 0).any():
                    raise ValueError(f"Column {col} contains non-positive values; log10 undefined.")
                df_processed[col] = np.log10(df_processed[col])

    #  Split features and target
    X = df_processed.drop(columns=[target])
    y = df_processed[target]

    #  Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    #  Scaling (only on X)
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


