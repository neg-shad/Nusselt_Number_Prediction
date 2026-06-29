"""
Empirical Nusselt correlations for turbulent pipe flow.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Tuple


class EmpiricalCorrelations:
    """Collection of classical Nusselt correlations."""
    
    @staticmethod
    def dittus_boelter(Re: np.ndarray, Pr: np.ndarray) -> np.ndarray:
        """
        Dittus-Boelter correlation (1930).
        Valid: Re ≥ 10,000, 0.7 ≤ Pr ≤ 160
        
        Nu = 0.023 * Re^0.8 * Pr^0.4
        """
        return 0.023 * np.power(Re, 0.8) * np.power(Pr, 0.4)
    
    @staticmethod
    def sieder_tate(Re: np.ndarray, Pr: np.ndarray, mu_ratio: float = 1.0) -> np.ndarray:
        """
        Sieder-Tate correlation (1936).
        Valid: Re ≥ 10,000, 0.7 ≤ Pr ≤ 16,700
        
        Nu = 0.027 * Re^0.8 * Pr^(1/3) * (μ/μ_w)^0.14
        
        Note: Simplified with μ/μ_w = 1.0 (isothermal assumption)
        """
        return 0.027 * np.power(Re, 0.8) * np.power(Pr, 1/3) * mu_ratio
    
    @staticmethod
    def gnielinski(Re: np.ndarray, Pr: np.ndarray) -> np.ndarray:
        """
        Gnielinski correlation (1976).
        Valid: 3,000 ≤ Re ≤ 5×10⁶, 0.5 ≤ Pr ≤ 2,000
        
        f = (0.79*ln(Re) - 1.64)^(-2)
        Nu = (f/8)*(Re - 1000)*Pr / (1 + 12.7*(f/8)^0.5*(Pr^(2/3) - 1))
        """
        # Darcy friction factor (smooth tube)
        f = np.power(0.79 * np.log(Re) - 1.64, -2)
        
        numerator = (f / 8) * (Re - 1000) * Pr
        denominator = 1 + 12.7 * np.sqrt(f / 8) * (np.power(Pr, 2/3) - 1)
        
        return numerator / denominator
    
    @staticmethod
    def petukhov(Re: np.ndarray, Pr: np.ndarray) -> np.ndarray:
        """
        Petukhov correlation (1970).
        Valid: 10⁴ ≤ Re ≤ 5. 10⁶, 0.5 ≤ Pr ≤ 200
        
        f = (0.79*ln(Re) - 1.64)^(-2)
        Nu = (f/8)*Re*Pr / (1.07 + 12.7*(f/8)^0.5*(Pr^(2/3) - 1))
        """
        f = np.power(0.79 * np.log(Re) - 1.64, -2)
        
        numerator = (f / 8) * Re * Pr
        denominator = 1.07 + 12.7 * np.sqrt(f / 8) * (np.power(Pr, 2/3) - 1)
        
        return numerator / denominator


def load_data(path: str = "data/nusselt_dataset.csv") -> pd.DataFrame:
    """
    Load raw dataset for empirical correlation evaluation.
    
    Args:
        path: Path to CSV file containing Re, Pr, Nu columns
        
    Returns:
        DataFrame with original (non-transformed) values
    """
    df = pd.read_csv(path)
    assert {"Re", "Pr", "Nu"}.issubset(df.columns), "Missing required columns."
    print(f"[✓] Loaded dataset  |  Shape: {df.shape}")
    return df


def check_validity_range(Re: np.ndarray, Pr: np.ndarray) -> Dict[str, Dict[str, int]]:
    """
    Check how many samples fall within validity ranges of each correlation.
    
    Returns:
        Dictionary with correlation names and their valid/invalid counts
    """
    validity_ranges = {
        'Dittus-Boelter': {
            'Re_min': 10_000, 'Re_max': np.inf,
            'Pr_min': 0.7, 'Pr_max': 160
        },
        'Sieder-Tate': {
            'Re_min': 10_000, 'Re_max': np.inf,
            'Pr_min': 0.7, 'Pr_max': 16_700
        },
        'Gnielinski': {
            'Re_min': 3_000, 'Re_max': 5e6,
            'Pr_min': 0.5, 'Pr_max': 2_000
        },
        'Petukhov': {
            'Re_min': 10_000, 'Re_max': 5e6,
            'Pr_min': 0.5, 'Pr_max': 200
        }
    }
    
    results = {}
    for name, ranges in validity_ranges.items():
        valid_mask = (
            (Re >= ranges['Re_min']) & (Re <= ranges['Re_max']) &
            (Pr >= ranges['Pr_min']) & (Pr <= ranges['Pr_max'])
        )
        results[name] = {
            'valid': int(valid_mask.sum()),
            'invalid': int((~valid_mask).sum()),
            'percentage': float(valid_mask.mean() * 100)
        }
    
    return results


def get_empirical_predictions(
    Re: np.ndarray, 
    Pr: np.ndarray,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute predictions from all empirical correlations.
    
    Args:
        Re: Reynolds numbers (original scale, not log-transformed)
        Pr: Prandtl numbers (original scale, not log-transformed)
        verbose: Print diagnostic information
        
    Returns:
        Dictionary mapping correlation names to Nu predictions
    """
    if verbose:
        print(f"\n[Data Range Check]")
        print(f"  Re: [{Re.min():.1f}, {Re.max():.1f}]")
        print(f"  Pr: [{Pr.min():.3f}, {Pr.max():.3f}]")
        
        # Check validity ranges
        validity = check_validity_range(Re, Pr)
        print(f"\n[Validity Range Check]")
        for name, stats in validity.items():
            print(f"  {name:18s}: {stats['valid']:5d} valid ({stats['percentage']:5.1f}%)")
    
    predictions = {
        'Dittus-Boelter': EmpiricalCorrelations.dittus_boelter(Re, Pr),
        'Sieder-Tate': EmpiricalCorrelations.sieder_tate(Re, Pr),
        'Gnielinski': EmpiricalCorrelations.gnielinski(Re, Pr),
        'Petukhov': EmpiricalCorrelations.petukhov(Re, Pr)
    }
    
    # Sanity check predictions
    if verbose:
        print(f"\n[Prediction Range Check]")
        for name, pred in predictions.items():
            if np.all(np.isfinite(pred)):
                print(f"  {name:18s}: [{pred.min():.2f}, {pred.max():.2f}]  ✓")
            else:
                n_invalid = (~np.isfinite(pred)).sum()
                print(f"  {name:18s}: {n_invalid} invalid values (NaN/Inf)  ✗")
    
    return predictions


def evaluate_empirical_correlations(
    data_path: str = "data/nusselt_dataset.csv",
    test_fraction: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Evaluate all empirical correlations on test data.
    
    Args:
        data_path: Path to dataset CSV
        test_fraction: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (metrics_dict, test_dataframe_with_predictions)
    """
    # Load raw data
    df = load_data(data_path)
    
    # Train-test split (same random state as ML pipeline for fair comparison)
    from sklearn.model_selection import train_test_split
    _, df_test = train_test_split(
        df, 
        test_size=test_fraction, 
        random_state=random_state
    )
    
    print(f"\n[Test Set] Shape: {df_test.shape}")
    
    # Extract test features and target
    Re_test = df_test['Re'].values
    Pr_test = df_test['Pr'].values
    Nu_true = df_test['Nu'].values
    
    # Get predictions
    predictions = get_empirical_predictions(Re_test, Pr_test, verbose=True)
    
    # Compute metrics
    metrics = {}
    print(f"\n{'='*60}")
    print(f"{'Correlation':<20} {'R²':>10} {'RMSE':>10} {'MAE':>10}")
    print(f"{'='*60}")
    
    for name, Nu_pred in predictions.items():
        # Handle potential NaN/Inf values
        valid_mask = np.isfinite(Nu_pred)
        
        if valid_mask.sum() > 0:
            r2 = r2_score(Nu_true[valid_mask], Nu_pred[valid_mask])
            rmse = np.sqrt(mean_squared_error(Nu_true[valid_mask], Nu_pred[valid_mask]))
            mae = mean_absolute_error(Nu_true[valid_mask], Nu_pred[valid_mask])
            
            metrics[name] = {
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae,
                'valid_samples': int(valid_mask.sum())
            }
            
            print(f"{name:<20} {r2:>10.4f} {rmse:>10.4f} {mae:>10.4f}")
        else:
            metrics[name] = {
                'R²': np.nan,
                'RMSE': np.nan,
                'MAE': np.nan,
                'valid_samples': 0
            }
            print(f"{name:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
    
    print(f"{'='*60}\n")
    
    # Create results dataframe
    results_df = df_test.copy()
    for name, pred in predictions.items():
        results_df[f'{name}_pred'] = pred
    
    return metrics, results_df


if __name__ == "__main__":
    """Standalone testing."""
    metrics, results = evaluate_empirical_correlations()
    
    print("\n[Summary] Best performing correlation:")
    best_model = max(metrics.items(), key=lambda x: x[1]['R²'] if not np.isnan(x[1]['R²']) else -np.inf)
    print(f"  {best_model[0]}: R² = {best_model[1]['R²']:.4f}")

