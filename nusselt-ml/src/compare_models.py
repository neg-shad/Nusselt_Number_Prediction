"""
Compare all models (ML + Empirical) and visualize errors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def evaluate_all_models(y_true, ml_predictions, empirical_predictions):
    """
    Evaluate all models (ML + Empirical) and return combined results.
    
    Args:
        y_true: True target values
        ml_predictions: Dict of ML model predictions
        empirical_predictions: Dict of empirical correlation predictions
    
    Returns:
        pd.DataFrame: Results with columns [Model, Type, R², RMSE, MAE]
    """
    results = []
    
    # ML models
    for name, y_pred in ml_predictions.items():
        results.append({
            'Model': name,
            'Type': 'ML',
            'R²': r2_score(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred)
        })
    
    # Empirical correlations
    for name, y_pred in empirical_predictions.items():
        results.append({
            'Model': name,
            'Type': 'Empirical',
            'R²': r2_score(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred)
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('R²', ascending=False).reset_index(drop=True)
    
    return df


def plot_error_comparison(results_df, save_path='results/error_comparison.png'):
    """
    Create comprehensive error comparison plot.
    
    Args:
        results_df: DataFrame with columns [Model, Type, R², RMSE, MAE]
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    models = results_df['Model'].values
    types = results_df['Type'].values
    
    # Color mapping
    colors = ['#2ecc71' if t == 'ML' else '#3498db' for t in types]
    
    # Metrics
    metrics = ['R²', 'RMSE', 'MAE']
    titles = ['R² Score (↑ better)', 'RMSE (↓ better)', 'MAE (↓ better)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        values = results_df[metric].values
        
        bars = ax.barh(range(len(models)), values, color=colors, alpha=0.8)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.01 * val, i, f'{val:.4f}', 
                   va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=10)
        ax.set_xlabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.8, label='ML Models'),
        Patch(facecolor='#3498db', alpha=0.8, label='Empirical Correlations')
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Error comparison saved → {save_path}")


def plot_predictions_vs_actual_combined(y_true, ml_predictions, empirical_predictions, 
                                       save_path='results/predictions_comparison_all.png'):
    """
    Plot all predictions vs actual in a grid.
    
    Args:
        y_true: True values
        ml_predictions: Dict of ML predictions
        empirical_predictions: Dict of empirical predictions
        save_path: Save path
    """
    all_predictions = {**ml_predictions, **empirical_predictions}
    n_models = len(all_predictions)
    
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    
    for idx, (name, y_pred) in enumerate(all_predictions.items()):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.4, s=15, color='steelblue')
        
        # Perfect prediction line
        lims = [y_true.min(), y_true.max()]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect prediction')
        
        # Metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        ax.set_xlabel('True Nu', fontsize=11)
        ax.set_ylabel('Predicted Nu', fontsize=11)
        ax.set_title(f'{name}\n$R^2$={r2:.4f}, RMSE={rmse:.4f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[✓] Predictions comparison saved → {save_path}")
