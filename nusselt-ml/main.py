"""
═══════════════════════════════════════════════════════════════════════════════
    Nusselt Number Prediction: ML vs Empirical Correlations
    Complete Pipeline with Comprehensive Analysis and Visualization
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import os
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Project imports
from src.dataset import load_data, split_data
from src.models import get_all_models
from src.evaluate import train_and_evaluate, feature_importance
from src.empirical_correlations import (
    get_empirical_predictions,
    evaluate_empirical_correlations
)
from src.compare_models import (
    evaluate_all_models,
    plot_error_comparison,
    plot_predictions_vs_actual_combined
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_PATH = "data/nusselt_dataset.csv"
RESULTS_DIR = "results"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature names (after log transformation)
FEATURE_NAMES = ["log10(Re)", "log10(Pr)"]

# Create results directory
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def print_header(text: str, char: str = "═"):
    """Print formatted section header."""
    width = 80
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")


def main():
    """
    Main execution pipeline:
    
    Phase 1: Data Loading & Exploration
    Phase 2: ML Model Training & Evaluation
    Phase 3: Empirical Correlations Evaluation
    Phase 4: Comprehensive Comparison
    Phase 5: Final Report & Recommendations
    """
    
    print_header("🚀 NUSSELT NUMBER PREDICTION PIPELINE", "═")
    print(f"📊 Dataset: {DATASET_PATH}")
    print(f"💾 Results: {RESULTS_DIR}/")
    print(f"🎲 Random State: {RANDOM_STATE}")
    print(f"✂️  Test Split: {TEST_SIZE * 100:.0f}%\n")
    
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1: DATA LOADING & EXPLORATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    print_header("[PHASE 1] DATA LOADING & EXPLORATION", "─")
    
    # Load raw data
    df_raw = load_data(DATASET_PATH)
    
    # Display basic statistics
    print("\n[Dataset Statistics - Original Scale]")
    print(df_raw.describe().T)
    
    
    
    # Split data with log transformation
    X_train, X_test, y_train, y_test, scaler = split_data(
        df_raw,
        target="Nu",
        test_size=TEST_SIZE,
        seed=RANDOM_STATE,
        scale=True,
        log_transform=True
    )
    
    print(f"\n[✓] Data preprocessing complete")
    print(f"    • Features: log10(Re), log10(Pr)")
    print(f"    • Target: log10(Nu)")
    print(f"    • Scaling: StandardScaler applied")
    
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: ML MODEL TRAINING & EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    print_header("[PHASE 2] ML MODEL TRAINING & EVALUATION", "─")
    
    # Get all ML models
    models = get_all_models(seed=RANDOM_STATE)
    print(f"[Models] {len(models)} ML models loaded:")
    for name in models.keys():
        print(f"    • {name}")
    
    # Train and evaluate (hold-out)
    print("\n[Training]")
    ml_results_df, ml_predictions = train_and_evaluate(
        models, X_train, X_test, y_train, y_test
    )
    
    print("\n[ML Model Results - Hold-out Test Set]")
    print(ml_results_df.to_string(index=False))
    
    # ── Cross-Validation ──────────────────────────────────────────────────────
    from src.evaluate import cross_validate_models, detect_overfitting

    cv_df = cross_validate_models(
        models,
        X_train, y_train,
        n_splits=5,
        seed=RANDOM_STATE,
    )

    overfit_df = detect_overfitting(cv_df, ml_results_df, threshold=0.05)

    # Save CV results
    cv_df.to_csv(f"{RESULTS_DIR}/cv_results.csv", index=False)
    overfit_df.to_csv(f"{RESULTS_DIR}/overfitting_check.csv", index=False)
    print(f"[✓] Saved: {RESULTS_DIR}/cv_results.csv")
    print(f"[✓] Saved: {RESULTS_DIR}/overfitting_check.csv")
    # ─────────────────────────────────────────────────────────────────────────

    # Save ML results
    ml_results_df.to_csv(f"{RESULTS_DIR}/ml_model_results.csv", index=False)
    print(f"[✓] Saved: {RESULTS_DIR}/ml_model_results.csv")


    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3: EMPIRICAL CORRELATIONS EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    print_header("[PHASE 3] EMPIRICAL CORRELATIONS EVALUATION", "─")
    
    # Load raw data for empirical correlations (no log transform needed)
    df_raw_reload = load_data(DATASET_PATH)
    
    # Split using same random state for fair comparison
    from sklearn.model_selection import train_test_split
    _, df_test_empirical = train_test_split(
        df_raw_reload,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    # Extract test features and target (original scale)
    Re_test = df_test_empirical['Re'].values
    Pr_test = df_test_empirical['Pr'].values
    Nu_test_true = df_test_empirical['Nu'].values
    
    # Get empirical predictions
    print("\n[Computing Empirical Correlations]")
    empirical_preds = get_empirical_predictions(Re_test, Pr_test, verbose=True)
    
    # Evaluate empirical correlations
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    empirical_results = []
    print(f"\n{'='*70}")
    print(f"{'Correlation':<25} {'R²':>12} {'RMSE':>12} {'MAE':>12}")
    print(f"{'='*70}")
    
    for name, Nu_pred in empirical_preds.items():
        valid_mask = np.isfinite(Nu_pred)
        
        if valid_mask.sum() > 0:
            r2 = r2_score(Nu_test_true[valid_mask], Nu_pred[valid_mask])
            rmse = np.sqrt(mean_squared_error(Nu_test_true[valid_mask], Nu_pred[valid_mask]))
            mae = mean_absolute_error(Nu_test_true[valid_mask], Nu_pred[valid_mask])
            
            empirical_results.append({
                'Model': name,
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae
            })
            
            print(f"{name:<25} {r2:>12.6f} {rmse:>12.4f} {mae:>12.4f}")
        else:
            print(f"{name:<25} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
    
    print(f"{'='*70}\n")
    
    empirical_results_df = pd.DataFrame(empirical_results)
    empirical_results_df = empirical_results_df.sort_values('R²', ascending=False)
    
    # Save empirical results
    empirical_results_df.to_csv(f"{RESULTS_DIR}/empirical_correlations_results.csv", index=False)
    print(f"[✓] Saved: {RESULTS_DIR}/empirical_correlations_results.csv")
    
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 4: COMPREHENSIVE COMPARISON (ML vs EMPIRICAL)
    # ═══════════════════════════════════════════════════════════════════════════
    
    print_header("[PHASE 4] COMPREHENSIVE COMPARISON: ML vs EMPIRICAL", "─")
    
    # Convert ML predictions back to original scale for fair comparison
    ml_predictions_original = {}
    for name, y_pred_log in ml_predictions.items():
        ml_predictions_original[name] = np.power(10, y_pred_log)
    
    # Combine all results
    combined_results = evaluate_all_models(
        Nu_test_true,
        ml_predictions_original,
        empirical_preds
    )
    
    print("\n[Combined Results: All Models]")
    print(combined_results.to_string(index=False))
    
    # Save combined results
    combined_results.to_csv(f"{RESULTS_DIR}/combined_results_all_models.csv", index=False)
    print(f"\n[✓] Saved: {RESULTS_DIR}/combined_results_all_models.csv")
    
    # Visualizations
    print("\n[Generating Comparison Plots]")
    plot_error_comparison(combined_results)
    plot_predictions_vs_actual_combined(
        Nu_test_true,
        ml_predictions_original,
        empirical_preds
    )
    
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 5: FINAL REPORT & RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    print_header("[PHASE 5] FINAL REPORT & RECOMMENDATIONS", "─")
    
    # Best ML model
    best_ml = combined_results[combined_results['Type'] == 'ML'].iloc[0]
    print(f"\n🏆 [Best ML Model]")
    print(f"   Model: {best_ml['Model']}")
    print(f"   R²:    {best_ml['R²']:.6f}")
    print(f"   RMSE:  {best_ml['RMSE']:.4f}")
    print(f"   MAE:   {best_ml['MAE']:.4f}")
    
    # Best Empirical correlation
    best_empirical = combined_results[combined_results['Type'] == 'Empirical'].iloc[0]
    print(f"\n⚙️  [Best Empirical Correlation]")
    print(f"   Model: {best_empirical['Model']}")
    print(f"   R²:    {best_empirical['R²']:.6f}")
    print(f"   RMSE:  {best_empirical['RMSE']:.4f}")
    print(f"   MAE:   {best_empirical['MAE']:.4f}")
    
    # Overall best
    overall_best = combined_results.iloc[0]
    print(f"\n🥇 [Overall Best Performer]")
    print(f"   Model: {overall_best['Model']}")
    print(f"   Type:  {overall_best['Type']}")
    print(f"   R²:    {overall_best['R²']:.6f}")
    print(f"   RMSE:  {overall_best['RMSE']:.4f}")
    print(f"   MAE:   {overall_best['MAE']:.4f}")
    
    # Performance improvement
    if overall_best['Type'] == 'ML':
        improvement = ((overall_best['R²'] - best_empirical['R²']) / best_empirical['R²']) * 100
        print(f"\n📈 [ML Improvement over Best Empirical]")
        print(f"   R² improvement: {improvement:+.2f}%")
        print(f"   RMSE reduction: {((best_empirical['RMSE'] - overall_best['RMSE']) / best_empirical['RMSE'] * 100):+.2f}%")
    
    # Summary statistics
    print(f"\n📊 [Summary Statistics]")
    print(f"   ML Models tested:         {len(combined_results[combined_results['Type'] == 'ML'])}")
    print(f"   Empirical Correlations:   {len(combined_results[combined_results['Type'] == 'Empirical'])}")
    print(f"   Test samples:             {len(Nu_test_true)}")
    print(f"   Feature space:            {X_train.shape[1]}D (log-transformed)")
    
    # Generate final report
    report_path = f"{RESULTS_DIR}/FINAL_REPORT.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("═" * 80 + "\n")
        f.write("  NUSSELT NUMBER PREDICTION: FINAL REPORT\n")
        f.write("═" * 80 + "\n\n")
        
        f.write("DATASET INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total samples:       {len(df_raw)}\n")
        f.write(f"Training samples:    {len(X_train)}\n")
        f.write(f"Test samples:        {len(X_test)}\n")
        f.write(f"Features:            {', '.join(FEATURE_NAMES)}\n")
        f.write(f"Random state:        {RANDOM_STATE}\n\n")
        
        f.write("COMBINED RESULTS (SORTED BY R²)\n")
        f.write("-" * 80 + "\n")
        f.write(combined_results.to_string(index=False))
        f.write("\n\n")
        
        f.write("BEST PERFORMERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Overall Best:              {overall_best['Model']} ({overall_best['Type']})\n")
        f.write(f"Best ML Model:             {best_ml['Model']}\n")
        f.write(f"Best Empirical:            {best_empirical['Model']}\n\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        if overall_best['Type'] == 'ML':
            f.write("✓ Machine Learning models outperform classical empirical correlations.\n")
            f.write(f"✓ Recommended model for deployment: {overall_best['Model']}\n")
            f.write("✓ Consider ensemble methods for further improvement.\n")
        else:
            f.write("✓ Empirical correlations remain competitive for this dataset.\n")
            f.write(f"✓ {overall_best['Model']} provides excellent accuracy with physical interpretability.\n")
        
        f.write("\n" + "═" * 80 + "\n")
    
    print(f"\n[✓] Final report saved: {report_path}")
    
    print_header("✅ PIPELINE COMPLETE", "═")
    print(f"📁 All results saved to: {RESULTS_DIR}/")
    print(f"🎯 Best model: {overall_best['Model']} (R² = {overall_best['R²']:.6f})\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ [ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

