"""
main.py — Entry point for Nusselt Number ML Prediction Project

Usage:
    python main.py

Outputs:
    - results/predicted_vs_actual.png
    - results/feature_importance_*.png
    - results/model_comparison.png
    - results/results_table.csv
"""

import os
import pandas as pd


from src.dataset           import load_data, split_data
from src.models            import get_all_models
from src.evaluate          import train_and_evaluate, feature_importance
from src.visualize         import (
    plot_predicted_vs_actual,
    plot_feature_importance,
    plot_model_comparison,
    plot_nu_distribution
)

DATASET_PATH = "data/nusselt_dataset.csv"
FEATURE_NAMES = ["Re", "Pr"]


def main():
    print("=" * 55)
    print("  Nusselt Number Prediction — ML Study")
    print("=" * 55)

    # ── 1. Generate or load dataset ──────────────────────
    if not os.path.exists(DATASET_PATH):
        print("\n[Phase 1] Generating dataset ...")
        generate_dataset(n_samples=10_000, noise_std=0.0)
    else:
        print(f"\n[Phase 1] Dataset found at {DATASET_PATH}")

    df = load_data(DATASET_PATH)
    plot_nu_distribution(df["Nu"].values)

    # ── 2. Split ──────────────────────────────────────────
    print("\n[Phase 2] Splitting data (80/20) ...")
    X_train, X_test, y_train, y_test, scaler = split_data(df, scale=True)

    # ── 3. Train & Evaluate ───────────────────────────────
    print("\n[Phase 3] Training models ...")
    models = get_all_models(seed=42)
    results_df, predictions = train_and_evaluate(
        models, X_train, X_test, y_train, y_test
    )

    # ── 4. Results Table ──────────────────────────────────
    print("\n" + "=" * 45)
    print("  Final Results")
    print("=" * 45)
    print(results_df.to_string(index=False))
    results_df.to_csv("results/results_table.csv", index=False)
    print("\n[✓] Saved → results/results_table.csv")

    # ── 5. Visualize ──────────────────────────────────────
    print("\n[Phase 4] Generating plots ...")
    plot_predicted_vs_actual(y_test, predictions)
    plot_model_comparison(results_df)

    # ── 6. Feature Importance ─────────────────────────────
    print("\n[Phase 5] Feature Importance Analysis ...")
    for name, model in models.items():
        imp_df = feature_importance(model, FEATURE_NAMES)
        if imp_df is not None:
            print(f"\n  {name}:")
            print(imp_df.to_string(index=False))
            plot_feature_importance(imp_df, name)

    print("\n[✓] All done. Check the results/ folder.")


if __name__ == "__main__":
    main()
