"""
All plots: Predicted vs Actual, Feature Importance, Comparison Bar Chart.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
import os

SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":     150,
    "font.size":      11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def plot_predicted_vs_actual(
    y_test: np.ndarray,
    predictions: dict,
    save: bool = True
):
    """Scatter plot of Predicted vs Actual Nu for each model."""
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ax.scatter(y_test, y_pred, alpha=0.3, s=8, color="steelblue")
        lims = [min(y_test.min(), y_pred.min()),
                max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual $Nu$")
        ax.set_ylabel("Predicted $Nu$")
        ax.set_title(name)
        ax.legend(fontsize=9)

    fig.suptitle("Predicted vs Actual Nusselt Number", fontsize=13, y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(f"{SAVE_DIR}/predicted_vs_actual.png", bbox_inches="tight")
    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str, save: bool = True):
    """Bar chart of feature importances."""
    if importance_df is None:
        return

    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ["#2196F3", "#FF5722"]
    ax.barh(importance_df["Feature"], importance_df["Importance"],
            color=colors[:len(importance_df)])
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance — {model_name}")
    plt.tight_layout()
    if save:
        fig.savefig(f"{SAVE_DIR}/feature_importance_{model_name.replace(' ', '_')}.png",
                    bbox_inches="tight")
    plt.show()


def plot_model_comparison(results_df: pd.DataFrame, save: bool = True):
    """Bar chart comparing RMSE and R² across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    palette = sns.color_palette("muted", len(results_df))

    sns.barplot(data=results_df, x="Model", y="RMSE", palette=palette, ax=ax1)
    ax1.set_title("RMSE Comparison (lower is better)")
    ax1.tick_params(axis="x", rotation=15)

    sns.barplot(data=results_df, x="Model", y="R²", palette=palette, ax=ax2)
    ax2.set_title("R² Comparison (higher is better)")
    ax2.set_ylim([results_df["R²"].min() - 0.01, 1.0])
    ax2.tick_params(axis="x", rotation=15)

    fig.suptitle("Model Comparison — Nusselt Number Prediction", fontsize=13)
    plt.tight_layout()
    if save:
        fig.savefig(f"{SAVE_DIR}/model_comparison.png", bbox_inches="tight")
    plt.show()


def plot_nu_distribution(nu: np.ndarray, save: bool = True):
    """Distribution of Nu values in dataset."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(nu, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Nusselt Number ($Nu$)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of $Nu$ in Dataset")
    plt.tight_layout()
    if save:
        fig.savefig(f"{SAVE_DIR}/nu_distribution.png", bbox_inches="tight")
    plt.show()
