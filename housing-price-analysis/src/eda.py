"""
Exploratory Data Analysis (EDA) module for Housing Price Analysis.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# Use a non-interactive backend so plots can be saved without a display
import matplotlib
matplotlib.use("Agg")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def _ensure_output_dir(output_dir: str = OUTPUT_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_target_distribution(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> str:
    """Plot and save histogram + KDE of the target variable."""
    output_dir = _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["MedianHouseValue"], bins=50, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Distribution of Median House Value", fontsize=13)
    axes[0].set_xlabel("Median House Value ($100k)")
    axes[0].set_ylabel("Count")

    log_vals = np.log1p(df["MedianHouseValue"])
    axes[1].hist(log_vals, bins=50, color="#DD8452", edgecolor="white")
    axes[1].set_title("Log Distribution of Median House Value", fontsize=13)
    axes[1].set_xlabel("log(1 + Median House Value)")
    axes[1].set_ylabel("Count")

    fig.suptitle("Target Variable Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "01_target_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_distributions(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> str:
    """Plot histograms for all numeric features."""
    output_dir = _ensure_output_dir(output_dir)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(numeric_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=40, color="#55A868", edgecolor="white")
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "02_feature_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> str:
    """Plot and save a correlation heatmap."""
    output_dir = _ensure_output_dir(output_dir)
    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "03_correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_geo_scatter(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> str:
    """Scatter plot of house locations coloured by median house value."""
    output_dir = _ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter(
        df["Longitude"],
        df["Latitude"],
        c=df["MedianHouseValue"],
        cmap="viridis",
        alpha=0.4,
        s=5,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Median House Value ($100k)", fontsize=11)
    ax.set_title("California Housing – Geographic Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    path = os.path.join(output_dir, "04_geo_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scatter_vs_target(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> str:
    """Scatter plots of top correlated features vs target."""
    output_dir = _ensure_output_dir(output_dir)
    target = "MedianHouseValue"
    corr = df.corr(numeric_only=True)[target].drop(target).abs().sort_values(ascending=False)
    top_features = corr.head(4).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, feat in enumerate(top_features):
        axes[i].scatter(df[feat], df[target], alpha=0.2, s=5, color=colors[i])
        axes[i].set_xlabel(feat, fontsize=11)
        axes[i].set_ylabel(target, fontsize=11)
        axes[i].set_title(f"{feat} vs {target}", fontsize=12)

    fig.suptitle("Top Features vs Target", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "05_scatter_vs_target.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def print_summary_stats(df: pd.DataFrame) -> None:
    """Print descriptive statistics."""
    print("=" * 60)
    print("Dataset shape:", df.shape)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDescriptive statistics:")
    print(df.describe().round(3).to_string())
    print("=" * 60)
