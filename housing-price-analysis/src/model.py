"""
Model training and evaluation module for Housing Price Analysis.
"""

import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_models() -> dict:
    """Return a dictionary of {name: model_instance} to evaluate."""
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        ),
    }


def evaluate_model(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Fit a model and return evaluation metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "predictions": y_pred,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
    }


def compare_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str = OUTPUT_DIR,
) -> pd.DataFrame:
    """Train all models, print a comparison table, return results DataFrame."""
    _ensure_dirs(output_dir)
    models = get_models()
    results = []

    for name, model in models.items():
        print(f"  Training {name}...")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        results.append(
            {
                "Model": name,
                "RMSE": round(metrics["RMSE"], 4),
                "MAE": round(metrics["MAE"], 4),
                "R²": round(metrics["R²"], 4),
            }
        )
        print(f"    RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R²={metrics['R²']:.4f}")

    results_df = pd.DataFrame(results).set_index("Model")
    print("\nModel Comparison:")
    print(results_df.to_string())
    return results_df


def plot_model_comparison(results_df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> str:
    """Bar chart comparing RMSE, MAE and R² across models."""
    _ensure_dirs(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    metrics = ["RMSE", "MAE", "R²"]

    for ax, metric, color in zip(axes, metrics, colors):
        bars = ax.bar(results_df.index, results_df[metric], color=color, edgecolor="white")
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "06_model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_predictions_vs_actual(
    model,
    X_test: np.ndarray,
    y_test: pd.Series,
    model_name: str = "Best Model",
    output_dir: str = OUTPUT_DIR,
) -> str:
    """Scatter plot of predicted vs actual values for the best model."""
    _ensure_dirs(output_dir)
    y_pred = model.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Predicted vs Actual
    lim = max(y_test.max(), y_pred.max()) * 1.05
    axes[0].scatter(y_test, y_pred, alpha=0.3, s=5, color="#4C72B0")
    axes[0].plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect Prediction")
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    axes[0].set_xlabel("Actual Value ($100k)")
    axes[0].set_ylabel("Predicted Value ($100k)")
    axes[0].set_title(f"{model_name} – Predicted vs Actual", fontsize=12)
    axes[0].legend()

    # Residuals
    residuals = y_test.values - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=5, color="#DD8452")
    axes[1].axhline(0, color="r", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Predicted Value ($100k)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title(f"{model_name} – Residuals", fontsize=12)

    plt.tight_layout()
    path = os.path.join(output_dir, "07_predictions_vs_actual.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(
    model,
    feature_names: list[str],
    model_name: str = "Best Model",
    output_dir: str = OUTPUT_DIR,
) -> str:
    """Bar chart of feature importances (tree-based models only)."""
    _ensure_dirs(output_dir)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_imp = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_names[::-1], sorted_imp[::-1], color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importances – {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "08_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_best_model(
    model, model_name: str, models_dir: str = MODELS_DIR
) -> str:
    """Persist the best model to disk with joblib."""
    _ensure_dirs(models_dir)
    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(models_dir, f"{safe_name}.pkl")
    joblib.dump(model, path)
    print(f"Best model saved → {path}")
    return path
