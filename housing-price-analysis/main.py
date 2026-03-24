"""
Main entry point for the Housing Price Analysis project.

Run this script to execute the complete data science pipeline:
  1. Data loading & feature engineering
  2. Exploratory Data Analysis (EDA) with saved plots
  3. Model training & evaluation
  4. Best-model diagnostics & persistence

Usage
-----
    python main.py
"""

import os
import sys

# Ensure src/ is importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd

import data_preprocessing as dp
import eda
import model as md

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Load & engineer data                                              #
    # ------------------------------------------------------------------ #
    print("\n📥  Loading California Housing dataset …")
    df = dp.load_data()
    df = dp.add_engineered_features(df)
    df = dp.remove_outliers(df)
    print(f"    Dataset shape after cleaning: {df.shape}")

    # ------------------------------------------------------------------ #
    # 2. EDA                                                               #
    # ------------------------------------------------------------------ #
    print("\n📊  Running Exploratory Data Analysis …")
    eda.print_summary_stats(df)
    paths = [
        eda.plot_target_distribution(df, OUTPUT_DIR),
        eda.plot_feature_distributions(df, OUTPUT_DIR),
        eda.plot_correlation_heatmap(df, OUTPUT_DIR),
        eda.plot_geo_scatter(df, OUTPUT_DIR),
        eda.plot_scatter_vs_target(df, OUTPUT_DIR),
    ]
    for p in paths:
        print(f"    Saved → {os.path.relpath(p)}")

    # ------------------------------------------------------------------ #
    # 3. Pre-process for modelling                                         #
    # ------------------------------------------------------------------ #
    print("\n⚙️   Preparing features …")
    X, y = dp.get_feature_target_split(df)
    X_train, X_test, y_train, y_test, scaler, feature_names = dp.split_and_scale(X, y)
    print(f"    Train samples: {len(y_train)}  |  Test samples: {len(y_test)}")
    print(f"    Features: {feature_names}")

    # ------------------------------------------------------------------ #
    # 4. Train & compare models                                            #
    # ------------------------------------------------------------------ #
    print("\n🤖  Training models …")
    results_df = md.compare_models(X_train, X_test, y_train, y_test, OUTPUT_DIR)

    comparison_path = md.plot_model_comparison(results_df, OUTPUT_DIR)
    print(f"    Saved → {os.path.relpath(comparison_path)}")

    # ------------------------------------------------------------------ #
    # 5. Best-model diagnostics                                            #
    # ------------------------------------------------------------------ #
    best_model_name = results_df["R²"].idxmax()
    print(f"\n🏆  Best model: {best_model_name} (R²={results_df.loc[best_model_name, 'R²']:.4f})")

    best_model = md.get_models()[best_model_name]
    best_model.fit(X_train, y_train)

    pred_path = md.plot_predictions_vs_actual(best_model, X_test, y_test, best_model_name, OUTPUT_DIR)
    print(f"    Saved → {os.path.relpath(pred_path)}")

    if hasattr(best_model, "feature_importances_"):
        imp_path = md.plot_feature_importance(best_model, feature_names, best_model_name, OUTPUT_DIR)
        print(f"    Saved → {os.path.relpath(imp_path)}")

    md.save_best_model(best_model, best_model_name, MODELS_DIR)

    print("\n✅  Pipeline complete!  Check the outputs/ directory for plots.\n")


if __name__ == "__main__":
    main()
