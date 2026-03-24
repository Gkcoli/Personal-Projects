"""
Unit tests for the Housing Price Analysis project.
Run with:  python -m pytest tests/ -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_preprocessing as dp
import eda
import model as md


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def raw_df():
    return dp.load_data()


@pytest.fixture(scope="module")
def cleaned_df(raw_df):
    df = dp.add_engineered_features(raw_df)
    return dp.remove_outliers(df)


@pytest.fixture(scope="module")
def split_data(cleaned_df):
    X, y = dp.get_feature_target_split(cleaned_df)
    return dp.split_and_scale(X, y)


# ------------------------------------------------------------------ #
# data_preprocessing tests                                             #
# ------------------------------------------------------------------ #

class TestDataPreprocessing:
    def test_load_data_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_load_data_has_expected_columns(self, raw_df):
        expected = {"MedInc", "HouseAge", "AveRooms", "AveBedrms",
                    "Population", "AveOccup", "Latitude", "Longitude",
                    "MedianHouseValue"}
        assert expected.issubset(set(raw_df.columns))

    def test_load_data_no_missing_values(self, raw_df):
        assert raw_df.isnull().sum().sum() == 0

    def test_add_engineered_features_adds_columns(self, raw_df):
        df = dp.add_engineered_features(raw_df)
        for col in ("RoomsPerHousehold", "BedroomsPerRoom", "PopulationPerHousehold"):
            assert col in df.columns

    def test_add_engineered_features_no_negatives(self, raw_df):
        df = dp.add_engineered_features(raw_df)
        for col in ("RoomsPerHousehold", "BedroomsPerRoom", "PopulationPerHousehold"):
            assert (df[col] >= 0).all(), f"{col} has negative values"

    def test_remove_outliers_caps_at_5(self, raw_df):
        df = dp.add_engineered_features(raw_df)
        cleaned = dp.remove_outliers(df)
        assert cleaned["MedianHouseValue"].max() < 5.0

    def test_get_feature_target_split_shapes(self, cleaned_df):
        X, y = dp.get_feature_target_split(cleaned_df)
        assert len(X) == len(y)
        assert "MedianHouseValue" not in X.columns

    def test_split_and_scale_shapes(self, split_data):
        X_train, X_test, y_train, y_test, scaler, feature_names = split_data
        assert len(X_train) + len(X_test) > 0
        assert X_train.shape[1] == X_test.shape[1]
        assert len(feature_names) == X_train.shape[1]

    def test_scaled_data_near_zero_mean(self, split_data):
        X_train_scaled, _, _, _, _, _ = split_data
        col_means = X_train_scaled.mean(axis=0)
        assert np.allclose(col_means, 0, atol=0.01)


# ------------------------------------------------------------------ #
# EDA tests                                                            #
# ------------------------------------------------------------------ #

class TestEDA:
    def test_print_summary_stats_runs(self, cleaned_df, capsys):
        eda.print_summary_stats(cleaned_df)
        captured = capsys.readouterr()
        assert "Dataset shape" in captured.out

    def test_plot_target_distribution_creates_file(self, cleaned_df, tmp_path):
        path = eda.plot_target_distribution(cleaned_df, str(tmp_path))
        assert os.path.isfile(path)

    def test_plot_correlation_heatmap_creates_file(self, cleaned_df, tmp_path):
        path = eda.plot_correlation_heatmap(cleaned_df, str(tmp_path))
        assert os.path.isfile(path)

    def test_plot_geo_scatter_creates_file(self, cleaned_df, tmp_path):
        path = eda.plot_geo_scatter(cleaned_df, str(tmp_path))
        assert os.path.isfile(path)

    def test_plot_scatter_vs_target_creates_file(self, cleaned_df, tmp_path):
        path = eda.plot_scatter_vs_target(cleaned_df, str(tmp_path))
        assert os.path.isfile(path)

    def test_plot_feature_distributions_creates_file(self, cleaned_df, tmp_path):
        path = eda.plot_feature_distributions(cleaned_df, str(tmp_path))
        assert os.path.isfile(path)


# ------------------------------------------------------------------ #
# Model tests                                                          #
# ------------------------------------------------------------------ #

class TestModel:
    def test_get_models_returns_dict(self):
        models = md.get_models()
        assert isinstance(models, dict)
        assert len(models) >= 3

    def test_evaluate_model_metrics_keys(self, split_data):
        X_train, X_test, y_train, y_test, _, _ = split_data
        ridge = md.get_models()["Ridge Regression"]
        result = md.evaluate_model(ridge, X_train, X_test, y_train, y_test)
        for key in ("RMSE", "MAE", "R²", "predictions", "model"):
            assert key in result

    def test_evaluate_model_r2_reasonable(self, split_data):
        X_train, X_test, y_train, y_test, _, _ = split_data
        ridge = md.get_models()["Ridge Regression"]
        result = md.evaluate_model(ridge, X_train, X_test, y_train, y_test)
        assert result["R²"] > 0.5, "R² should be above 0.5 for a reasonable model"

    def test_evaluate_model_rmse_positive(self, split_data):
        X_train, X_test, y_train, y_test, _, _ = split_data
        ridge = md.get_models()["Ridge Regression"]
        result = md.evaluate_model(ridge, X_train, X_test, y_train, y_test)
        assert result["RMSE"] > 0

    def test_compare_models_returns_dataframe(self, split_data, tmp_path):
        X_train, X_test, y_train, y_test, _, _ = split_data
        df = md.compare_models(X_train, X_test, y_train, y_test, str(tmp_path))
        assert isinstance(df, pd.DataFrame)
        for col in ("RMSE", "MAE", "R²"):
            assert col in df.columns

    def test_plot_model_comparison_creates_file(self, split_data, tmp_path):
        X_train, X_test, y_train, y_test, _, _ = split_data
        results_df = md.compare_models(X_train, X_test, y_train, y_test, str(tmp_path))
        path = md.plot_model_comparison(results_df, str(tmp_path))
        assert os.path.isfile(path)

    def test_save_best_model_creates_file(self, split_data, tmp_path):
        X_train, _, y_train, _, _, _ = split_data
        rf = md.get_models()["Random Forest"]
        rf.fit(X_train, y_train)
        path = md.save_best_model(rf, "Random Forest", str(tmp_path))
        assert os.path.isfile(path)
