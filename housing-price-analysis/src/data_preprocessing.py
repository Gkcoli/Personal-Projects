"""
Data preprocessing module for Housing Price Analysis.
"""

import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "housing.csv")


def load_data(data_path: str = _DATA_PATH) -> pd.DataFrame:
    """Load the bundled housing CSV and return as a DataFrame."""
    return pd.read_csv(data_path)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features derived from the raw columns."""
    df = df.copy()
    df["RoomsPerHousehold"] = df["AveRooms"] / df["HouseAge"].clip(lower=1)
    df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"].clip(lower=1)
    df["PopulationPerHousehold"] = df["Population"] / df["HouseAge"].clip(lower=1)
    return df


def remove_outliers(df: pd.DataFrame, target_col: str = "MedianHouseValue") -> pd.DataFrame:
    """Remove rows where the target value is at the clipping ceiling (5.0)."""
    return df[df[target_col] < 5.0].copy()


def get_feature_target_split(
    df: pd.DataFrame,
    target_col: str = "MedianHouseValue",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) feature / target split."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Perform train/test split and standard-scale the features.

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)
