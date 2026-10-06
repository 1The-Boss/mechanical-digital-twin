import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


class DataCleaner:
    def __init__(self, strategy: str = 'interpolate', max_gap: int = 10):
        self.strategy = strategy
        self.max_gap = max_gap
        self.imputer = SimpleImputer(strategy='linear')

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()

        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

        if self.strategy == 'interpolate':
            df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='linear', limit=self.max_gap)
            df_clean[numeric_cols] = df_clean[numeric_cols].ffill().bfill()
        elif self.strategy == 'drop':
            df_clean = df_clean.dropna()
        elif self.strategy == 'impute':
            df_clean[numeric_cols] = self.imputer.fit_transform(df_clean[numeric_cols])

        return df_clean

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        issues = []

        if df.isnull().any().any():
            issues.append("DataFrame contains NaN values")

        if np.isinf(df.select_dtypes(include=[np.number])).any().any():
            issues.append("DataFrame contains infinite values")

        required_cols = ['time', 'theta2', 'omega2', 'alpha2', 'input_torque']
        for col in required_cols:
            if col not in df.columns:
                issues.append(f"Missing required column: {col}")

        if 'time' in df.columns:
            time_diff = df['time'].diff().dropna()
            if not np.allclose(time_diff, time_diff.iloc[0], rtol=0.1):
                issues.append("Time steps are not uniform")

        return len(issues) == 0, issues

    def remove_outliers(self, df: pd.DataFrame, columns: List[str], n_std: float = 3.0) -> pd.DataFrame:
        df_clean = df.copy()
        for col in columns:
            if col in df_clean.columns:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                mask = np.abs(df_clean[col] - mean) <= n_std * std
                df_clean = df_clean[mask]
        return df_clean


def clean_simulation_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaner = DataCleaner()
    return cleaner.clean(df)


def validate_simulation_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    cleaner = DataCleaner()
    return cleaner.validate(df)