import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
from pathlib import Path


class FeatureScaler:
    def __init__(self, scaler_type: str = 'standard', feature_names: Optional[List[str]] = None):
        self.scaler_type = scaler_type
        self.feature_names = feature_names
        self.scaler = None
        self._create_scaler()

    def _create_scaler(self):
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'FeatureScaler':
        if feature_names is not None:
            self.feature_names = feature_names
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        self.fit(X, feature_names)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return self.scaler.inverse_transform(X)

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'scaler_type': self.scaler_type,
            'feature_names': self.feature_names
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'FeatureScaler':
        data = joblib.load(filepath)
        obj = cls(data['scaler_type'], data['feature_names'])
        obj.scaler = data['scaler']
        return obj


class MultiScaler:
    def __init__(self):
        self.scalers: Dict[str, FeatureScaler] = {}

    def add_scaler(self, name: str, scaler: FeatureScaler):
        self.scalers[name] = scaler

    def fit(self, data_dict: Dict[str, np.ndarray]) -> 'MultiScaler':
        for name, X in data_dict.items():
            if name not in self.scalers:
                self.scalers[name] = FeatureScaler()
            self.scalers[name].fit(X)
        return self

    def transform(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        result = {}
        for name, X in data_dict.items():
            if name in self.scalers:
                result[name] = self.scalers[name].transform(X)
            else:
                result[name] = X
        return result

    def save_all(self, base_dir: str):
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        for name, scaler in self.scalers.items():
            scaler.save(f"{base_dir}/{name}_scaler.pkl")

    @classmethod
    def load_all(cls, base_dir: str, scaler_names: List[str]) -> 'MultiScaler':
        multi = cls()
        for name in scaler_names:
            multi.scalers[name] = FeatureScaler.load(f"{base_dir}/{name}_scaler.pkl")
        return multi


def get_default_feature_columns() -> List[str]:
    return [
        'theta2', 'theta3', 'theta4',
        'omega2', 'omega3', 'omega4',
        'alpha2', 'alpha3', 'alpha4',
        'input_torque'
    ]


def get_residual_feature_columns() -> List[str]:
    return [
        'residual_theta2', 'residual_omega2', 'residual_alpha2', 'residual_torque',
        'residual_theta2_mean', 'residual_theta2_std', 'residual_theta2_rms',
        'residual_omega2_mean', 'residual_omega2_std', 'residual_omega2_rms',
        'residual_alpha2_mean', 'residual_alpha2_std', 'residual_alpha2_rms',
        'residual_torque_mean', 'residual_torque_std', 'residual_torque_rms'
    ]


def scale_features(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    scaler_type: str = 'standard'
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], FeatureScaler]:
    scaler = FeatureScaler(scaler_type)
    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler