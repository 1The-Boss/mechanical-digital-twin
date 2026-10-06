import torch
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from models import Autoencoder, compute_reconstruction_error
from preprocessing import FeatureScaler
from preprocessing.windowing import create_windows, extract_statistical_features


class AnomalyDetector:
    def __init__(
        self,
        model_path: str = 'models_saved/autoencoder.pt',
        scaler_path: str = 'models_saved/autoencoder_scaler.pkl',
        threshold_path: str = 'models_saved/anomaly_threshold.json',
        hidden_dims: list = None,
        latent_dim: int = 8,
        window_size: int = 50,
        stride: int = 10,
        feature_columns: list = None,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.window_size = window_size
        self.stride = stride
        self.feature_columns = feature_columns or [
            'residual_theta2', 'residual_omega2', 'residual_alpha2', 'residual_torque'
        ]

        scaler_data = joblib.load(scaler_path)
        self.scaler = scaler_data['scaler'] if isinstance(scaler_data, dict) else scaler_data
        input_dim = self.scaler.n_features_in_

        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)
        self.threshold = threshold_data['threshold']
        self.threshold_percentile = threshold_data.get('percentile', 95)

        checkpoint = torch.load(model_path, map_location=self.device)
        encoder_weight_shape = checkpoint.get('encoder.0.weight', None)
        if encoder_weight_shape is not None:
            input_dim = encoder_weight_shape.shape[1]
            # encoder.0: input_dim -> hidden_dims[0]
            # encoder.2: hidden_dims[0] -> hidden_dims[1]
            # encoder.4: hidden_dims[1] -> latent_dim
            hidden_dims = [
                checkpoint['encoder.0.weight'].shape[0],  # 64
                checkpoint['encoder.2.weight'].shape[0],  # 32
            ]
            latent_dim = checkpoint['encoder.4.weight'].shape[0]  # 8

        self.model = Autoencoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims or [64, 32],
            latent_dim=latent_dim
        )
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def prepare_features(self, residuals: Dict[str, np.ndarray]) -> np.ndarray:
        n = len(residuals[self.feature_columns[0]])
        data = np.column_stack([residuals[col] for col in self.feature_columns])

        windows = create_windows(data, self.window_size, self.stride)
        stat_features, _ = extract_statistical_features(windows, self.feature_columns)
        stat_features_scaled = self.scaler.transform(stat_features)

        return stat_features_scaled

    def compute_anomaly_scores(self, residuals: Dict[str, np.ndarray]) -> np.ndarray:
        features = self.prepare_features(residuals)

        if len(features) == 0:
            return np.array([])

        features_tensor = torch.FloatTensor(features).to(self.device)

        with torch.no_grad():
            recon = self.model(features_tensor)
            errors = torch.mean((features_tensor - recon) ** 2, dim=1)

        return errors.cpu().numpy()

    def detect_anomalies(self, residuals: Dict[str, np.ndarray]) -> Dict[str, Any]:
        scores = self.compute_anomaly_scores(residuals)

        if len(scores) == 0:
            return {
                'anomaly_scores': np.array([]),
                'anomalies': np.array([]),
                'threshold': self.threshold,
                'status': 'NORMAL'
            }

        anomalies = scores > self.threshold
        status = 'ANOMALY' if np.any(anomalies) else 'NORMAL'

        return {
            'anomaly_scores': scores,
            'anomalies': anomalies,
            'threshold': self.threshold,
            'status': status,
            'max_score': float(np.max(scores)),
            'mean_score': float(np.mean(scores)),
            'anomaly_ratio': float(np.mean(anomalies))
        }

    def detect_single_window(self, residual_window: np.ndarray) -> Dict[str, Any]:
        stat_features = compute_window_features(residual_window)
        stat_features = stat_features.reshape(1, -1)
        stat_features_scaled = self.scaler.transform(stat_features)

        features_tensor = torch.FloatTensor(stat_features_scaled).to(self.device)

        with torch.no_grad():
            recon = self.model(features_tensor)
            error = torch.mean((features_tensor - recon) ** 2).item()

        is_anomaly = error > self.threshold
        status = 'ANOMALY' if is_anomaly else 'NORMAL'

        return {
            'anomaly_score': error,
            'is_anomaly': is_anomaly,
            'threshold': self.threshold,
            'status': status
        }


def compute_window_features(window: np.ndarray) -> np.ndarray:
    features = []
    features.append(np.mean(window, axis=0))
    features.append(np.std(window, axis=0))
    features.append(np.sqrt(np.mean(window**2, axis=0)))
    features.append(np.max(window, axis=0))
    features.append(np.min(window, axis=0))
    features.append(np.ptp(window, axis=0))
    return np.concatenate(features)


class OnlineAnomalyDetector:
    def __init__(
        self,
        model_path: str = 'models_saved/autoencoder.pt',
        scaler_path: str = 'models_saved/autoencoder_scaler.pkl',
        threshold_path: str = 'models_saved/anomaly_threshold.json',
        window_size: int = 50,
        feature_columns: list = None,
        device: str = 'cpu'
    ):
        self.detector = AnomalyDetector(
            model_path, scaler_path, threshold_path,
            window_size=window_size,
            feature_columns=feature_columns,
            device=device
        )
        self.buffer = {col: [] for col in (feature_columns or [
            'residual_theta2', 'residual_omega2', 'residual_alpha2', 'residual_torque'
        ])}
        self.window_size = window_size

    def update(self, residuals: Dict[str, float]) -> Optional[Dict[str, Any]]:
        for col in self.buffer:
            if col in residuals:
                self.buffer[col].append(residuals[col])

        for col in self.buffer:
            if len(self.buffer[col]) > self.window_size:
                self.buffer[col] = self.buffer[col][-self.window_size:]

        if all(len(self.buffer[col]) >= self.window_size for col in self.buffer):
            residual_arrays = {col: np.array(self.buffer[col]) for col in self.buffer}
            return self.detector.detect_single_window(
                np.column_stack([residual_arrays[col] for col in self.buffer])
            )

        return None

    def reset(self):
        for col in self.buffer:
            self.buffer[col] = []