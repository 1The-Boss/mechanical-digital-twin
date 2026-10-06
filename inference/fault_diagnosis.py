import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from preprocessing import FeatureScaler
from preprocessing.windowing import create_windows, extract_statistical_features
from inference.anomaly_detector import AnomalyDetector
from physics import FAULT_CLASSES


class FaultDiagnosis:
    def __init__(
        self,
        classifier_path: str = 'models_saved/fault_classifier.pkl',
        scaler_path: str = 'models_saved/classifier_scaler.pkl',
        anomaly_detector: Optional[AnomalyDetector] = None,
        window_size: int = 50,
        stride: int = 10,
        feature_columns: list = None,
        class_names: list = None
    ):
        self.classifier = joblib.load(classifier_path)
        scaler_data = joblib.load(scaler_path)
        self.scaler = scaler_data['scaler'] if isinstance(scaler_data, dict) else scaler_data
        self.anomaly_detector = anomaly_detector
        self.window_size = window_size
        self.stride = stride
        self.feature_columns = feature_columns or [
            'residual_theta2', 'residual_omega2', 'residual_alpha2', 'residual_torque'
        ]
        # Use classifier's actual classes
        self.class_names = class_names or [str(c) for c in self.classifier.classes_]

    def prepare_features(self, residuals: Dict[str, np.ndarray]) -> np.ndarray:
        data = np.column_stack([residuals[col] for col in self.feature_columns])

        windows = create_windows(data, self.window_size, self.stride)
        stat_features, _ = extract_statistical_features(windows, self.feature_columns)
        stat_features_scaled = self.scaler.transform(stat_features)

        return stat_features_scaled

    def diagnose(self, residuals: Dict[str, np.ndarray]) -> Dict[str, Any]:
        features = self.prepare_features(residuals)

        if len(features) == 0:
            return {
                'fault_type': 'NORMAL',
                'fault_probabilities': {name: 0.0 for name in self.class_names},
                'confidence': 0.0,
                'anomaly_detected': False
            }

        predictions = self.classifier.predict(features)
        probabilities = self.classifier.predict_proba(features)

        final_prediction = np.bincount(predictions).argmax()
        final_probs = np.mean(probabilities, axis=0)

        fault_type = self.class_names[final_prediction]
        confidence = float(final_probs[final_prediction])

        anomaly_info = {'anomaly_detected': False}
        if self.anomaly_detector is not None:
            anomaly_info = self.anomaly_detector.detect_anomalies(residuals)

        return {
            'fault_type': fault_type,
            'fault_class': int(final_prediction),
            'fault_probabilities': {name: float(final_probs[i]) for i, name in enumerate(self.class_names)},
            'confidence': confidence,
            'anomaly_detected': anomaly_info.get('status') == 'ANOMALY',
            'anomaly_score': anomaly_info.get('max_score', 0.0),
            'anomaly_threshold': anomaly_info.get('threshold', 0.0),
            'window_predictions': predictions.tolist(),
            'window_probabilities': probabilities.tolist()
        }

    def diagnose_single_window(self, residual_window: np.ndarray) -> Dict[str, Any]:
        from ..inference.anomaly_detector import compute_window_features

        stat_features = compute_window_features(residual_window)
        stat_features = stat_features.reshape(1, -1)
        stat_features_scaled = self.scaler.transform(stat_features)

        prediction = self.classifier.predict(stat_features_scaled)[0]
        probabilities = self.classifier.predict_proba(stat_features_scaled)[0]

        fault_type = self.class_names[prediction]
        confidence = float(probabilities[prediction])

        anomaly_info = {'anomaly_detected': False}
        if self.anomaly_detector is not None:
            anomaly_info = self.anomaly_detector.detect_single_window(residual_window)

        return {
            'fault_type': fault_type,
            'fault_class': int(prediction),
            'fault_probabilities': {name: float(probabilities[i]) for i, name in enumerate(self.class_names)},
            'confidence': confidence,
            'anomaly_detected': anomaly_info.get('is_anomaly', False),
            'anomaly_score': anomaly_info.get('anomaly_score', 0.0),
            'anomaly_threshold': anomaly_info.get('threshold', 0.0)
        }


class OnlineFaultDiagnosis:
    def __init__(
        self,
        classifier_path: str = 'models_saved/fault_classifier.pkl',
        scaler_path: str = 'models_saved/classifier_scaler.pkl',
        anomaly_detector: Optional[AnomalyDetector] = None,
        window_size: int = 50,
        feature_columns: list = None,
        class_names: list = None
    ):
        self.diagnosis = FaultDiagnosis(
            classifier_path, scaler_path, anomaly_detector,
            window_size, 1, feature_columns, class_names
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
            return self.diagnosis.diagnose_single_window(
                np.column_stack([residual_arrays[col] for col in self.buffer])
            )

        return None

    def reset(self):
        for col in self.buffer:
            self.buffer[col] = []


def create_diagnosis_pipeline(
    config_path: str = 'config.yaml',
    model_dir: str = 'models_saved',
    device: str = 'cpu'
) -> Dict[str, Any]:
    from .anomaly_detector import AnomalyDetector

    anomaly_detector = AnomalyDetector(
        model_path=f'{model_dir}/autoencoder.pt',
        scaler_path=f'{model_dir}/autoencoder_scaler.pkl',
        threshold_path=f'{model_dir}/anomaly_threshold.json',
        device=device
    )

    fault_diagnosis = FaultDiagnosis(
        classifier_path=f'{model_dir}/fault_classifier.pkl',
        scaler_path=f'{model_dir}/classifier_scaler.pkl',
        anomaly_detector=anomaly_detector
    )

    return {
        'anomaly_detector': anomaly_detector,
        'fault_diagnosis': fault_diagnosis
    }