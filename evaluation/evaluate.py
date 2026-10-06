import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import json

from physics import FourBarParams, load_params_from_config, DEFAULT_FRICTION
from physics.kinematics import solve_position, calculate_velocity, calculate_acceleration
from physics.dynamics import inverse_dynamics
from simulation import load_consolidated_data
from models import ResidualMLP, Autoencoder
from inference import AnomalyDetector, FaultDiagnosis


def compute_physics_predictions(df: pd.DataFrame, params: FourBarParams, friction_model) -> Dict[str, np.ndarray]:
    n = len(df)
    physics_theta3 = np.zeros(n)
    physics_theta4 = np.zeros(n)
    physics_omega3 = np.zeros(n)
    physics_omega4 = np.zeros(n)
    physics_alpha2 = np.zeros(n)
    physics_alpha3 = np.zeros(n)
    physics_alpha4 = np.zeros(n)
    physics_torque = np.zeros(n)

    coulomb = friction_model.coulomb
    viscous = friction_model.viscous

    for i in range(n):
        theta2 = df.iloc[i]['theta2']
        omega2 = df.iloc[i]['omega2']
        alpha2 = df.iloc[i]['alpha2'] if i < n - 1 else 0.0

        try:
            theta3, theta4 = solve_position(theta2, params)
            omega3, omega4 = calculate_velocity(theta2, omega2, params)
            alpha3, alpha4 = calculate_acceleration(theta2, omega2, alpha2, params)

            torque = inverse_dynamics(
                theta2, theta3, theta4,
                omega2, omega3, omega4,
                alpha2, alpha3, alpha4,
                params, coulomb, viscous
            )

            physics_theta3[i] = theta3
            physics_theta4[i] = theta4
            physics_omega3[i] = omega3
            physics_omega4[i] = omega4
            physics_alpha2[i] = alpha2
            physics_alpha3[i] = alpha3
            physics_alpha4[i] = alpha4
            physics_torque[i] = torque
        except ValueError:
            pass

    return {
        'physics_theta3': physics_theta3,
        'physics_theta4': physics_theta4,
        'physics_omega3': physics_omega3,
        'physics_omega4': physics_omega4,
        'physics_alpha2': physics_alpha2,
        'physics_alpha3': physics_alpha3,
        'physics_alpha4': physics_alpha4,
        'physics_torque': physics_torque
    }


def evaluate_physics_model(df: pd.DataFrame, physics_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
    metrics = {}

    for key in ['theta3', 'theta4', 'omega3', 'omega4', 'alpha3', 'alpha4', 'torque']:
        actual_key = key if key != 'torque' else 'input_torque'
        pred_key = f'physics_{key}' if key != 'torque' else 'physics_torque'

        if actual_key in df.columns and pred_key in physics_predictions:
            actual = df[actual_key].values
            pred = physics_predictions[pred_key]

            metrics[f'{key}_mae'] = mean_absolute_error(actual, pred)
            metrics[f'{key}_rmse'] = np.sqrt(mean_squared_error(actual, pred))
            metrics[f'{key}_r2'] = r2_score(actual, pred)

    return metrics


def evaluate_residual_model(
    df: pd.DataFrame,
    physics_predictions: Dict[str, np.ndarray],
    model_path: str = 'models_saved/residual_model.pt',
    input_scaler_path: str = 'models_saved/residual_input_scaler.pkl',
    output_scaler_path: str = 'models_saved/residual_output_scaler.pkl',
    device: str = 'cpu'
) -> Dict[str, float]:
    import torch

    input_scaler = joblib.load(input_scaler_path)
    output_scaler = joblib.load(output_scaler_path)

    feature_cols = ['theta2', 'omega2', 'alpha2', 'input_torque',
                    'theta3', 'theta4', 'omega3', 'omega4', 'alpha3', 'alpha4']

    X = df[feature_cols].values
    X_scaled = input_scaler.transform(X)

    target_cols = ['residual_theta2', 'residual_omega2', 'residual_alpha2', 'residual_torque']
    for col in target_cols:
        if col not in df.columns:
            physics_key = col.replace('residual_', 'physics_')
            if physics_key in physics_predictions:
                df[col] = df[col.replace('residual_', '')] - physics_predictions[physics_key]
            else:
                df[col] = 0.0

    y = df[target_cols].values

    model = ResidualMLP(input_dim=X.shape[1], output_dim=y.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        pred = model(X_tensor).cpu().numpy()
        pred_orig = output_scaler.inverse_transform(pred)

    metrics = {}
    for i, col in enumerate(target_cols):
        metrics[f'{col}_mae'] = mean_absolute_error(y[:, i], pred_orig[:, i])
        metrics[f'{col}_rmse'] = np.sqrt(mean_squared_error(y[:, i], pred_orig[:, i]))
        metrics[f'{col}_r2'] = r2_score(y[:, i], pred_orig[:, i])

    metrics['mean_mae'] = np.mean([metrics[f'{col}_mae'] for col in target_cols])
    metrics['mean_rmse'] = np.mean([metrics[f'{col}_rmse'] for col in target_cols])
    metrics['mean_r2'] = np.mean([metrics[f'{col}_r2'] for col in target_cols])

    return metrics


def evaluate_hybrid_model(
    df: pd.DataFrame,
    physics_predictions: Dict[str, np.ndarray],
    model_path: str = 'models_saved/residual_model.pt',
    input_scaler_path: str = 'models_saved/residual_input_scaler.pkl',
    output_scaler_path: str = 'models_saved/residual_output_scaler.pkl',
    device: str = 'cpu'
) -> Dict[str, float]:
    import torch

    residual_metrics = evaluate_residual_model(
        df, physics_predictions, model_path, input_scaler_path, output_scaler_path, device
    )

    input_scaler = joblib.load(input_scaler_path)
    output_scaler = joblib.load(output_scaler_path)

    feature_cols = ['theta2', 'omega2', 'alpha2', 'input_torque',
                    'theta3', 'theta4', 'omega3', 'omega4', 'alpha3', 'alpha4']

    X = df[feature_cols].values
    X_scaled = input_scaler.transform(X)

    target_cols = ['residual_theta2', 'residual_omega2', 'residual_alpha2', 'residual_torque']

    model = ResidualMLP(input_dim=X.shape[1], output_dim=len(target_cols))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        pred = model(X_tensor).cpu().numpy()
        pred_orig = output_scaler.inverse_transform(pred)

    hybrid_metrics = {}
    for i, col in enumerate(target_cols):
        actual_key = col.replace('residual_', '')
        if actual_key in df.columns:
            actual = df[actual_key].values
            physics_pred = physics_predictions.get(f'physics_{actual_key}', np.zeros_like(actual))
            hybrid_pred = physics_pred + pred_orig[:, i]

            hybrid_metrics[f'{col}_mae'] = mean_absolute_error(actual, hybrid_pred)
            hybrid_metrics[f'{col}_rmse'] = np.sqrt(mean_squared_error(actual, hybrid_pred))
            hybrid_metrics[f'{col}_r2'] = r2_score(actual, hybrid_pred)

    hybrid_metrics['mean_mae'] = np.mean([hybrid_metrics[f'{col}_mae'] for col in target_cols if f'{col}_mae' in hybrid_metrics])
    hybrid_metrics['mean_rmse'] = np.mean([hybrid_metrics[f'{col}_rmse'] for col in target_cols if f'{col}_rmse' in hybrid_metrics])
    hybrid_metrics['mean_r2'] = np.mean([hybrid_metrics[f'{col}_r2'] for col in target_cols if f'{col}_r2' in hybrid_metrics])

    return hybrid_metrics


def evaluate_anomaly_detection(
    df: pd.DataFrame,
    physics_predictions: Dict[str, np.ndarray],
    anomaly_detector: AnomalyDetector
) -> Dict[str, Any]:
    for col in ['residual_theta2', 'residual_omega2', 'residual_alpha2', 'residual_torque']:
        if col not in df.columns:
            physics_key = col.replace('residual_', 'physics_')
            if physics_key in physics_predictions:
                df[col] = df[col.replace('residual_', '')] - physics_predictions[physics_key]
            else:
                df[col] = 0.0

    residuals = {
        'residual_theta2': df['residual_theta2'].values,
        'residual_omega2': df['residual_omega2'].values,
        'residual_alpha2': df['residual_alpha2'].values,
        'residual_torque': df['residual_torque'].values
    }

    result = anomaly_detector.detect_anomalies(residuals)

    true_labels = (df['fault_class'] > 0).astype(int).values

    window_size = anomaly_detector.window_size
    stride = anomaly_detector.stride
    n_windows = len(result['anomalies'])

    if n_windows > 0:
        true_window_labels = np.zeros(n_windows)
        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            if end <= len(true_labels):
                true_window_labels[i] = 1 if np.any(true_labels[start:end] > 0) else 0

        pred_labels = result['anomalies'].astype(int)

        accuracy = accuracy_score(true_window_labels, pred_labels)
        precision = precision_score(true_window_labels, pred_labels, zero_division=0)
        recall = recall_score(true_window_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_window_labels, pred_labels, zero_division=0)

        result['accuracy'] = accuracy
        result['precision'] = precision
        result['recall'] = recall
        result['f1'] = f1
        result['confusion_matrix'] = confusion_matrix(true_window_labels, pred_labels).tolist()

    return result


def evaluate_fault_classification(
    df: pd.DataFrame,
    physics_predictions: Dict[str, np.ndarray],
    fault_diagnosis: FaultDiagnosis
) -> Dict[str, Any]:
    for col in ['residual_theta2', 'residual_omega2', 'residual_alpha2', 'residual_torque']:
        if col not in df.columns:
            physics_key = col.replace('residual_', 'physics_')
            if physics_key in physics_predictions:
                df[col] = df[col.replace('residual_', '')] - physics_predictions[physics_key]
            else:
                df[col] = 0.0

    result = fault_diagnosis.diagnose({
        'residual_theta2': df['residual_theta2'].values,
        'residual_omega2': df['residual_omega2'].values,
        'residual_alpha2': df['residual_alpha2'].values,
        'residual_torque': df['residual_torque'].values
    })

    true_labels = df['fault_class'].values

    window_size = fault_diagnosis.window_size
    stride = fault_diagnosis.stride

    n_windows = len(result['window_predictions'])
    if n_windows > 0:
        true_window_labels = np.zeros(n_windows, dtype=int)
        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            if end <= len(true_labels):
                true_window_labels[i] = true_labels[end - 1]

        pred_labels = result['window_predictions']

        accuracy = accuracy_score(true_window_labels, pred_labels)
        precision = precision_score(true_window_labels, pred_labels, average='weighted', zero_division=0)
        recall = recall_score(true_window_labels, pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_window_labels, pred_labels, average='weighted', zero_division=0)

        result['accuracy'] = accuracy
        result['precision'] = precision
        result['recall'] = recall
        result['f1'] = f1
        result['confusion_matrix'] = confusion_matrix(true_window_labels, pred_labels).tolist()
        result['classification_report'] = classification_report(
            true_window_labels, pred_labels,
            target_names=fault_diagnosis.class_names,
            output_dict=True, zero_division=0
        )

    return result


def evaluate_all(
    config_path: str = 'config.yaml',
    data_path: str = 'data/processed',
    model_dir: str = 'models_saved',
    device: str = 'cpu'
) -> Dict[str, Any]:
    params = load_params_from_config(config_path)
    friction_model = DEFAULT_FRICTION

    import glob
    files = glob.glob(f"{data_path}/*.parquet")
    if not files:
        files = glob.glob(f"{data_path}/*.csv")

    dfs = []
    for f in files:
        if f.endswith('.parquet'):
            dfs.append(pd.read_parquet(f))
        else:
            dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)

    physics_predictions = compute_physics_predictions(df, params, friction_model)

    print("Evaluating physics model...")
    physics_metrics = evaluate_physics_model(df, physics_predictions)

    print("Evaluating residual model...")
    residual_metrics = evaluate_residual_model(
        df, physics_predictions,
        f'{model_dir}/residual_model.pt',
        f'{model_dir}/residual_input_scaler.pkl',
        f'{model_dir}/residual_output_scaler.pkl',
        device
    )

    print("Evaluating hybrid model...")
    hybrid_metrics = evaluate_hybrid_model(
        df, physics_predictions,
        f'{model_dir}/residual_model.pt',
        f'{model_dir}/residual_input_scaler.pkl',
        f'{model_dir}/residual_output_scaler.pkl',
        device
    )

    print("Evaluating anomaly detection...")
    anomaly_detector = AnomalyDetector(
        model_path=f'{model_dir}/autoencoder.pt',
        scaler_path=f'{model_dir}/autoencoder_scaler.pkl',
        threshold_path=f'{model_dir}/anomaly_threshold.json',
        device=device
    )
    anomaly_metrics = evaluate_anomaly_detection(df, physics_predictions, anomaly_detector)

    print("Evaluating fault classification...")
    fault_diagnosis = FaultDiagnosis(
        classifier_path=f'{model_dir}/fault_classifier.pkl',
        scaler_path=f'{model_dir}/classifier_scaler.pkl',
        anomaly_detector=anomaly_detector
    )
    classification_metrics = evaluate_fault_classification(df, physics_predictions, fault_diagnosis)

    results = {
        'physics_model': physics_metrics,
        'residual_model': residual_metrics,
        'hybrid_model': hybrid_metrics,
        'anomaly_detection': anomaly_metrics,
        'fault_classification': classification_metrics
    }

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{model_dir}/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Physics Model - Mean MAE: {np.mean([v for k,v in physics_metrics.items() if 'mae' in k]):.6f}")
    print(f"Residual Model - Mean MAE: {residual_metrics['mean_mae']:.6f}, R²: {residual_metrics['mean_r2']:.4f}")
    print(f"Hybrid Model - Mean MAE: {hybrid_metrics['mean_mae']:.6f}, R²: {hybrid_metrics['mean_r2']:.4f}")
    print(f"Anomaly Detection - F1: {anomaly_metrics.get('f1', 'N/A'):.4f}")
    print(f"Fault Classification - Accuracy: {classification_metrics.get('accuracy', 'N/A'):.4f}")

    return results


def compare_models(
    physics_metrics: Dict,
    residual_metrics: Dict,
    hybrid_metrics: Dict
) -> pd.DataFrame:
    comparison = []

    for key in ['theta3', 'theta4', 'omega3', 'omega4', 'alpha3', 'alpha4', 'torque']:
        for metric in ['mae', 'rmse', 'r2']:
            p_key = f'{key}_{metric}'
            h_key = f'{key}_{metric}'

            row = {'variable': key, 'metric': metric}
            row['physics'] = physics_metrics.get(p_key, np.nan)
            row['hybrid'] = hybrid_metrics.get(h_key, np.nan)
            row['improvement'] = ((row['physics'] - row['hybrid']) / row['physics'] * 100) if row['physics'] != 0 else np.nan
            comparison.append(row)

    return pd.DataFrame(comparison)


if __name__ == '__main__':
    evaluate_all()