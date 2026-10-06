import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any

from physics import FourBarParams, DEFAULT_PARAMS, load_params_from_config
from physics.kinematics import solve_position, calculate_velocity, calculate_acceleration
from physics.dynamics import inverse_dynamics
from physics.friction import DEFAULT_FRICTION
from simulation import load_consolidated_data
from training.train_residual import run_residual_training, prepare_residual_data
from training.train_autoencoder import run_autoencoder_training
from training.train_classifier import run_classifier_training


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


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_all_data(data_dir: str) -> pd.DataFrame:
    import glob
    files = glob.glob(f"{data_dir}/*.parquet")
    if not files:
        files = glob.glob(f"{data_dir}/*.csv")

    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    dfs = []
    for f in files:
        if f.endswith('.parquet'):
            dfs.append(pd.read_parquet(f))
        else:
            dfs.append(pd.read_csv(f))

    return pd.concat(dfs, ignore_index=True)


def train_all(config_path: str = 'mechanical_digital_twin/config.yaml', data_path: str = None):
    print("=" * 60)
    print("PHYSICS-INFORMED HYBRID DIGITAL TWIN - TRAINING PIPELINE")
    print("=" * 60)

    config = load_config(config_path)
    params = load_params_from_config(config_path)
    friction_model = DEFAULT_FRICTION

    # Resolve paths relative to mechanical_digital_twin directory
    base_dir = Path(config_path).parent
    if data_path is None:
        data_path = config.get('paths', {}).get('data_processed', 'data/processed')
    data_path = str(base_dir / data_path)

    print(f"\nLoading data from {data_path}...")
    df = load_all_data(data_path)
    print(f"Loaded {len(df)} samples from {df['simulation_id'].nunique()} simulations")
    print(f"Fault classes: {sorted(df['fault_class'].unique())}")

    print("\nComputing physics predictions...")
    physics_predictions = compute_physics_predictions(df, params, friction_model)

    df['residual_theta2'] = df['theta2'] - df['theta2']
    df['residual_omega2'] = df['omega2'] - df['omega2']
    df['residual_alpha2'] = df['alpha2'] - physics_predictions['physics_alpha2']
    df['residual_torque'] = df['input_torque'] - physics_predictions['physics_torque']

    # Model save paths
    models_dir = base_dir / 'models_saved'
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("TRAINING RESIDUAL MODEL")
    print("=" * 60)
    residual_model, input_scaler, output_scaler, residual_metrics = run_residual_training(
        df, physics_predictions, config.get('training', {}),
        model_save_path=str(models_dir / 'residual_model.pt'),
        scaler_save_dir=str(models_dir)
    )

    print("\n" + "=" * 60)
    print("TRAINING AUTOENCODER")
    print("=" * 60)
    autoencoder, ae_scaler, threshold, ae_metrics = run_autoencoder_training(
        df, physics_predictions, config.get('training', {}),
        model_save_path=str(models_dir / 'autoencoder.pt'),
        scaler_save_path=str(models_dir / 'autoencoder_scaler.pkl'),
        threshold_save_path=str(models_dir / 'anomaly_threshold.json')
    )

    print("\n" + "=" * 60)
    print("TRAINING FAULT CLASSIFIER")
    print("=" * 60)
    classifier, clf_scaler, clf_metrics = run_classifier_training(
        df, physics_predictions, config.get('training', {}),
        model_save_path=str(models_dir / 'fault_classifier.pkl'),
        scaler_save_path=str(models_dir / 'classifier_scaler.pkl')
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Residual Model - Test RMSE: {residual_metrics.get('rmse', 'N/A'):.6f}, R²: {residual_metrics.get('mean_r2', 'N/A'):.4f}")
    print(f"Autoencoder - Test Recon Error: {ae_metrics.get('mean_error', 'N/A'):.6f}, Threshold: {threshold:.6f}")
    print(f"Classifier - Test Accuracy: {clf_metrics.get('accuracy', 'N/A'):.4f}, F1: {clf_metrics.get('f1', 'N/A'):.4f}")

    training_summary = {
        'residual_model': residual_metrics,
        'autoencoder': {**ae_metrics, 'threshold': threshold},
        'classifier': clf_metrics
    }

    with open(models_dir / 'training_summary.json', 'w') as f:
        json.dump(training_summary, f, indent=2, default=str)

    print(f"\nAll models saved to {models_dir}/")
    return training_summary


if __name__ == '__main__':
    train_all()