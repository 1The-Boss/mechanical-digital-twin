import torch
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
from physics import FourBarParams, DEFAULT_PARAMS, load_params_from_config
from physics.kinematics import solve_position, calculate_velocity, calculate_acceleration
from physics.dynamics import inverse_dynamics
from physics.friction import DEFAULT_FRICTION, FrictionModel
from models import ResidualMLP, PhysicsPredictor, HybridPredictor, HybridModel


class PhysicsPredictorWrapper:
    def __init__(self, params: FourBarParams, friction_model: FrictionModel = None):
        self.params = params
        self.friction_model = friction_model or DEFAULT_FRICTION

    def predict(self, theta2: float, omega2: float, alpha2: float = 0.0) -> Dict[str, float]:
        try:
            theta3, theta4 = solve_position(theta2, self.params)
            omega3, omega4 = calculate_velocity(theta2, omega2, self.params)
            alpha3, alpha4 = calculate_acceleration(theta2, omega2, alpha2, self.params)

            torque = inverse_dynamics(
                theta2, theta3, theta4,
                omega2, omega3, omega4,
                alpha2, alpha3, alpha4,
                self.params,
                self.friction_model.coulomb,
                self.friction_model.viscous
            )

            return {
                'theta3': theta3,
                'theta4': theta4,
                'omega3': omega3,
                'omega4': omega4,
                'alpha2': alpha2,
                'alpha3': alpha3,
                'alpha4': alpha4,
                'torque': torque
            }
        except ValueError as e:
            return {
                'theta3': 0.0, 'theta4': 0.0,
                'omega3': 0.0, 'omega4': 0.0,
                'alpha2': 0.0, 'alpha3': 0.0, 'alpha4': 0.0,
                'torque': 0.0,
                'error': str(e)
            }

    def predict_batch(self, theta2: np.ndarray, omega2: np.ndarray, alpha2: np.ndarray = None) -> Dict[str, np.ndarray]:
        n = len(theta2)
        if alpha2 is None:
            alpha2 = np.zeros(n)

        results = {
            'theta3': np.zeros(n),
            'theta4': np.zeros(n),
            'omega3': np.zeros(n),
            'omega4': np.zeros(n),
            'alpha2': np.zeros(n),
            'alpha3': np.zeros(n),
            'alpha4': np.zeros(n),
            'torque': np.zeros(n)
        }

        for i in range(n):
            pred = self.predict(theta2[i], omega2[i], alpha2[i])
            for k in results:
                results[k][i] = pred[k]

        return results


class ResidualPredictor:
    def __init__(
        self,
        model_path: str = 'models_saved/residual_model.pt',
        input_scaler_path: str = 'models_saved/residual_input_scaler.pkl',
        output_scaler_path: str = 'models_saved/residual_output_scaler.pkl',
        input_dim: int = 10,
        output_dim: int = 4,
        hidden_dims: list = None,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.input_scaler = joblib.load(input_scaler_path)
        self.output_scaler = joblib.load(output_scaler_path)

        self.model = ResidualMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims or [128, 64, 32]
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(1, -1)

        features_scaled = self.input_scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)

        with torch.no_grad():
            pred = self.model(features_tensor).cpu().numpy()

        pred_orig = self.output_scaler.inverse_transform(pred)
        return pred_orig


class HybridPredictorWrapper:
    def __init__(
        self,
        params: FourBarParams,
        friction_model: FrictionModel,
        residual_predictor: ResidualPredictor,
        physics_predictor: PhysicsPredictorWrapper
    ):
        self.params = params
        self.friction_model = friction_model
        self.residual_predictor = residual_predictor
        self.physics_predictor = physics_predictor

    def predict(self, theta2: float, omega2: float, alpha2: float = 0.0) -> Dict[str, Any]:
        physics = self.physics_predictor.predict(theta2, omega2, alpha2)

        features = np.array([
            theta2, omega2, alpha2, physics['torque'],
            physics['theta3'], physics['theta4'],
            physics['omega3'], physics['omega4'],
            physics['alpha3'], physics['alpha4']
        ])

        residual = self.residual_predictor.predict(features)

        return {
            'physics': physics,
            'residual': {
                'theta': residual[0, 0] if residual.shape[1] > 0 else 0.0,
                'omega': residual[0, 1] if residual.shape[1] > 1 else 0.0,
                'alpha': residual[0, 2] if residual.shape[1] > 2 else 0.0,
                'torque': residual[0, 3] if residual.shape[1] > 3 else 0.0
            },
            'hybrid': {
                'theta3': physics['theta3'] + (residual[0, 0] if residual.shape[1] > 0 else 0.0),
                'theta4': physics['theta4'] + (residual[0, 1] if residual.shape[1] > 1 else 0.0),
                'omega3': physics['omega3'] + (residual[0, 1] if residual.shape[1] > 1 else 0.0),
                'omega4': physics['omega4'] + (residual[0, 1] if residual.shape[1] > 1 else 0.0),
                'alpha3': physics['alpha3'] + (residual[0, 2] if residual.shape[1] > 2 else 0.0),
                'alpha4': physics['alpha4'] + (residual[0, 2] if residual.shape[1] > 2 else 0.0),
                'torque': physics['torque'] + (residual[0, 3] if residual.shape[1] > 3 else 0.0)
            }
        }


def create_predictors(
    config_path: str = 'config.yaml',
    model_dir: str = 'models_saved',
    device: str = 'cpu'
) -> Dict[str, Any]:
    params = load_params_from_config(config_path)
    friction_model = DEFAULT_FRICTION

    physics_predictor = PhysicsPredictorWrapper(params, friction_model)

    residual_predictor = ResidualPredictor(
        model_path=f'{model_dir}/residual_model.pt',
        input_scaler_path=f'{model_dir}/residual_input_scaler.pkl',
        output_scaler_path=f'{model_dir}/residual_output_scaler.pkl',
        device=device
    )

    hybrid_predictor = HybridPredictorWrapper(
        params, friction_model, residual_predictor, physics_predictor
    )

    return {
        'physics': physics_predictor,
        'residual': residual_predictor,
        'hybrid': hybrid_predictor
    }