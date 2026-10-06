import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
from physics import FourBarParams
from physics.kinematics import solve_position, calculate_velocity, calculate_acceleration
from physics.dynamics import inverse_dynamics
from physics.friction import FrictionModel
from models.residual_model import ResidualMLP


class PhysicsPredictor:
    def __init__(
        self,
        params: FourBarParams,
        friction_model: Optional[FrictionModel] = None
    ):
        self.params = params
        self.friction_model = friction_model

    def predict_kinematics(self, theta2: np.ndarray, omega2: np.ndarray) -> Dict[str, np.ndarray]:
        n = len(theta2)
        theta3 = np.zeros(n)
        theta4 = np.zeros(n)
        omega3 = np.zeros(n)
        omega4 = np.zeros(n)
        alpha2 = np.zeros(n)
        alpha3 = np.zeros(n)
        alpha4 = np.zeros(n)

        for i in range(n):
            try:
                theta3[i], theta4[i] = solve_position(theta2[i], self.params)
                omega3[i], omega4[i] = calculate_velocity(theta2[i], omega2[i], self.params)

                if i > 0:
                    dt = 1.0 / 1000.0
                    alpha2[i] = (omega2[i] - omega2[i-1]) / dt
                else:
                    alpha2[i] = 0.0

                alpha3[i], alpha4[i] = calculate_acceleration(
                    theta2[i], omega2[i], alpha2[i], self.params
                )
            except ValueError:
                pass

        return {
            'theta3': theta3,
            'theta4': theta4,
            'omega3': omega3,
            'omega4': omega4,
            'alpha2': alpha2,
            'alpha3': alpha3,
            'alpha4': alpha4
        }

    def predict_torque(
        self,
        theta2: np.ndarray, theta3: np.ndarray, theta4: np.ndarray,
        omega2: np.ndarray, omega3: np.ndarray, omega4: np.ndarray,
        alpha2: np.ndarray, alpha3: np.ndarray, alpha4: np.ndarray
    ) -> np.ndarray:
        n = len(theta2)
        torque = np.zeros(n)

        coulomb = np.zeros(3) if self.friction_model is None else self.friction_model.coulomb
        viscous = np.zeros(3) if self.friction_model is None else self.friction_model.viscous

        for i in range(n):
            try:
                torque[i] = inverse_dynamics(
                    theta2[i], theta3[i], theta4[i],
                    omega2[i], omega3[i], omega4[i],
                    alpha2[i], alpha3[i], alpha4[i],
                    self.params, coulomb, viscous
                )
            except ValueError:
                torque[i] = 0.0

        return torque


class HybridModel(nn.Module):
    def __init__(
        self,
        physics_predictor: PhysicsPredictor,
        residual_model: ResidualMLP,
        input_scaler: Optional[Any] = None,
        output_scaler: Optional[Any] = None
    ):
        super().__init__()
        self.physics_predictor = physics_predictor
        self.residual_model = residual_model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        residual_pred = self.residual_model(x)
        return {
            'residual': residual_pred,
            'physics': None,
            'hybrid': None
        }

    def predict_hybrid(
        self,
        theta2: np.ndarray,
        omega2: np.ndarray,
        physics_features: np.ndarray
    ) -> Dict[str, np.ndarray]:
        physics_kin = self.physics_predictor.predict_kinematics(theta2, omega2)
        physics_torque = self.physics_predictor.predict_torque(
            theta2, physics_kin['theta3'], physics_kin['theta4'],
            omega2, physics_kin['omega3'], physics_kin['omega4'],
            physics_kin['alpha2'], physics_kin['alpha3'], physics_kin['alpha4']
        )

        with torch.no_grad():
            if self.input_scaler is not None:
                physics_features_scaled = self.input_scaler.transform(physics_features.reshape(1, -1))
            else:
                physics_features_scaled = physics_features.reshape(1, -1)

            physics_features_tensor = torch.FloatTensor(physics_features_scaled)
            residual_pred = self.residual_model(physics_features_tensor).numpy()

            if self.output_scaler is not None:
                residual_pred = self.output_scaler.inverse_transform(residual_pred)

        return {
            'physics_theta3': physics_kin['theta3'],
            'physics_theta4': physics_kin['theta4'],
            'physics_omega3': physics_kin['omega3'],
            'physics_omega4': physics_kin['omega4'],
            'physics_alpha2': physics_kin['alpha2'],
            'physics_alpha3': physics_kin['alpha3'],
            'physics_alpha4': physics_kin['alpha4'],
            'physics_torque': physics_torque,
            'residual': residual_pred.flatten(),
            'hybrid_torque': physics_torque + residual_pred.flatten()[3] if len(residual_pred.flatten()) > 3 else physics_torque
        }


def create_hybrid_model(
    params: FourBarParams,
    friction_model: FrictionModel,
    residual_model: ResidualMLP,
    input_scaler: Optional[Any] = None,
    output_scaler: Optional[Any] = None
) -> HybridModel:
    physics_predictor = PhysicsPredictor(params, friction_model)
    return HybridModel(physics_predictor, residual_model, input_scaler, output_scaler)


class HybridPredictor:
    def __init__(
        self,
        params: FourBarParams,
        friction_model: FrictionModel,
        residual_model_path: str,
        input_scaler_path: str,
        output_scaler_path: str,
        device: str = 'cpu'
    ):
        self.params = params
        self.friction_model = friction_model
        self.device = torch.device(device)

        self.physics_predictor = PhysicsPredictor(params, friction_model)

        self.residual_model = ResidualMLP()
        self.residual_model.load_state_dict(torch.load(residual_model_path, map_location=self.device))
        self.residual_model.to(self.device)
        self.residual_model.eval()

        import joblib
        self.input_scaler = joblib.load(input_scaler_path)
        self.output_scaler = joblib.load(output_scaler_path)

    def predict(self, theta2: float, omega2: float, alpha2: float = 0.0) -> Dict[str, float]:
        physics_kin = self.physics_predictor.predict_kinematics(
            np.array([theta2]), np.array([omega2])
        )

        physics_torque = self.physics_predictor.predict_torque(
            np.array([theta2]), np.array([physics_kin['theta3'][0]]), np.array([physics_kin['theta4'][0]]),
            np.array([omega2]), np.array([physics_kin['omega3'][0]]), np.array([physics_kin['omega4'][0]]),
            np.array([alpha2]), np.array([physics_kin['alpha3'][0]]), np.array([physics_kin['alpha4'][0]])
        )

        features = np.array([theta2, omega2, alpha2, physics_torque[0],
                            physics_kin['theta3'][0], physics_kin['theta4'][0],
                            physics_kin['omega3'][0], physics_kin['omega4'][0],
                            physics_kin['alpha3'][0], physics_kin['alpha4'][0]])

        features_scaled = self.input_scaler.transform(features.reshape(1, -1))
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)

        with torch.no_grad():
            residual_pred = self.residual_model(features_tensor).cpu().numpy()
            residual_pred = self.output_scaler.inverse_transform(residual_pred)

        return {
            'physics_theta3': physics_kin['theta3'][0],
            'physics_theta4': physics_kin['theta4'][0],
            'physics_omega3': physics_kin['omega3'][0],
            'physics_omega4': physics_kin['omega4'][0],
            'physics_torque': physics_torque[0],
            'residual_torque': residual_pred[0, 3] if residual_pred.shape[1] > 3 else 0.0,
            'hybrid_torque': physics_torque[0] + (residual_pred[0, 3] if residual_pred.shape[1] > 3 else 0.0)
        }