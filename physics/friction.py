import numpy as np
from typing import Union


def friction_torque(
    omega: Union[float, np.ndarray],
    coulomb: float,
    viscous: float
) -> Union[float, np.ndarray]:
    omega = np.asarray(omega)
    return coulomb * np.sign(omega) + viscous * omega


def friction_torque_smooth(
    omega: Union[float, np.ndarray],
    coulomb: float,
    viscous: float,
    stiction_velocity: float = 1e-3
) -> Union[float, np.ndarray]:
    omega = np.asarray(omega)
    sign_smooth = np.tanh(omega / stiction_velocity)
    return coulomb * sign_smooth + viscous * omega


def stribeck_friction(
    omega: Union[float, np.ndarray],
    coulomb: float,
    viscous: float,
    stribeck_velocity: float = 0.01,
    static_friction: float = None
) -> Union[float, np.ndarray]:
    omega = np.asarray(omega)
    omega_abs = np.abs(omega)

    if static_friction is None:
        static_friction = coulomb * 1.5

    stribeck_factor = np.exp(-(omega_abs / stribeck_velocity)**2)
    friction = (static_friction * stribeck_factor + coulomb * (1 - stribeck_factor)) * np.sign(omega)
    friction += viscous * omega

    return friction


class FrictionModel:
    def __init__(
        self,
        coulomb: np.ndarray,
        viscous: np.ndarray,
        model_type: str = "coulomb_viscous",
        stiction_velocity: float = 1e-3,
        stribeck_velocity: float = 0.01,
        static_friction: np.ndarray = None
    ):
        self.coulomb = np.asarray(coulomb)
        self.viscous = np.asarray(viscous)
        self.model_type = model_type
        self.stiction_velocity = stiction_velocity
        self.stribeck_velocity = stribeck_velocity
        self.static_friction = static_friction

    def compute(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega)

        if self.model_type == "coulomb_viscous":
            return friction_torque(omega, self.coulomb, self.viscous)
        elif self.model_type == "smooth":
            return friction_torque_smooth(omega, self.coulomb, self.viscous, self.stiction_velocity)
        elif self.model_type == "stribeck":
            if self.static_friction is None:
                self.static_friction = self.coulomb * 1.5
            return stribeck_friction(omega, self.coulomb, self.viscous,
                                     self.stribeck_velocity, self.static_friction)
        else:
            raise ValueError(f"Unknown friction model: {self.model_type}")

    @classmethod
    def from_config(cls, config: dict) -> 'FrictionModel':
        return cls(
            coulomb=np.array(config.get('coulomb', [0.1, 0.05, 0.05])),
            viscous=np.array(config.get('viscous', [0.01, 0.005, 0.005])),
            model_type=config.get('model_type', 'coulomb_viscous'),
            stiction_velocity=config.get('stiction_velocity', 1e-3),
            stribeck_velocity=config.get('stribeck_velocity', 0.01),
            static_friction=np.array(config.get('static_friction')) if config.get('static_friction') else None
        )


DEFAULT_FRICTION = FrictionModel(
    coulomb=np.array([0.1, 0.05, 0.05]),
    viscous=np.array([0.01, 0.005, 0.005])
)