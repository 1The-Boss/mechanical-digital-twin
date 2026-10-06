import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm

from physics import (
    FourBarParams,
    solve_position,
    calculate_velocity,
    calculate_acceleration,
    forward_dynamics,
    inverse_dynamics,
    vector_loop_error,
    DEFAULT_FRICTION,
    FrictionModel,
    FaultConfig,
    apply_fault,
    get_fault_friction_params
)


@dataclass
class SimulationResult:
    time: np.ndarray
    theta2: np.ndarray
    theta3: np.ndarray
    theta4: np.ndarray
    omega2: np.ndarray
    omega3: np.ndarray
    omega4: np.ndarray
    alpha2: np.ndarray
    alpha3: np.ndarray
    alpha4: np.ndarray
    input_torque: np.ndarray
    fault_type: str
    fault_severity: float
    simulation_id: int
    params: FourBarParams


class FourBarSimulator:
    def __init__(
        self,
        params: FourBarParams,
        friction_model: Optional[FrictionModel] = None,
        fault: Optional[FaultConfig] = None,
        integration_method: str = "rk4"
    ):
        self.params = params
        self.friction_model = friction_model or DEFAULT_FRICTION
        self.fault = fault or FaultConfig("normal", 1.0)
        self.integration_method = integration_method

        self.fault_params = apply_fault(params, self.fault)
        self.coulomb, self.viscous = get_fault_friction_params(
            self.fault,
            self.friction_model.coulomb,
            self.friction_model.viscous
        )

    def simulate(
        self,
        duration: float,
        timestep: float,
        initial_conditions: Optional[Dict] = None,
        input_torque_profile: Optional[np.ndarray] = None,
        input_omega_profile: Optional[np.ndarray] = None
    ) -> SimulationResult:
        n_steps = int(duration / timestep) + 1
        time = np.linspace(0, duration, n_steps)

        if initial_conditions is None:
            theta2_0 = 0.0
            omega2_0 = 10.0
        else:
            theta2_0 = initial_conditions.get('theta2', 0.0)
            omega2_0 = initial_conditions.get('omega2', 10.0)

        theta2 = np.zeros(n_steps)
        theta3 = np.zeros(n_steps)
        theta4 = np.zeros(n_steps)
        omega2 = np.zeros(n_steps)
        omega3 = np.zeros(n_steps)
        omega4 = np.zeros(n_steps)
        alpha2 = np.zeros(n_steps)
        alpha3 = np.zeros(n_steps)
        alpha4 = np.zeros(n_steps)
        input_torque = np.zeros(n_steps)

        theta2[0] = theta2_0
        omega2[0] = omega2_0

        try:
            theta3_0, theta4_0 = solve_position(theta2_0, self.fault_params)
        except ValueError as e:
            raise ValueError(f"Initial position invalid: {e}")

        theta3[0] = theta3_0
        theta4[0] = theta4_0

        try:
            omega3[0], omega4[0] = calculate_velocity(theta2_0, omega2_0, self.fault_params)
        except ValueError:
            omega3[0], omega4[0] = 0.0, 0.0

        if input_torque_profile is not None:
            tau_profile = input_torque_profile
        elif input_omega_profile is not None:
            tau_profile = None
        else:
            tau_profile = np.full(n_steps, 5.0)

        for i in tqdm(range(n_steps - 1), desc="Simulating", leave=False):
            dt = timestep

            if input_omega_profile is not None:
                omega2[i+1] = input_omega_profile[i+1] if i+1 < len(input_omega_profile) else omega2[i]
                alpha2[i] = (omega2[i+1] - omega2[i]) / dt
                theta2[i+1] = theta2[i] + omega2[i] * dt + 0.5 * alpha2[i] * dt**2
            else:
                if input_torque_profile is not None:
                    tau = tau_profile[i]
                else:
                    tau = 5.0

                try:
                    a2, a3, a4 = forward_dynamics(
                        theta2[i], theta3[i], theta4[i],
                        omega2[i], omega3[i], omega4[i],
                        tau, self.fault_params,
                        self.coulomb, self.viscous
                    )
                except ValueError:
                    a2, a3, a4 = 0.0, 0.0, 0.0

                alpha2[i] = a2
                alpha3[i] = a3
                alpha4[i] = a4

                omega2[i+1] = omega2[i] + alpha2[i] * dt
                omega3[i+1] = omega3[i] + alpha3[i] * dt
                omega4[i+1] = omega4[i] + alpha4[i] * dt

                theta2[i+1] = theta2[i] + omega2[i] * dt + 0.5 * alpha2[i] * dt**2
                theta3[i+1] = theta3[i] + omega3[i] * dt + 0.5 * alpha3[i] * dt**2
                theta4[i+1] = theta4[i] + omega4[i] * dt + 0.5 * alpha4[i] * dt**2

                input_torque[i] = tau

            try:
                theta3[i+1], theta4[i+1] = solve_position(theta2[i+1], self.fault_params)
                omega3[i+1], omega4[i+1] = calculate_velocity(theta2[i+1], omega2[i+1], self.fault_params)
                alpha3[i+1], alpha4[i+1] = calculate_acceleration(
                    theta2[i+1], omega2[i+1], alpha2[i+1] if i+1 < n_steps-1 else alpha2[i],
                    self.fault_params
                )
            except ValueError:
                pass

            error = vector_loop_error(theta2[i+1], theta3[i+1], theta4[i+1], self.fault_params)
            if error > 1e-3:
                pass

        if input_omega_profile is not None:
            try:
                tau_computed = inverse_dynamics(
                    theta2[-1], theta3[-1], theta4[-1],
                    omega2[-1], omega3[-1], omega4[-1],
                    alpha2[-1], alpha3[-1], alpha4[-1],
                    self.fault_params, self.coulomb, self.viscous
                )
                input_torque[-1] = tau_computed
            except ValueError:
                input_torque[-1] = 0.0
        else:
            input_torque[-1] = input_torque[-2] if n_steps > 1 else 0.0

        return SimulationResult(
            time=time,
            theta2=theta2,
            theta3=theta3,
            theta4=theta4,
            omega2=omega2,
            omega3=omega3,
            omega4=omega4,
            alpha2=alpha2,
            alpha3=alpha3,
            alpha4=alpha4,
            input_torque=input_torque,
            fault_type=self.fault.fault_type,
            fault_severity=self.fault.severity,
            simulation_id=0,
            params=self.fault_params
        )

    def simulate_constant_speed(
        self,
        duration: float,
        timestep: float,
        omega2_nominal: float = 10.0,
        initial_theta2: float = 0.0
    ) -> SimulationResult:
        n_steps = int(duration / timestep) + 1
        input_omega = np.full(n_steps, omega2_nominal)
        return self.simulate(
            duration, timestep,
            initial_conditions={'theta2': initial_theta2, 'omega2': omega2_nominal},
            input_omega_profile=input_omega
        )

    def simulate_constant_torque(
        self,
        duration: float,
        timestep: float,
        tau_nominal: float = 5.0,
        initial_theta2: float = 0.0,
        initial_omega2: float = 10.0
    ) -> SimulationResult:
        n_steps = int(duration / timestep) + 1
        input_torque = np.full(n_steps, tau_nominal)
        return self.simulate(
            duration, timestep,
            initial_conditions={'theta2': initial_theta2, 'omega2': initial_omega2},
            input_torque_profile=input_torque
        )


def run_simulation(
    params: FourBarParams,
    duration: float,
    timestep: float,
    fault: Optional[FaultConfig] = None,
    omega2_nominal: float = 10.0,
    initial_theta2: float = 0.0,
    friction_model: Optional[FrictionModel] = None
) -> SimulationResult:
    simulator = FourBarSimulator(params, friction_model, fault)
    return simulator.simulate_constant_speed(duration, timestep, omega2_nominal, initial_theta2)