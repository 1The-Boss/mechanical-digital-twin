import numpy as np
from typing import Tuple, Optional
from .mechanism import FourBarParams
from .kinematics import solve_position, calculate_velocity, calculate_acceleration
from .friction import friction_torque


def kinetic_energy(
    theta2: float, theta3: float, theta4: float,
    omega2: float, omega3: float, omega4: float,
    params: FourBarParams
) -> float:
    T = 0.5 * params.I2 * omega2**2
    T += 0.5 * params.I3 * omega3**2
    T += 0.5 * params.I4 * omega4**2

    O2 = np.array([0.0, 0.0])
    O4 = np.array([params.L1, 0.0])

    A = O2 + params.L2 * np.array([np.cos(theta2), np.sin(theta2)])
    G3 = A + 0.5 * params.L3 * np.array([np.cos(theta3), np.sin(theta3)])

    v_G3 = np.array([
        -params.L2 * np.sin(theta2) * omega2 - 0.5 * params.L3 * np.sin(theta3) * omega3,
        params.L2 * np.cos(theta2) * omega2 + 0.5 * params.L3 * np.cos(theta3) * omega3
    ])

    T += 0.5 * params.m3 * np.dot(v_G3, v_G3)

    G2 = O2 + 0.5 * params.L2 * np.array([np.cos(theta2), np.sin(theta2)])
    v_G2 = 0.5 * params.L2 * np.array([
        -np.sin(theta2) * omega2,
        np.cos(theta2) * omega2
    ])
    T += 0.5 * params.m2 * np.dot(v_G2, v_G2)

    G4 = O4 + 0.5 * params.L4 * np.array([np.cos(theta4), np.sin(theta4)])
    v_G4 = 0.5 * params.L4 * np.array([
        -np.sin(theta4) * omega4,
        np.cos(theta4) * omega4
    ])
    T += 0.5 * params.m4 * np.dot(v_G4, v_G4)

    return T


def potential_energy(
    theta2: float, theta3: float, theta4: float,
    params: FourBarParams
) -> float:
    O2 = np.array([0.0, 0.0])
    O4 = np.array([params.L1, 0.0])

    G2 = O2 + 0.5 * params.L2 * np.array([np.cos(theta2), np.sin(theta2)])
    A = O2 + params.L2 * np.array([np.cos(theta2), np.sin(theta2)])
    G3 = A + 0.5 * params.L3 * np.array([np.cos(theta3), np.sin(theta3)])
    G4 = O4 + 0.5 * params.L4 * np.array([np.cos(theta4), np.sin(theta4)])

    V = params.m2 * params.g * G2[1]
    V += params.m3 * params.g * G3[1]
    V += params.m4 * params.g * G4[1]

    return V


def lagrangian(
    theta2: float, theta3: float, theta4: float,
    omega2: float, omega3: float, omega4: float,
    params: FourBarParams
) -> float:
    return kinetic_energy(theta2, theta3, theta4, omega2, omega3, omega4, params) - \
           potential_energy(theta2, theta3, theta4, params)


def mass_matrix(theta2: float, theta3: float, theta4: float, params: FourBarParams) -> np.ndarray:
    L2, L3, L4 = params.L2, params.L3, params.L4
    m2, m3, m4 = params.m2, params.m3, params.m4
    I2, I3, I4 = params.I2, params.I3, params.I4

    J2 = 0.5 * L2 * np.array([-np.sin(theta2), np.cos(theta2)])
    J3_A = L2 * np.array([-np.sin(theta2), np.cos(theta2)])
    J3_G = J3_A + 0.5 * L3 * np.array([-np.sin(theta3), np.cos(theta3)])
    J4 = 0.5 * L4 * np.array([-np.sin(theta4), np.cos(theta4)])

    M = np.zeros((3, 3))

    M[0, 0] = I2 + m2 * np.dot(J2, J2)
    M[1, 1] = I3 + m3 * np.dot(J3_G, J3_G)
    M[2, 2] = I4 + m4 * np.dot(J4, J4)

    M[0, 1] = M[1, 0] = m3 * np.dot(J2, J3_G)
    M[0, 2] = M[2, 0] = 0
    M[1, 2] = M[2, 1] = 0

    return M


def coriolis_vector(
    theta2: float, theta3: float, theta4: float,
    omega2: float, omega3: float, omega4: float,
    params: FourBarParams
) -> np.ndarray:
    L2, L3 = params.L2, params.L3
    m3 = params.m3

    C = np.zeros(3)

    J2 = 0.5 * L2 * np.array([-np.sin(theta2), np.cos(theta2)])
    J3_A = L2 * np.array([-np.sin(theta2), np.cos(theta2)])
    J3_G = J3_A + 0.5 * L3 * np.array([-np.sin(theta3), np.cos(theta3)])

    dJ2_dt = 0.5 * L2 * omega2 * np.array([-np.cos(theta2), -np.sin(theta2)])
    dJ3_G_dt = L2 * omega2 * np.array([-np.cos(theta2), -np.sin(theta2)]) + \
               0.5 * L3 * omega3 * np.array([-np.cos(theta3), -np.sin(theta3)])

    C[0] = m3 * np.dot(J2, dJ3_G_dt) * omega3
    C[1] = m3 * np.dot(J3_G, dJ2_dt) * omega2
    C[2] = 0

    return C


def gravity_vector(theta2: float, theta3: float, theta4: float, params: FourBarParams) -> np.ndarray:
    G = np.zeros(3)

    G[0] = params.m2 * params.g * 0.5 * params.L2 * np.cos(theta2)
    G[0] += params.m3 * params.g * (params.L2 * np.cos(theta2) + 0.5 * params.L3 * np.cos(theta3))
    G[1] = params.m3 * params.g * 0.5 * params.L3 * np.cos(theta3)
    G[2] = params.m4 * params.g * 0.5 * params.L4 * np.cos(theta4)

    return G


def generalized_forces(
    tau_input: float,
    omega2: float, omega3: float, omega4: float,
    params: FourBarParams,
    coulomb_friction: np.ndarray,
    viscous_friction: np.ndarray
) -> np.ndarray:
    Q = np.zeros(3)
    Q[0] = tau_input
    Q[0] -= friction_torque(omega2, coulomb_friction[0], viscous_friction[0])
    Q[1] -= friction_torque(omega3, coulomb_friction[1], viscous_friction[1])
    Q[2] -= friction_torque(omega4, coulomb_friction[2], viscous_friction[2])
    return Q


def forward_dynamics(
    theta2: float, theta3: float, theta4: float,
    omega2: float, omega3: float, omega4: float,
    tau_input: float,
    params: FourBarParams,
    coulomb_friction: Optional[np.ndarray] = None,
    viscous_friction: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    if coulomb_friction is None:
        coulomb_friction = np.zeros(3)
    if viscous_friction is None:
        viscous_friction = np.zeros(3)

    M = mass_matrix(theta2, theta3, theta4, params)
    C = coriolis_vector(theta2, theta3, theta4, omega2, omega3, omega4, params)
    G = gravity_vector(theta2, theta3, theta4, params)
    Q = generalized_forces(tau_input, omega2, omega3, omega4, params, coulomb_friction, viscous_friction)

    try:
        alpha = np.linalg.solve(M, Q - C - G)
    except np.linalg.LinAlgError:
        raise ValueError("Dynamics solve failed: singular mass matrix")

    return alpha[0], alpha[1], alpha[2]


def inverse_dynamics(
    theta2: float, theta3: float, theta4: float,
    omega2: float, omega3: float, omega4: float,
    alpha2: float, alpha3: float, alpha4: float,
    params: FourBarParams,
    coulomb_friction: Optional[np.ndarray] = None,
    viscous_friction: Optional[np.ndarray] = None
) -> float:
    if coulomb_friction is None:
        coulomb_friction = np.zeros(3)
    if viscous_friction is None:
        viscous_friction = np.zeros(3)

    M = mass_matrix(theta2, theta3, theta4, params)
    C = coriolis_vector(theta2, theta3, theta4, omega2, omega3, omega4, params)
    G = gravity_vector(theta2, theta3, theta4, params)

    alpha = np.array([alpha2, alpha3, alpha4])
    Q = M @ alpha + C + G

    tau_input = Q[0] + friction_torque(omega2, coulomb_friction[0], viscous_friction[0])

    return tau_input


def compute_joint_forces(
    theta2: float, theta3: float, theta4: float,
    omega2: float, omega3: float, omega4: float,
    alpha2: float, alpha3: float, alpha4: float,
    tau_input: float,
    params: FourBarParams
) -> dict:
    M = mass_matrix(theta2, theta3, theta4, params)
    C = coriolis_vector(theta2, theta3, theta4, omega2, omega3, omega4, params)
    G = gravity_vector(theta2, theta3, theta4, params)

    alpha = np.array([alpha2, alpha3, alpha4])
    Q = M @ alpha + C + G

    return {
        'tau_input': Q[0],
        'Q2': Q[1],
        'Q3': Q[2]
    }