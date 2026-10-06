import numpy as np
from typing import Tuple, Optional
from .mechanism import FourBarParams


def solve_position(theta2: float, params: FourBarParams, branch: int = 1) -> Tuple[float, float]:
    L1, L2, L3, L4 = params.L1, params.L2, params.L3, params.L4

    # Use Newton-Raphson to solve the vector loop equations
    # L2*cos(theta2) + L3*cos(theta3) = L1 + L4*cos(theta4)
    # L2*sin(theta2) + L3*sin(theta3) = L4*sin(theta4)

    # Initial guess using law of cosines
    Ax = L2 * np.cos(theta2)
    Ay = L2 * np.sin(theta2)
    dx = L1 - Ax
    dy = -Ay
    d = np.sqrt(dx**2 + dy**2)

    # Check if mechanism can close
    if d > L3 + L4 or d < abs(L3 - L4):
        raise ValueError(f"Position analysis failed: distance d={d:.6f} not in valid range [{abs(L3-L4):.6f}, {L3+L4:.6f}]")

    # Law of cosines for triangle A-O4-B
    cos_phi = (d**2 + L4**2 - L3**2) / (2 * d * L4)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.arccos(cos_phi)

    gamma = np.arctan2(-dy, dx)

    theta4_1 = gamma + branch * phi
    theta4_2 = gamma - branch * phi

    # Use the first branch
    theta4 = theta4_1

    # Compute theta3 from geometry
    Bx = L1 + L4 * np.cos(theta4)
    By = L4 * np.sin(theta4)
    theta3 = np.arctan2(By - Ay, Bx - Ax)

    # Refine with Newton-Raphson to ensure vector loop closure
    theta3, theta4 = _newton_raphson_refine(theta2, theta3, theta4, params)

    return theta3, theta4


def _newton_raphson_refine(theta2: float, theta3: float, theta4: float, params: FourBarParams, max_iter: int = 10, tol: float = 1e-12) -> Tuple[float, float]:
    L1, L2, L3, L4 = params.L1, params.L2, params.L3, params.L4

    for _ in range(max_iter):
        f1 = L2 * np.cos(theta2) + L3 * np.cos(theta3) - L1 - L4 * np.cos(theta4)
        f2 = L2 * np.sin(theta2) + L3 * np.sin(theta3) - L4 * np.sin(theta4)

        error = np.sqrt(f1**2 + f2**2)
        if error < tol:
            break

        # Jacobian
        J = np.array([
            [-L3 * np.sin(theta3), L4 * np.sin(theta4)],
            [L3 * np.cos(theta3), -L4 * np.cos(theta4)]
        ])

        try:
            delta = np.linalg.solve(J, np.array([-f1, -f2]))
            theta3 += delta[0]
            theta4 += delta[1]
        except np.linalg.LinAlgError:
            break

    return theta3, theta4


def calculate_velocity(theta2: float, omega2: float, params: FourBarParams) -> Tuple[float, float]:
    theta3, theta4 = solve_position(theta2, params)

    L2, L3, L4 = params.L2, params.L3, params.L4

    a = np.array([
        [-L3 * np.sin(theta3), L4 * np.sin(theta4)],
        [L3 * np.cos(theta3), -L4 * np.cos(theta4)]
    ])
    b = np.array([
        L2 * np.sin(theta2) * omega2,
        -L2 * np.cos(theta2) * omega2
    ])

    try:
        omega3, omega4 = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        raise ValueError("Velocity analysis failed: singular Jacobian")

    return omega3, omega4


def calculate_acceleration(
    theta2: float,
    omega2: float,
    alpha2: float,
    params: FourBarParams
) -> Tuple[float, float]:
    theta3, theta4 = solve_position(theta2, params)
    omega3, omega4 = calculate_velocity(theta2, omega2, params)

    L2, L3, L4 = params.L2, params.L3, params.L4

    a = np.array([
        [-L3 * np.sin(theta3), L4 * np.sin(theta4)],
        [L3 * np.cos(theta3), -L4 * np.cos(theta4)]
    ])
    b = np.array([
        L2 * (np.cos(theta2) * omega2**2 - np.sin(theta2) * alpha2) +
        L3 * np.cos(theta3) * omega3**2 -
        L4 * np.cos(theta4) * omega4**2,
        L2 * (np.sin(theta2) * omega2**2 + np.cos(theta2) * alpha2) +
        L3 * np.sin(theta3) * omega3**2 -
        L4 * np.sin(theta4) * omega4**2
    ])

    try:
        alpha3, alpha4 = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        raise ValueError("Acceleration analysis failed: singular Jacobian")

    return alpha3, alpha4


def calculate_joint_positions(
    theta2: float,
    theta3: float,
    theta4: float,
    params: FourBarParams
) -> dict:
    L1, L2, L3, L4 = params.L1, params.L2, params.L3, params.L4

    O2 = np.array([0.0, 0.0])
    O4 = np.array([L1, 0.0])

    A = O2 + L2 * np.array([np.cos(theta2), np.sin(theta2)])
    B = O4 + L4 * np.array([np.cos(theta4), np.sin(theta4)])

    return {
        'O2': O2,
        'A': A,
        'B': B,
        'O4': O4
    }


def vector_loop_error(theta2: float, theta3: float, theta4: float, params: FourBarParams) -> float:
    L1, L2, L3, L4 = params.L1, params.L2, params.L3, params.L4

    x_error = L2 * np.cos(theta2) + L3 * np.cos(theta3) - L1 - L4 * np.cos(theta4)
    y_error = L2 * np.sin(theta2) + L3 * np.sin(theta3) - L4 * np.sin(theta4)

    return np.sqrt(x_error**2 + y_error**2)


def check_mechanism_validity(params: FourBarParams) -> Tuple[bool, str]:
    L1, L2, L3, L4 = params.L1, params.L2, params.L3, params.L4

    links = sorted([L1, L2, L3, L4])
    s, p, q, l = links

    if s + l <= p + q:
        return True, "Valid Grashof mechanism"

    return False, "Non-Grashof: No link can rotate fully"


def is_valid_configuration(theta2: float, params: FourBarParams) -> bool:
    try:
        solve_position(theta2, params)
        return True
    except ValueError:
        return False