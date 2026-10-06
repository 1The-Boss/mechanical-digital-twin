import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from typing import Optional, Dict, List, Tuple
from physics import FourBarParams
from physics.kinematics import calculate_joint_positions, solve_position


def plot_mechanism_geometry(
    params: FourBarParams,
    theta2: float,
    theta3: float,
    theta4: float,
    ax: Optional[plt.Axes] = None,
    show_ground: bool = True,
    show_joints: bool = True,
    show_coupler_point: bool = False,
    coupler_ratio: float = 0.5
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    positions = calculate_joint_positions(theta2, theta3, theta4, params)
    O2, A, B, O4 = positions['O2'], positions['A'], positions['B'], positions['O4']

    if show_ground:
        ax.plot([O2[0], O4[0]], [O2[1], O4[1]], 'k-', linewidth=3, label='Ground (L1)')

    ax.plot([O2[0], A[0]], [O2[1], A[1]], 'b-', linewidth=2, label='Crank (L2)')
    ax.plot([A[0], B[0]], [A[1], B[1]], 'g-', linewidth=2, label='Coupler (L3)')
    ax.plot([B[0], O4[0]], [B[1], O4[1]], 'r-', linewidth=2, label='Rocker (L4)')

    if show_joints:
        ax.plot(O2[0], O2[1], 'ko', markersize=10, label='Joint O2')
        ax.plot(A[0], A[1], 'bo', markersize=8, label='Joint A')
        ax.plot(B[0], B[1], 'go', markersize=8, label='Joint B')
        ax.plot(O4[0], O4[1], 'ro', markersize=10, label='Joint O4')

    if show_coupler_point:
        P = A + coupler_ratio * (B - A)
        ax.plot(P[0], P[1], 'mo', markersize=6, label='Coupler Point')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Four-Bar Linkage\nθ₂={np.degrees(theta2):.1f}°, θ₃={np.degrees(theta3):.1f}°, θ₄={np.degrees(theta4):.1f}°')

    margin = 0.05
    x_min = min(O2[0], O4[0], A[0], B[0]) - margin
    x_max = max(O2[0], O4[0], A[0], B[0]) + margin
    y_min = min(O2[1], O4[1], A[1], B[1]) - margin
    y_max = max(O2[1], O4[1], A[1], B[1]) + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    return ax


def animate_mechanism(
    params: FourBarParams,
    theta2_array: np.ndarray,
    theta3_array: np.ndarray,
    theta4_array: np.ndarray,
    interval: int = 50,
    save_path: Optional[str] = None
) -> animation.FuncAnimation:
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        ax.clear()
        plot_mechanism_geometry(params, theta2_array[frame], theta3_array[frame], theta4_array[frame], ax)
        ax.set_title(f'Four-Bar Linkage Animation - Frame {frame}/{len(theta2_array)-1}')

    ani = animation.FuncAnimation(fig, update, frames=len(theta2_array), interval=interval, repeat=True)

    if save_path:
        ani.save(save_path, writer='pillow', fps=1000//interval)

    return ani


def plot_mechanism_trajectory(
    params: FourBarParams,
    theta2_array: np.ndarray,
    theta3_array: np.ndarray,
    theta4_array: np.ndarray,
    n_points: int = 200,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    indices = np.linspace(0, len(theta2_array) - 1, min(n_points, len(theta2_array)), dtype=int)

    for idx in indices:
        positions = calculate_joint_positions(
            theta2_array[idx], theta3_array[idx], theta4_array[idx], params
        )
        A, B = positions['A'], positions['B']
        ax.plot(A[0], A[1], 'b.', alpha=0.3, markersize=2)
        ax.plot(B[0], B[1], 'g.', alpha=0.3, markersize=2)

    plot_mechanism_geometry(
        params, theta2_array[0], theta3_array[0], theta4_array[0], ax,
        show_ground=True, show_joints=False, show_coupler_point=False
    )
    ax.set_title('Coupler Curve Trajectory')

    return ax


def plot_workspace(
    params: FourBarParams,
    n_samples: int = 1000,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    # solve_position already imported at module level

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    theta2_vals = np.linspace(0, 2*np.pi, n_samples)
    A_points = []
    B_points = []

    for theta2 in theta2_vals:
        try:
            theta3, theta4 = solve_position(theta2, params)
            positions = calculate_joint_positions(theta2, theta3, theta4, params)
            A_points.append(positions['A'])
            B_points.append(positions['B'])
        except ValueError:
            pass

    if A_points:
        A_points = np.array(A_points)
        B_points = np.array(B_points)
        ax.scatter(A_points[:, 0], A_points[:, 1], c='blue', s=1, alpha=0.5, label='Joint A')
        ax.scatter(B_points[:, 0], B_points[:, 1], c='green', s=1, alpha=0.5, label='Joint B')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Four-Bar Linkage Workspace')

    return ax