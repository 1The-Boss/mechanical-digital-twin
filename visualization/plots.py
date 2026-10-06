import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, List, Tuple
from pathlib import Path


def plot_joint_angles(
    time: np.ndarray,
    theta2: np.ndarray,
    theta3: np.ndarray,
    theta4: np.ndarray,
    theta2_measured: Optional[np.ndarray] = None,
    title: str = "Joint Angles",
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time, np.degrees(theta2), 'b-', label='θ₂ (input)', linewidth=1)
    ax.plot(time, np.degrees(theta3), 'g-', label='θ₃ (coupler)', linewidth=1)
    ax.plot(time, np.degrees(theta4), 'r-', label='θ₄ (rocker)', linewidth=1)

    if theta2_measured is not None:
        ax.plot(time, np.degrees(theta2_measured), 'b.', markersize=1, alpha=0.5, label='θ₂ measured')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_angular_velocities(
    time: np.ndarray,
    omega2: np.ndarray,
    omega3: np.ndarray,
    omega4: np.ndarray,
    omega2_measured: Optional[np.ndarray] = None,
    title: str = "Angular Velocities",
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time, omega2, 'b-', label='ω₂', linewidth=1)
    ax.plot(time, omega3, 'g-', label='ω₃', linewidth=1)
    ax.plot(time, omega4, 'r-', label='ω₄', linewidth=1)

    if omega2_measured is not None:
        ax.plot(time, omega2_measured, 'b.', markersize=1, alpha=0.5, label='ω₂ measured')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_angular_accelerations(
    time: np.ndarray,
    alpha2: np.ndarray,
    alpha3: np.ndarray,
    alpha4: np.ndarray,
    alpha2_measured: Optional[np.ndarray] = None,
    title: str = "Angular Accelerations",
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time, alpha2, 'b-', label='α₂', linewidth=1)
    ax.plot(time, alpha3, 'g-', label='α₃', linewidth=1)
    ax.plot(time, alpha4, 'r-', label='α₄', linewidth=1)

    if alpha2_measured is not None:
        ax.plot(time, alpha2_measured, 'b.', markersize=1, alpha=0.5, label='α₂ measured')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Acceleration (rad/s²)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_input_torque(
    time: np.ndarray,
    torque: np.ndarray,
    torque_measured: Optional[np.ndarray] = None,
    title: str = "Input Torque",
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time, torque, 'k-', label='Input Torque', linewidth=1)

    if torque_measured is not None:
        ax.plot(time, torque_measured, 'k.', markersize=1, alpha=0.5, label='Measured')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_actual_vs_physics(
    time: np.ndarray,
    actual: np.ndarray,
    physics: np.ndarray,
    variable_name: str,
    unit: str,
    title: str = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    if title is None:
        title = f"{variable_name}: Actual vs Physics Prediction"

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(time, actual, 'b-', label='Actual', linewidth=1)
    axes[0].plot(time, physics, 'r--', label='Physics', linewidth=1)
    axes[0].set_ylabel(f'{variable_name} ({unit})')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    residual = actual - physics
    axes[1].plot(time, residual, 'g-', label='Residual', linewidth=1)
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel(f'Residual ({unit})')
    axes[1].set_title('Residual (Actual - Physics)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_actual_vs_hybrid(
    time: np.ndarray,
    actual: np.ndarray,
    physics: np.ndarray,
    hybrid: np.ndarray,
    variable_name: str,
    unit: str,
    title: str = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    if title is None:
        title = f"{variable_name}: Actual vs Physics vs Hybrid"

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(time, actual, 'b-', label='Actual', linewidth=1)
    axes[0].plot(time, physics, 'r--', label='Physics', linewidth=1)
    axes[0].plot(time, hybrid, 'g-.', label='Hybrid', linewidth=1)
    axes[0].set_ylabel(f'{variable_name} ({unit})')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    residual_physics = actual - physics
    residual_hybrid = actual - hybrid
    axes[1].plot(time, residual_physics, 'r--', label='Physics Residual', linewidth=1)
    axes[1].plot(time, residual_hybrid, 'g-', label='Hybrid Residual', linewidth=1)
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel(f'Residual ({unit})')
    axes[1].set_title('Residuals Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_residuals(
    time: np.ndarray,
    residuals: Dict[str, np.ndarray],
    title: str = "Residual Signals",
    save_path: Optional[str] = None
) -> plt.Figure:
    n_residuals = len(residuals)
    fig, axes = plt.subplots(n_residuals, 1, figsize=(12, 3*n_residuals), sharex=True)

    if n_residuals == 1:
        axes = [axes]

    for i, (name, residual) in enumerate(residuals.items()):
        axes[i].plot(time, residual, 'g-', linewidth=0.8)
        axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[i].set_ylabel(name)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f'{name} (RMS: {np.sqrt(np.mean(residual**2)):.4f})')

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_anomaly_scores(
    time: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
    anomalies: Optional[np.ndarray] = None,
    title: str = "Anomaly Detection",
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time[:len(anomaly_scores)], anomaly_scores, 'b-', label='Anomaly Score', linewidth=1)
    ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')

    if anomalies is not None:
        anomaly_times = time[:len(anomalies)][anomalies]
        anomaly_scores_at = anomaly_scores[anomalies]
        ax.scatter(anomaly_times, anomaly_scores_at, c='r', s=20, zorder=5, label='Anomaly')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Anomaly Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_fault_classification(
    time: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str],
    true_labels: Optional[np.ndarray] = None,
    title: str = "Fault Classification",
    save_path: Optional[str] = None
) -> plt.Figure:
    n_classes = len(class_names)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(time[:len(predictions)], predictions, 'k-', linewidth=1, label='Predicted')
    if true_labels is not None:
        axes[0].plot(time[:len(true_labels)], true_labels, 'r--', linewidth=1, label='True')
    axes[0].set_yticks(range(n_classes))
    axes[0].set_yticklabels(class_names)
    axes[0].set_ylabel('Fault Class')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for i, name in enumerate(class_names):
        axes[1].plot(time[:len(probabilities)], probabilities[:, i], label=name, linewidth=1)

    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Class Probabilities')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if 'train_losses' in history:
        axes[0].plot(history['train_losses'], label='Train Loss')
    if 'val_losses' in history:
        axes[0].plot(history['val_losses'], label='Val Loss')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    if 'train_losses' in history and 'val_losses' in history:
        axes[1].plot(history['train_losses'], label='Train')
        axes[1].plot(history['val_losses'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss (Linear Scale)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    title: str = "Model Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['mae', 'rmse', 'r2']
    metric_labels = ['MAE', 'RMSE', 'R²']

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        subset = comparison_df[comparison_df['metric'] == metric]
        if len(subset) > 0:
            x = np.arange(len(subset))
            width = 0.35

            axes[i].bar(x - width/2, subset['physics'], width, label='Physics', alpha=0.7)
            axes[i].bar(x + width/2, subset['hybrid'], width, label='Hybrid', alpha=0.7)

            axes[i].set_xticks(x)
            axes[i].set_xticklabels(subset['variable'], rotation=45)
            axes[i].set_ylabel(label)
            axes[i].set_title(f'{label} Comparison')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_interactive_dashboard(
    time: np.ndarray,
    data: Dict[str, np.ndarray],
    physics_predictions: Optional[Dict[str, np.ndarray]] = None,
    hybrid_predictions: Optional[Dict[str, np.ndarray]] = None,
    anomaly_scores: Optional[np.ndarray] = None,
    anomaly_threshold: Optional[float] = None,
    fault_predictions: Optional[np.ndarray] = None
) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Joint Angles', 'Angular Velocities',
            'Angular Accelerations', 'Input Torque',
            'Residuals', 'Anomaly Scores',
            'Fault Classification', 'Model Comparison'
        ),
        vertical_spacing=0.08
    )

    row = 1
    for name, values in data.items():
        if 'theta' in name:
            fig.add_trace(go.Scatter(x=time, y=np.degrees(values), name=name), row=row, col=1)
    if physics_predictions:
        for name, values in physics_predictions.items():
            if 'theta' in name:
                fig.add_trace(go.Scatter(x=time, y=np.degrees(values), name=f'{name} (physics)', line=dict(dash='dash')), row=row, col=1)

    row = 1
    for name, values in data.items():
        if 'omega' in name:
            fig.add_trace(go.Scatter(x=time, y=values, name=name), row=row, col=2)
    if physics_predictions:
        for name, values in physics_predictions.items():
            if 'omega' in name:
                fig.add_trace(go.Scatter(x=time, y=values, name=f'{name} (physics)', line=dict(dash='dash')), row=row, col=2)

    row = 2
    for name, values in data.items():
        if 'alpha' in name:
            fig.add_trace(go.Scatter(x=time, y=values, name=name), row=row, col=1)

    row = 2
    if 'input_torque' in data:
        fig.add_trace(go.Scatter(x=time, y=data['input_torque'], name='Input Torque'), row=row, col=2)

    row = 3
    if anomaly_scores is not None:
        fig.add_trace(go.Scatter(x=time[:len(anomaly_scores)], y=anomaly_scores, name='Anomaly Score'), row=row, col=2)
        if anomaly_threshold:
            fig.add_hline(y=anomaly_threshold, line_dash="dash", line_color="red", row=row, col=2)

    fig.update_layout(height=1000, title_text="Digital Twin Dashboard", showlegend=True)
    return fig


def save_all_plots(
    time: np.ndarray,
    data: Dict[str, np.ndarray],
    physics_predictions: Dict[str, np.ndarray],
    hybrid_predictions: Dict[str, np.ndarray],
    anomaly_scores: np.ndarray,
    anomaly_threshold: float,
    fault_results: Dict,
    output_dir: str = 'plots'
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_joint_angles(time, data['theta2'], data['theta3'], data['theta4'],
                      data.get('theta2_measured'), save_path=f'{output_dir}/joint_angles.png')
    plot_angular_velocities(time, data['omega2'], data['omega3'], data['omega4'],
                            data.get('omega2_measured'), save_path=f'{output_dir}/angular_velocities.png')
    plot_angular_accelerations(time, data['alpha2'], data['alpha3'], data['alpha4'],
                               data.get('alpha2_measured'), save_path=f'{output_dir}/angular_accelerations.png')
    plot_input_torque(time, data['input_torque'], data.get('torque_measured'),
                      save_path=f'{output_dir}/input_torque.png')

    for var in ['theta3', 'theta4', 'omega3', 'omega4', 'alpha3', 'alpha4', 'torque']:
        actual_key = var if var != 'torque' else 'input_torque'
        if actual_key in data and f'physics_{var}' in physics_predictions:
            plot_actual_vs_physics(time, data[actual_key], physics_predictions[f'physics_{var}'],
                                   var, 'rad' if 'alpha' not in var else 'rad/s²',
                                   save_path=f'{output_dir}/physics_vs_actual_{var}.png')

    for var in ['theta3', 'theta4', 'omega3', 'omega4', 'alpha3', 'alpha4', 'torque']:
        actual_key = var if var != 'torque' else 'input_torque'
        hybrid_key = f'hybrid_{var}' if var != 'torque' else 'hybrid_torque'
        if actual_key in data and f'physics_{var}' in physics_predictions and hybrid_key in hybrid_predictions:
            plot_actual_vs_hybrid(time, data[actual_key], physics_predictions[f'physics_{var}'],
                                  hybrid_predictions[hybrid_key], var,
                                  'rad' if 'alpha' not in var else 'rad/s²',
                                  save_path=f'{output_dir}/hybrid_vs_actual_{var}.png')

    residuals = {k: v for k, v in data.items() if k.startswith('residual_')}
    if residuals:
        plot_residuals(time, residuals, save_path=f'{output_dir}/residuals.png')

    if len(anomaly_scores) > 0:
        plot_anomaly_scores(time[:len(anomaly_scores)], anomaly_scores, anomaly_threshold,
                            fault_results.get('anomalies'),
                            save_path=f'{output_dir}/anomaly_scores.png')

    if 'window_predictions' in fault_results:
        plot_fault_classification(
            time[:len(fault_results['window_predictions'])],
            fault_results['window_predictions'],
            np.array(fault_results['window_probabilities']),
            fault_results.get('class_names', ['NORMAL', 'FRICTION', 'MASS', 'JOINT', 'INERTIA']),
            save_path=f'{output_dir}/fault_classification.png'
        )

    if 'confusion_matrix' in fault_results:
        plot_confusion_matrix(
            np.array(fault_results['confusion_matrix']),
            fault_results.get('class_names', ['NORMAL', 'FRICTION', 'MASS', 'JOINT', 'INERTIA']),
            save_path=f'{output_dir}/confusion_matrix.png'
        )

    print(f"All plots saved to {output_dir}/")