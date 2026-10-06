import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import joblib
import torch
from pathlib import Path
import sys
import os
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from physics import FourBarParams, load_params_from_config, DEFAULT_FRICTION
from physics.kinematics import solve_position, calculate_velocity, calculate_acceleration, calculate_joint_positions
from physics.dynamics import inverse_dynamics
from physics.faults import FaultConfig, DEFAULT_FAULT_CONFIGS, FAULT_CLASSES
from simulation import FourBarSimulator, SensorSuite, DEFAULT_SENSOR_CONFIGS
from inference import create_predictors, create_diagnosis_pipeline


HF_REPO = os.environ.get("HF_MODEL_REPO", "1The-Boss/mechanical-digital-twin-models")
MODEL_FILES = [
    "residual_model.pt",
    "residual_input_scaler.pkl",
    "residual_output_scaler.pkl",
    "autoencoder.pt",
    "autoencoder_scaler.pkl",
    "anomaly_threshold.json",
    "fault_classifier.pkl",
    "classifier_scaler.pkl",
    "training_summary.json",
]


def download_models_from_hf(target_dir: str = "models_saved") -> bool:
    """Download model files from Hugging Face Hub if not present locally."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    
    missing = [f for f in MODEL_FILES if not (target / f).exists()]
    if not missing:
        return True
    
    st.info(f"Downloading {len(missing)} model files from Hugging Face Hub...")
    progress = st.progress(0)
    
    base_url = f"https://huggingface.co/{HF_REPO}/resolve/main"
    success = True
    
    for i, fname in enumerate(missing):
        url = f"{base_url}/{fname}"
        try:
            resp = requests.get(url, stream=True, timeout=60)
            if resp.status_code == 200:
                total = int(resp.headers.get("content-length", 0))
                with open(target / fname, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, leave=False, desc=fname
                ) as pbar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                st.error(f"Failed to download {fname}: HTTP {resp.status_code}")
                success = False
        except Exception as e:
            st.error(f"Error downloading {fname}: {e}")
            success = False
        progress.progress((i + 1) / len(missing))
    
    progress.empty()
    return success


@st.cache_resource
def ensure_models(device="cpu"):
    """Ensure models exist locally, downloading from HF if needed."""
    models_dir = Path("models_saved")
    if not all((models_dir / f).exists() for f in MODEL_FILES):
        with st.spinner("Models not found locally. Downloading from Hugging Face Hub..."):
            if not download_models_from_hf("models_saved"):
                st.error("Failed to download models. Check HF_MODEL_REPO env var or internet connection.")
                return None, None, False
    return load_models(device)


st.set_page_config(
    page_title="Physics-Informed Hybrid Digital Twin",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_mechanism_params():
    return load_params_from_config('config.yaml')


@st.cache_resource
def load_models(device='cpu'):
    try:
        predictors = create_predictors('config.yaml', 'models_saved', device)
        diagnosis = create_diagnosis_pipeline('config.yaml', 'models_saved', device)
        models_loaded = True
    except Exception as e:
        st.warning(f"Models not found: {e}. Train models first.")
        predictors = None
        diagnosis = None
        models_loaded = False
    return predictors, diagnosis, models_loaded


@st.cache_resource
def ensure_models(device="cpu"):
    """Ensure models exist locally, downloading from HF if needed."""
    models_dir = Path("models_saved")
    if not all((models_dir / f).exists() for f in MODEL_FILES):
        with st.spinner("Models not found locally. Downloading from Hugging Face Hub..."):
            if not download_models_from_hf("models_saved"):
                st.error("Failed to download models. Check HF_MODEL_REPO env var or internet connection.")
                return None, None, False
    return load_models(device)


def run_simulation(params, fault_config, duration, timestep, omega2, initial_theta2):
    simulator = FourBarSimulator(params, DEFAULT_FRICTION, fault_config)
    result = simulator.simulate_constant_speed(duration, timestep, omega2, initial_theta2)
    return result


def apply_sensors(result):
    suite = SensorSuite(DEFAULT_SENSOR_CONFIGS)
    return suite.measure_all(result)


def compute_physics_predictions(result, params):
    n = len(result.time)
    physics = {
        'theta3': np.zeros(n),
        'theta4': np.zeros(n),
        'omega3': np.zeros(n),
        'omega4': np.zeros(n),
        'alpha3': np.zeros(n),
        'alpha4': np.zeros(n),
        'torque': np.zeros(n)
    }

    coulomb = DEFAULT_FRICTION.coulomb
    viscous = DEFAULT_FRICTION.viscous

    for i in range(n):
        try:
            theta3, theta4 = solve_position(result.theta2[i], params)
            omega3, omega4 = calculate_velocity(result.theta2[i], result.omega2[i], params)
            alpha3, alpha4 = calculate_acceleration(result.theta2[i], result.omega2[i], 0.0, params)

            torque = inverse_dynamics(
                result.theta2[i], theta3, theta4,
                result.omega2[i], omega3, omega4,
                0.0, alpha3, alpha4,
                params, coulomb, viscous
            )

            physics['theta3'][i] = theta3
            physics['theta4'][i] = theta4
            physics['omega3'][i] = omega3
            physics['omega4'][i] = omega4
            physics['alpha3'][i] = alpha3
            physics['alpha4'][i] = alpha4
            physics['torque'][i] = torque
        except ValueError:
            pass

    return physics


def compute_hybrid_predictions(result, predictors):
    n = len(result.time)
    hybrid = {
        'theta3': np.zeros(n),
        'theta4': np.zeros(n),
        'omega3': np.zeros(n),
        'omega4': np.zeros(n),
        'torque': np.zeros(n)
    }

    if predictors is None:
        return hybrid

    for i in range(n):
        try:
            pred = predictors['hybrid'].predict(result.theta2[i], result.omega2[i], 0.0)
            hybrid['theta3'][i] = pred['hybrid']['theta3']
            hybrid['theta4'][i] = pred['hybrid']['theta4']
            hybrid['omega3'][i] = pred['hybrid']['omega3']
            hybrid['omega4'][i] = pred['hybrid']['omega4']
            hybrid['torque'][i] = pred['hybrid']['torque']
        except Exception:
            pass

    return hybrid


def plot_mechanism(params, theta2, theta3, theta4):
    positions = calculate_joint_positions(theta2, theta3, theta4, params)
    O2, A, B, O4 = positions['O2'], positions['A'], positions['B'], positions['O4']

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=[O2[0], O4[0]], y=[O2[1], O4[1]],
                             mode='lines', line=dict(color='black', width=4),
                             name='Ground (L1)'))
    fig.add_trace(go.Scatter(x=[O2[0], A[0]], y=[O2[1], A[1]],
                             mode='lines', line=dict(color='blue', width=3),
                             name='Crank (L2)'))
    fig.add_trace(go.Scatter(x=[A[0], B[0]], y=[A[1], B[1]],
                             mode='lines', line=dict(color='green', width=3),
                             name='Coupler (L3)'))
    fig.add_trace(go.Scatter(x=[B[0], O4[0]], y=[B[1], O4[1]],
                             mode='lines', line=dict(color='red', width=3),
                             name='Rocker (L4)'))

    fig.add_trace(go.Scatter(x=[O2[0]], y=[O2[1]], mode='markers',
                             marker=dict(color='black', size=12), name='O2'))
    fig.add_trace(go.Scatter(x=[A[0]], y=[A[1]], mode='markers',
                             marker=dict(color='blue', size=10), name='A'))
    fig.add_trace(go.Scatter(x=[B[0]], y=[B[1]], mode='markers',
                             marker=dict(color='green', size=10), name='B'))
    fig.add_trace(go.Scatter(x=[O4[0]], y=[O4[1]], mode='markers',
                             marker=dict(color='red', size=12), name='O4'))

    fig.update_layout(
        title=f'Four-Bar Linkage (θ₂={np.degrees(theta2):.1f}°)',
        xaxis_title='X (m)', yaxis_title='Y (m)',
        width=500, height=500,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        showlegend=True
    )

    return fig


def main():
    st.title("⚙️ Physics-Informed Hybrid Digital Twin for Four-Bar Linkage")
    st.markdown("**Anomaly Detection & Fault Diagnosis**")

    config = load_config()
    params = load_mechanism_params()
    predictors, diagnosis, models_loaded = ensure_models()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔧 Mechanism Config", "▶️ Simulation", "📐 Visualization",
        "📊 Sensor Data", "🤖 Digital Twin", "🔍 Diagnosis"
    ])

    with tab1:
        st.header("Mechanism Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Link Lengths (m)")
            L1 = st.number_input("L1 (Ground)", value=params.L1, step=0.01, format="%.3f")
            L2 = st.number_input("L2 (Crank)", value=params.L2, step=0.01, format="%.3f")
            L3 = st.number_input("L3 (Coupler)", value=params.L3, step=0.01, format="%.3f")
            L4 = st.number_input("L4 (Rocker)", value=params.L4, step=0.01, format="%.3f")

        with col2:
            st.subheader("Mass Properties")
            m2 = st.number_input("m2 (kg)", value=params.m2, step=0.1, format="%.2f")
            m3 = st.number_input("m3 (kg)", value=params.m3, step=0.1, format="%.2f")
            m4 = st.number_input("m4 (kg)", value=params.m4, step=0.1, format="%.2f")
            I2 = st.number_input("I2 (kg·m²)", value=params.I2, step=0.0001, format="%.5f")
            I3 = st.number_input("I3 (kg·m²)", value=params.I3, step=0.0001, format="%.5f")
            I4 = st.number_input("I4 (kg·m²)", value=params.I4, step=0.0001, format="%.5f")

        if st.button("Update Parameters"):
            params = FourBarParams(L1=L1, L2=L2, L3=L3, L4=L4,
                                   m2=m2, m3=m3, m4=m4,
                                   I2=I2, I3=I3, I4=I4)
            st.success("Parameters updated!")
            st.rerun()

        valid, msg = params.is_grashof(), "Valid" if params.is_grashof() else "Invalid"
        st.info(f"Grashof Condition: {msg}")

    with tab2:
        st.header("Simulation Control")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Operating Conditions")
            duration = st.number_input("Duration (s)", value=5.0, min_value=0.1, max_value=30.0, step=0.5)
            timestep = st.number_input("Timestep (s)", value=0.001, min_value=0.0001, max_value=0.01, format="%.4f")
            omega2 = st.number_input("Input Speed ω₂ (rad/s)", value=10.0, min_value=0.1, max_value=50.0, step=0.5)
            initial_theta2 = st.number_input("Initial θ₂ (rad)", value=0.0, min_value=0.0, max_value=2*np.pi, format="%.3f")

        with col2:
            st.subheader("Fault Selection")
            fault_options = ["Normal"] + [f.fault_type for f in DEFAULT_FAULT_CONFIGS[1:]]
            selected_fault = st.selectbox("Fault Type", fault_options)

            if selected_fault != "Normal":
                fault_config = next(f for f in DEFAULT_FAULT_CONFIGS if f.fault_type == selected_fault)
                severity = st.slider("Severity", 1.0, 10.0, fault_config.severity, 0.1)
                fault_config = FaultConfig(selected_fault, severity, fault_config.affected_joint)
            else:
                fault_config = FaultConfig("normal", 1.0)
                severity = 1.0

            st.info(f"Fault: {fault_config.fault_type}, Severity: {fault_config.severity}x")

        if st.button("🚀 Run Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                result = run_simulation(params, fault_config, duration, timestep, omega2, initial_theta2)
                measurements = apply_sensors(result)
                physics = compute_physics_predictions(result, params)
                hybrid = compute_hybrid_predictions(result, predictors) if models_loaded else {}

                st.session_state['result'] = result
                st.session_state['measurements'] = measurements
                st.session_state['physics'] = physics
                st.session_state['hybrid'] = hybrid
                st.session_state['fault_config'] = fault_config

            st.success("Simulation complete!")

    with tab3:
        st.header("Mechanism Visualization")

        if 'result' in st.session_state:
            result = st.session_state['result']
            fault_config = st.session_state['fault_config']

            frame = st.slider("Time Frame", 0, len(result.time)-1, len(result.time)//4)

            theta2 = result.theta2[frame]
            theta3 = result.theta3[frame]
            theta4 = result.theta4[frame]

            col1, col2 = st.columns([1, 1])

            with col1:
                fig = plot_mechanism(params, theta2, theta3, theta4)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Current State")
                st.metric("θ₂ (Input)", f"{np.degrees(theta2):.1f}°")
                st.metric("θ₃ (Coupler)", f"{np.degrees(theta3):.1f}°")
                st.metric("θ₄ (Rocker)", f"{np.degrees(theta4):.1f}°")
                st.metric("ω₂", f"{result.omega2[frame]:.2f} rad/s")
                st.metric("Torque", f"{result.input_torque[frame]:.2f} Nm")
                st.metric("Fault", f"{fault_config.fault_type} ({fault_config.severity}x)")

            if st.button("🎬 Animate"):
                frames = []
                for i in range(0, len(result.time), max(1, len(result.time)//50)):
                    positions = calculate_joint_positions(result.theta2[i], result.theta3[i], result.theta4[i], params)
                    frames.append({
                        'O2': positions['O2'], 'A': positions['A'],
                        'B': positions['B'], 'O4': positions['O4']
                    })

                anim_fig = go.Figure(frames=[
                    go.Frame(data=[
                        go.Scatter(x=[f['O2'][0], f['O4'][0]], y=[f['O2'][1], f['O4'][1]], line=dict(color='black', width=4)),
                        go.Scatter(x=[f['O2'][0], f['A'][0]], y=[f['O2'][1], f['A'][1]], line=dict(color='blue', width=3)),
                        go.Scatter(x=[f['A'][0], f['B'][0]], y=[f['A'][1], f['B'][1]], line=dict(color='green', width=3)),
                        go.Scatter(x=[f['B'][0], f['O4'][0]], y=[f['B'][1], f['O4'][1]], line=dict(color='red', width=3)),
                    ]) for f in frames
                ])

                initial = frames[0]
                anim_fig.add_trace(go.Scatter(x=[initial['O2'][0], initial['O4'][0]], y=[initial['O2'][1], initial['O4'][1]], line=dict(color='black', width=4)))
                anim_fig.add_trace(go.Scatter(x=[initial['O2'][0], initial['A'][0]], y=[initial['O2'][1], initial['A'][1]], line=dict(color='blue', width=3)))
                anim_fig.add_trace(go.Scatter(x=[initial['A'][0], initial['B'][0]], y=[initial['A'][1], initial['B'][1]], line=dict(color='green', width=3)))
                anim_fig.add_trace(go.Scatter(x=[initial['B'][0], initial['O4'][0]], y=[initial['B'][1], initial['O4'][1]], line=dict(color='red', width=3)))

                anim_fig.update_layout(
                    xaxis=dict(range=[-0.1, 0.3], scaleanchor="y"),
                    yaxis=dict(range=[-0.2, 0.2]),
                    width=600, height=600,
                    updatemenus=[dict(type="buttons", buttons=[
                        dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50}}]),
                        dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}}])
                    ])]
                )
                st.plotly_chart(anim_fig, use_container_width=True)
        else:
            st.info("Run a simulation first to see visualization.")

    with tab4:
        st.header("Sensor Data")

        if 'result' in st.session_state:
            result = st.session_state['result']
            measurements = st.session_state['measurements']

            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                subplot_titles=('Angle θ₂', 'Velocity ω₂', 'Acceleration α₂', 'Torque'))

            fig.add_trace(go.Scatter(x=result.time, y=np.degrees(result.theta2), name='True θ₂', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=result.time, y=np.degrees(measurements['theta2_measured']), name='Measured θ₂', line=dict(color='lightblue', dash='dot')), row=1, col=1)

            fig.add_trace(go.Scatter(x=result.time, y=result.omega2, name='True ω₂', line=dict(color='green')), row=2, col=1)
            fig.add_trace(go.Scatter(x=result.time, y=measurements['omega2_measured'], name='Measured ω₂', line=dict(color='lightgreen', dash='dot')), row=2, col=1)

            fig.add_trace(go.Scatter(x=result.time, y=result.alpha2, name='True α₂', line=dict(color='red')), row=3, col=1)
            fig.add_trace(go.Scatter(x=result.time, y=measurements['alpha2_measured'], name='Measured α₂', line=dict(color='salmon', dash='dot')), row=3, col=1)

            fig.add_trace(go.Scatter(x=result.time, y=result.input_torque, name='True Torque', line=dict(color='black')), row=4, col=1)
            fig.add_trace(go.Scatter(x=result.time, y=measurements['torque_measured'], name='Measured Torque', line=dict(color='gray', dash='dot')), row=4, col=1)

            fig.update_layout(height=800, title_text="Sensor Measurements vs True Values")
            fig.update_xaxes(title_text="Time (s)", row=4, col=1)
            st.plotly_chart(fig, use_container_width=True)

            if st.checkbox("Show Statistics"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("θ₂ Mean", f"{np.mean(result.theta2):.4f} rad")
                    st.metric("θ₂ Std", f"{np.std(result.theta2):.4f} rad")
                with col2:
                    st.metric("ω₂ Mean", f"{np.mean(result.omega2):.2f} rad/s")
                    st.metric("ω₂ Std", f"{np.std(result.omega2):.2f} rad/s")
                with col3:
                    st.metric("Torque Mean", f"{np.mean(result.input_torque):.2f} Nm")
                    st.metric("Torque Std", f"{np.std(result.input_torque):.2f} Nm")
                with col4:
                    st.metric("α₂ Max", f"{np.max(np.abs(result.alpha2)):.2f} rad/s²")
                    st.metric("α₂ RMS", f"{np.sqrt(np.mean(result.alpha2**2)):.2f} rad/s²")
        else:
            st.info("Run a simulation first to see sensor data.")

    with tab5:
        st.header("Digital Twin Predictions")

        if 'result' in st.session_state and models_loaded:
            result = st.session_state['result']
            physics = st.session_state['physics']
            hybrid = st.session_state['hybrid']

            variables = ['theta3', 'theta4', 'omega3', 'omega4', 'torque']
            var_labels = ['θ₃ (rad)', 'θ₄ (rad)', 'ω₃ (rad/s)', 'ω₄ (rad/s)', 'Torque (Nm)']

            selected_var = st.selectbox("Select Variable", variables, format_func=lambda x: var_labels[variables.index(x)])

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=(f'{selected_var}: Actual vs Predictions', 'Residuals'))

            actual_key = selected_var if selected_var != 'torque' else 'input_torque'
            physics_key = selected_var if selected_var != 'torque' else 'torque'
            hybrid_key = f'hybrid_{selected_var}' if selected_var != 'torque' else 'torque'

            actual_values = getattr(result, actual_key)

            fig.add_trace(go.Scatter(x=result.time, y=actual_values, name='Actual', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=result.time, y=physics[physics_key], name='Physics', line=dict(color='red', dash='dash')), row=1, col=1)

            if hybrid_key in hybrid and np.any(hybrid[hybrid_key] != 0):
                fig.add_trace(go.Scatter(x=result.time, y=hybrid[hybrid_key], name='Hybrid', line=dict(color='green', dash='dot')), row=1, col=1)

            residual_physics = actual_values - physics[physics_key]
            fig.add_trace(go.Scatter(x=result.time, y=residual_physics, name='Physics Residual', line=dict(color='red')), row=2, col=1)

            if hybrid_key in hybrid and np.any(hybrid[hybrid_key] != 0):
                residual_hybrid = actual_values - hybrid[hybrid_key]
                fig.add_trace(go.Scatter(x=result.time, y=residual_hybrid, name='Hybrid Residual', line=dict(color='green')), row=2, col=1)

            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

            fig.update_layout(height=600)
            fig.update_xaxes(title_text="Time (s)", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

            if st.checkbox("Show Error Metrics"):
                rmse_physics = np.sqrt(np.mean(residual_physics**2))
                mae_physics = np.mean(np.abs(residual_physics))

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Physics RMSE", f"{rmse_physics:.6f}")
                    st.metric("Physics MAE", f"{mae_physics:.6f}")

                if hybrid_key in hybrid and np.any(hybrid[hybrid_key] != 0):
                    residual_hybrid = actual_values - hybrid[hybrid_key]
                    rmse_hybrid = np.sqrt(np.mean(residual_hybrid**2))
                    mae_hybrid = np.mean(np.abs(residual_hybrid))
                    improvement = (rmse_physics - rmse_hybrid) / rmse_physics * 100

                    with col2:
                        st.metric("Hybrid RMSE", f"{rmse_hybrid:.6f}", delta=f"{improvement:.1f}%")
                        st.metric("Hybrid MAE", f"{mae_hybrid:.6f}", delta=f"{(mae_physics - mae_hybrid) / mae_physics * 100:.1f}%")
        elif 'result' in st.session_state and not models_loaded:
            st.warning("Models not trained. Please run training first.")
        else:
            st.info("Run a simulation first to see digital twin predictions.")

    with tab6:
        st.header("Anomaly Detection & Fault Diagnosis")

        if 'result' in st.session_state and models_loaded and diagnosis is not None:
            result = st.session_state['result']
            physics = st.session_state['physics']

            residuals = {
                'residual_theta2': result.theta2 - result.theta2,
                'residual_omega2': result.omega2 - result.omega2,
                'residual_alpha2': result.alpha2 - physics['alpha3'],
                'residual_torque': result.input_torque - physics['torque']
            }

            anomaly_result = diagnosis['anomaly_detector'].detect_anomalies(residuals)
            fault_result = diagnosis['fault_diagnosis'].diagnose(residuals)

            col1, col2, col3 = st.columns(3)

            with col1:
                status_color = "🔴" if anomaly_result['status'] == 'ANOMALY' else "🟢"
                st.metric("Anomaly Status", f"{status_color} {anomaly_result['status']}")
                st.metric("Anomaly Score", f"{anomaly_result['max_score']:.6f}")
                st.metric("Threshold", f"{anomaly_result['threshold']:.6f}")

            with col2:
                st.metric("Predicted Fault", fault_result['fault_type'])
                st.metric("Confidence", f"{fault_result['confidence']:.2%}")
                st.metric("Anomaly Detected", "Yes" if fault_result['anomaly_detected'] else "No")

            with col3:
                st.subheader("Fault Probabilities")
                for name, prob in fault_result['fault_probabilities'].items():
                    st.progress(prob, text=f"{name}: {prob:.1%}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(len(anomaly_result['anomaly_scores'])), y=anomaly_result['anomaly_scores'], name='Anomaly Score'))
            fig.add_hline(y=anomaly_result['threshold'], line_dash="dash", line_color="red", annotation_text="Threshold")
            fig.update_layout(title="Anomaly Scores Over Time", xaxis_title="Window Index", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)

            if fault_result['window_probabilities']:
                probs = np.array(fault_result['window_probabilities'])
                fig2 = go.Figure()
                n_classes = probs.shape[1]
                class_names = fault_result.get('class_names', [f'Class_{i}' for i in range(n_classes)])
                for i, name in enumerate(class_names):
                    fig2.add_trace(go.Scatter(x=np.arange(len(probs)), y=probs[:, i], name=name, stackgroup='one'))
                fig2.update_layout(title="Fault Class Probabilities Over Time", xaxis_title="Window Index", yaxis_title="Probability")
                st.plotly_chart(fig2, use_container_width=True)

        elif 'result' in st.session_state and not models_loaded:
            st.warning("Models not trained. Please run training first.")
        else:
            st.info("Run a simulation first to see diagnosis results.")

    with st.sidebar:
        st.header("System Status")

        if models_loaded:
            st.success("✅ Models Loaded")
        else:
            st.error("❌ Models Not Loaded")
            if st.button("📥 Download Models from HF Hub"):
                with st.spinner("Downloading models..."):
                    if download_models_from_hf("models_saved"):
                        st.success("Models downloaded! Reloading...")
                        st.rerun()
                    else:
                        st.error("Download failed. Check HF_MODEL_REPO env var.")

        st.divider()
        st.subheader("Quick Actions")

        if st.button("📊 Generate Training Data"):
            st.info("Run `python -m training.train_all` in terminal")

        if st.button("🏋️ Train Models"):
            st.info("Run `python -m training.train_all` in terminal")

        if st.button("📈 Evaluate"):
            st.info("Run `python -m evaluation.evaluate` in terminal")

        st.divider()
        st.caption("Physics-Informed Hybrid Digital Twin")
        st.caption("For Anomaly Detection & Fault Diagnosis")


if __name__ == '__main__':
    main()