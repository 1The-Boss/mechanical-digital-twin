import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any, List

from models import Autoencoder, compute_reconstruction_error, compute_anomaly_scores
from preprocessing import FeatureScaler
from preprocessing.windowing import create_windows, extract_statistical_features, split_by_simulation


def prepare_autoencoder_data(
    df: pd.DataFrame,
    physics_predictions: Dict[str, np.ndarray],
    feature_columns: Optional[List[str]] = None,
    window_size: int = 50,
    stride: int = 10
) -> Tuple[np.ndarray, FeatureScaler, np.ndarray, Optional[np.ndarray]]:
    if feature_columns is None:
        feature_columns = [
            'residual_theta2', 'residual_omega2', 'residual_alpha2', 'residual_torque'
        ]

    for col in feature_columns:
        if col not in df.columns:
            physics_key = col.replace('residual_', 'physics_')
            if physics_key in physics_predictions:
                df[col] = df[col.replace('residual_', '')] - physics_predictions[physics_key]
            else:
                df[col] = 0.0

    from ..preprocessing.windowing import create_windows, extract_statistical_features

    data = df[feature_columns].values
    windows = create_windows(data, window_size, stride)

    stat_features, stat_names = extract_statistical_features(windows, feature_columns)

    simulation_ids = None
    fault_labels_windowed = None
    if 'simulation_id' in df.columns:
        sim_ids = df['simulation_id'].values
        sim_id_windows = create_windows(sim_ids.reshape(-1, 1), window_size, stride)
        simulation_ids = sim_id_windows[:, -1, 0].astype(int)
    
    if 'fault_class' in df.columns:
        fault_labels = df['fault_class'].values
        fault_windows = create_windows(fault_labels.reshape(-1, 1), window_size, stride)
        fault_labels_windowed = fault_windows[:, -1, 0].astype(int)

    scaler = FeatureScaler('standard')
    stat_features_scaled = scaler.fit_transform(stat_features)

    return stat_features_scaled, scaler, simulation_ids, fault_labels_windowed


def create_autoencoder_dataloaders(
    X: np.ndarray,
    batch_size: int = 64,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    simulation_ids: Optional[np.ndarray] = None,
    only_healthy: bool = False,
    fault_labels: Optional[np.ndarray] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    from ..preprocessing.windowing import split_by_simulation

    if only_healthy and fault_labels is not None:
        healthy_mask = fault_labels == 0
        X = X[healthy_mask]
        if simulation_ids is not None:
            simulation_ids = simulation_ids[healthy_mask]

    if simulation_ids is not None:
        X_train, X_val, X_test = split_by_simulation(
            X, None, simulation_ids, 1 - val_split - test_split, val_split, random_seed
        )[:3]
    else:
        n = len(X)
        indices = np.arange(n)
        rng = np.random.default_rng(random_seed)
        rng.shuffle(indices)

        n_train = int(n * (1 - val_split - test_split))
        n_val = max(1, int(n * val_split))

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]

    # Ensure we have at least 1 sample in each split
    if len(X_val) == 0:
        # Take 1 sample from training for validation
        X_val = X_train[:1]
        X_train = X_train[1:]
    if len(X_test) == 0:
        X_test = X_train[:1]
        X_train = X_train[1:]

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(X_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(X_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(X_test))

    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(X_train)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(X_val)), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=min(batch_size, len(X_test)), shuffle=False)

    return train_loader, val_loader, test_loader


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    early_stopping_patience: int = 15,
    model_save_path: str = 'models_saved/autoencoder.pt',
    scheduler_patience: int = 10
) -> Dict[str, list]:
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=scheduler_patience, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            X_batch = X_batch.to(device)

            optimizer.zero_grad()
            recon = model(X_batch)
            loss = criterion(recon, X_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                recon = model(X_batch)
                loss = criterion(recon, X_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1:3d}/{epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(model_save_path, map_location=device))

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }


def evaluate_autoencoder(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> Dict[str, float]:
    model.to(device)
    model.eval()
    criterion = nn.MSELoss()

    all_errors = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            recon = model(X_batch)
            error = torch.mean((X_batch - recon) ** 2, dim=1)
            all_errors.append(error.cpu().numpy())

    errors = np.concatenate(all_errors) if all_errors else np.array([])

    return {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'median_error': np.median(errors),
        'max_error': np.max(errors),
        'errors': errors
    }


def compute_anomaly_threshold(
    model: nn.Module,
    healthy_loader: DataLoader,
    percentile: float = 95,
    device: str = 'cpu'
) -> float:
    errors = compute_anomaly_scores(model, healthy_loader, device)
    threshold = np.percentile(errors, percentile)
    return float(threshold)


def run_autoencoder_training(
    df: pd.DataFrame,
    physics_predictions: Dict[str, np.ndarray],
    config: Dict[str, Any],
    model_save_path: str = 'models_saved/autoencoder.pt',
    scaler_save_path: str = 'models_saved/autoencoder_scaler.pkl',
    threshold_save_path: str = 'models_saved/anomaly_threshold.json'
) -> Tuple[nn.Module, FeatureScaler, float, Dict]:
    import json

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X_scaled, scaler, simulation_ids, fault_labels_windowed = prepare_autoencoder_data(
        df, physics_predictions,
        window_size=config.get('window_size', 50),
        stride=config.get('stride', 10)
    )

    train_loader, val_loader, test_loader = create_autoencoder_dataloaders(
        X_scaled,
        batch_size=config.get('batch_size', 64),
        val_split=config.get('validation_split', 0.15),
        test_split=config.get('test_split', 0.15),
        random_seed=config.get('random_seed', 42),
        simulation_ids=simulation_ids,
        only_healthy=True,
        fault_labels=fault_labels_windowed
    )

    healthy_loader = train_loader

    model = Autoencoder(
        input_dim=X_scaled.shape[1],
        hidden_dims=config.get('hidden_dims', [64, 32]),
        latent_dim=config.get('latent_dim', 8)
    )

    print(f"Autoencoder: {model}")

    history = train_autoencoder(
        model, train_loader, val_loader,
        epochs=config.get('epochs', 100),
        learning_rate=config.get('learning_rate', 0.001),
        device=device,
        early_stopping_patience=config.get('early_stopping_patience', 15),
        model_save_path=model_save_path
    )

    scaler.save(scaler_save_path)

    threshold = compute_anomaly_threshold(
        model, healthy_loader,
        percentile=config.get('threshold_percentile', 95),
        device=device
    )

    with open(threshold_save_path, 'w') as f:
        json.dump({'threshold': threshold, 'percentile': config.get('threshold_percentile', 95)}, f)

    test_metrics = evaluate_autoencoder(model, test_loader, device)
    print(f"Test Reconstruction Error: Mean={test_metrics['mean_error']:.6f}, Std={test_metrics['std_error']:.6f}")
    print(f"Anomaly Threshold (p={config.get('threshold_percentile', 95)}): {threshold:.6f}")

    return model, scaler, threshold, {**history, **test_metrics}