import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any

from models import ResidualMLP, count_parameters, model_summary
from preprocessing import FeatureScaler, get_default_feature_columns, get_residual_feature_columns
from preprocessing.windowing import split_by_simulation


def prepare_residual_data(
    df: pd.DataFrame,
    physics_predictions: Dict[str, np.ndarray],
    feature_columns: list = None,
    target_columns: list = None
) -> Tuple[np.ndarray, np.ndarray, FeatureScaler, FeatureScaler]:
    if feature_columns is None:
        feature_columns = get_default_feature_columns()
    if target_columns is None:
        target_columns = ['residual_theta2', 'residual_omega2', 'residual_alpha2', 'residual_torque']

    for col in target_columns:
        if col not in df.columns:
            physics_key = col.replace('residual_', 'physics_')
            if physics_key in physics_predictions:
                df[col] = df[col.replace('residual_', '')] - physics_predictions[physics_key]
            else:
                df[col] = 0.0

    X = df[feature_columns].values
    y = df[target_columns].values

    input_scaler = FeatureScaler('standard')
    output_scaler = FeatureScaler('standard')

    X_scaled = input_scaler.fit_transform(X)
    y_scaled = output_scaler.fit_transform(y)

    return X_scaled, y_scaled, input_scaler, output_scaler


def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    simulation_ids: Optional[np.ndarray] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    from ..preprocessing.windowing import split_by_simulation

    if simulation_ids is not None:
        X_train, X_val, X_test, y_train, y_val, y_test = split_by_simulation(
            X, y, simulation_ids, 1 - val_split - test_split, val_split, random_seed
        )
    else:
        n = len(X)
        indices = np.arange(n)
        rng = np.random.default_rng(random_seed)
        rng.shuffle(indices)

        n_train = int(n * (1 - val_split - test_split))
        n_val = int(n * val_split)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_residual_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    early_stopping_patience: int = 15,
    model_save_path: str = 'models_saved/residual_model.pt',
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

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
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


def evaluate_residual_model(
    model: nn.Module,
    test_loader: DataLoader,
    output_scaler: FeatureScaler,
    device: str = 'cpu'
) -> Dict[str, float]:
    model.to(device)
    model.eval()
    criterion = nn.MSELoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            all_preds.append(y_pred)
            all_targets.append(y_batch.numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    preds_orig = output_scaler.inverse_transform(preds)
    targets_orig = output_scaler.inverse_transform(targets)

    mse = np.mean((preds_orig - targets_orig) ** 2)
    mae = np.mean(np.abs(preds_orig - targets_orig))
    rmse = np.sqrt(mse)

    r2_scores = []
    for i in range(targets_orig.shape[1]):
        ss_res = np.sum((targets_orig[:, i] - preds_orig[:, i]) ** 2)
        ss_tot = np.sum((targets_orig[:, i] - np.mean(targets_orig[:, i])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r2_scores.append(r2)

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2_scores': r2_scores,
        'mean_r2': np.mean(r2_scores)
    }


def run_residual_training(
    df: pd.DataFrame,
    physics_predictions: Dict[str, np.ndarray],
    config: Dict[str, Any],
    model_save_path: str = 'models_saved/residual_model.pt',
    scaler_save_dir: str = 'models_saved'
) -> Tuple[nn.Module, FeatureScaler, FeatureScaler, Dict]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X_scaled, y_scaled, input_scaler, output_scaler = prepare_residual_data(df, physics_predictions)

    simulation_ids = df['simulation_id'].values if 'simulation_id' in df.columns else None

    train_loader, val_loader, test_loader = create_dataloaders(
        X_scaled, y_scaled,
        batch_size=config.get('batch_size', 64),
        val_split=config.get('validation_split', 0.15),
        test_split=config.get('test_split', 0.15),
        random_seed=config.get('random_seed', 42),
        simulation_ids=simulation_ids
    )

    model = ResidualMLP(
        input_dim=X_scaled.shape[1],
        output_dim=y_scaled.shape[1],
        hidden_dims=config.get('hidden_dims', [128, 64, 32]),
        dropout=config.get('dropout', 0.1)
    )

    print(model_summary(model))

    history = train_residual_model(
        model, train_loader, val_loader,
        epochs=config.get('epochs', 100),
        learning_rate=config.get('learning_rate', 0.001),
        device=device,
        early_stopping_patience=config.get('early_stopping_patience', 15),
        model_save_path=model_save_path
    )

    input_scaler.save(f"{scaler_save_dir}/residual_input_scaler.pkl")
    output_scaler.save(f"{scaler_save_dir}/residual_output_scaler.pkl")

    test_metrics = evaluate_residual_model(model, test_loader, output_scaler, device)
    print(f"Test Metrics: MSE={test_metrics['mse']:.6f}, MAE={test_metrics['mae']:.6f}, RMSE={test_metrics['rmse']:.6f}, R2={test_metrics['mean_r2']:.4f}")

    return model, input_scaler, output_scaler, {**history, **test_metrics}