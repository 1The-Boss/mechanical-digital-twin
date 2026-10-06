import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class WindowConfig:
    window_size: int = 50
    stride: int = 10
    min_periods: int = 1


def create_windows(
    data: np.ndarray,
    window_size: int,
    stride: int = 1,
    min_periods: int = 1
) -> np.ndarray:
    n_samples, n_features = data.shape
    n_windows = (n_samples - window_size) // stride + 1

    if n_windows <= 0:
        return np.empty((0, window_size, n_features))

    windows = np.lib.stride_tricks.sliding_window_view(data, window_size, axis=0)[::stride]

    if windows.shape[0] > n_windows:
        windows = windows[:n_windows]

    return windows


def create_windows_from_dataframe(
    df: pd.DataFrame,
    feature_columns: List[str],
    window_size: int,
    stride: int = 1,
    label_column: Optional[str] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    data = df[feature_columns].values
    windows = create_windows(data, window_size, stride)

    labels = None
    simulation_ids = None

    if label_column and label_column in df.columns:
        label_data = df[label_column].values
        label_windows = create_windows(label_data.reshape(-1, 1), window_size, stride)
        labels = label_windows[:, -1, 0].astype(int)

    if 'simulation_id' in df.columns:
        sim_id_data = df['simulation_id'].values
        sim_id_windows = create_windows(sim_id_data.reshape(-1, 1), window_size, stride)
        simulation_ids = sim_id_windows[:, -1, 0].astype(int)

    return windows, labels, simulation_ids


def compute_window_features(window: np.ndarray) -> np.ndarray:
    features = []

    features.append(np.mean(window, axis=0))
    features.append(np.std(window, axis=0))
    features.append(np.sqrt(np.mean(window**2, axis=0)))
    features.append(np.max(window, axis=0))
    features.append(np.min(window, axis=0))
    features.append(np.ptp(window, axis=0))

    return np.concatenate(features)


def extract_statistical_features(
    windows: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    n_windows, window_size, n_features = windows.shape

    stat_features = []
    stat_names = []

    for i in range(n_windows):
        window_feats = compute_window_features(windows[i])
        stat_features.append(window_feats)

    if feature_names is not None:
        for fn in feature_names:
            stat_names.extend([
                f'{fn}_mean', f'{fn}_std', f'{fn}_rms',
                f'{fn}_max', f'{fn}_min', f'{fn}_ptp'
            ])
    else:
        for j in range(n_features):
            stat_names.extend([
                f'feat{j}_mean', f'feat{j}_std', f'feat{j}_rms',
                f'feat{j}_max', f'feat{j}_min', f'feat{j}_ptp'
            ])

    return np.array(stat_features), stat_names


def extract_frequency_features(
    windows: np.ndarray,
    fs: float,
    n_fft: Optional[int] = None
) -> Tuple[np.ndarray, List[str]]:
    from scipy.fft import rfft, rfftfreq

    n_windows, window_size, n_features = windows.shape

    if n_fft is None:
        n_fft = window_size

    freq_features = []
    freq_names = []

    freqs = rfftfreq(n_fft, 1/fs)

    for i in range(n_windows):
        window_feats = []
        for j in range(n_features):
            signal = windows[i, :, j]
            spectrum = np.abs(rfft(signal, n=n_fft))

            window_feats.extend([
                np.mean(spectrum),
                np.std(spectrum),
                np.max(spectrum),
                freqs[np.argmax(spectrum)],
                np.sum(spectrum**2)
            ])

            if i == 0:
                freq_names.extend([
                    f'feat{j}_freq_mean', f'feat{j}_freq_std', f'feat{j}_freq_max',
                    f'feat{j}_dominant_freq', f'feat{j}_freq_energy'
                ])

        freq_features.append(window_feats)

    return np.array(freq_features), freq_names


def split_by_simulation(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    simulation_ids: Optional[np.ndarray] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple:
    if simulation_ids is None:
        raise ValueError("simulation_ids required for leakage-free splitting")

    rng = np.random.default_rng(random_seed)
    unique_ids = np.unique(simulation_ids)
    rng.shuffle(unique_ids)

    n_train = int(len(unique_ids) * train_ratio)
    n_val = int(len(unique_ids) * val_ratio)

    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train:n_train + n_val]
    test_ids = unique_ids[n_train + n_val:]

    train_mask = np.isin(simulation_ids, train_ids)
    val_mask = np.isin(simulation_ids, val_ids)
    test_mask = np.isin(simulation_ids, test_ids)

    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]

    result = (X_train, X_val, X_test)

    if y is not None:
        y_train = y[train_mask]
        y_val = y[val_mask]
        y_test = y[test_mask]
        result = result + (y_train, y_val, y_test)

    return result