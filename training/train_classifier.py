import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from preprocessing import FeatureScaler
from preprocessing.windowing import create_windows, extract_statistical_features, split_by_simulation


def prepare_classifier_data(
    df: pd.DataFrame,
    physics_predictions: Dict[str, np.ndarray],
    feature_columns: Optional[List[str]] = None,
    window_size: int = 50,
    stride: int = 10
) -> Tuple[np.ndarray, np.ndarray, FeatureScaler, np.ndarray]:
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

    fault_labels = df['fault_class'].values if 'fault_class' in df.columns else np.zeros(len(df))
    label_windows = create_windows(fault_labels.reshape(-1, 1), window_size, stride)
    labels = label_windows[:, -1, 0].astype(int)

    simulation_ids = None
    if 'simulation_id' in df.columns:
        sim_ids = df['simulation_id'].values
        sim_id_windows = create_windows(sim_ids.reshape(-1, 1), window_size, stride)
        simulation_ids = sim_id_windows[:, -1, 0].astype(int)

    scaler = FeatureScaler('standard')
    stat_features_scaled = scaler.fit_transform(stat_features)

    return stat_features_scaled, labels, scaler, simulation_ids


def create_classifier_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    simulation_ids: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_fault_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = 'random_forest',
    **kwargs
):
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 20),
            min_samples_split=kwargs.get('min_samples_split', 5),
            min_samples_leaf=kwargs.get('min_samples_leaf', 2),
            random_state=kwargs.get('random_state', 42),
            n_jobs=-1,
            class_weight='balanced'
        )
    elif model_type == 'svm':
        model = SVC(
            kernel=kwargs.get('kernel', 'rbf'),
            C=kwargs.get('C', 1.0),
            gamma=kwargs.get('gamma', 'scale'),
            probability=True,
            random_state=kwargs.get('random_state', 42),
            class_weight='balanced'
        )
    elif model_type == 'mlp':
        model = MLPClassifier(
            hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (128, 64, 32)),
            activation=kwargs.get('activation', 'relu'),
            alpha=kwargs.get('alpha', 0.001),
            learning_rate_init=kwargs.get('learning_rate', 0.001),
            max_iter=kwargs.get('max_iter', 500),
            early_stopping=True,
            validation_fraction=0.1,
            random_state=kwargs.get('random_state', 42)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

    # Use actual classes present in the data
    unique_classes = np.unique(y_test)
    if class_names is None:
        class_names = [f'Class_{i}' for i in unique_classes]
    else:
        # Filter class_names to match actual classes
        class_names = [class_names[i] for i in unique_classes]

    report = classification_report(y_test, y_pred, labels=unique_classes, target_names=class_names, output_dict=True, zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'classification_report': report,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def run_classifier_training(
    df: pd.DataFrame,
    physics_predictions: Dict[str, np.ndarray],
    config: Dict[str, Any],
    model_save_path: str = 'models_saved/fault_classifier.pkl',
    scaler_save_path: str = 'models_saved/classifier_scaler.pkl'
) -> Tuple[Any, FeatureScaler, Dict]:
    X_scaled, labels, scaler, simulation_ids = prepare_classifier_data(
        df, physics_predictions,
        window_size=config.get('window_size', 50),
        stride=config.get('stride', 10)
    )

    X_train, X_val, X_test, y_train, y_val, y_test = create_classifier_dataloaders(
        X_scaled, labels,
        val_split=config.get('validation_split', 0.15),
        test_split=config.get('test_split', 0.15),
        random_seed=config.get('random_seed', 42),
        simulation_ids=simulation_ids
    )

    model = train_fault_classifier(
        X_train, y_train, X_val, y_val,
        model_type=config.get('classifier_type', 'random_forest'),
        n_estimators=config.get('n_estimators', 200),
        max_depth=config.get('max_depth', 20),
        random_state=config.get('random_seed', 42)
    )

    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_save_path)
    scaler.save(scaler_save_path)

    class_names = ['NORMAL', 'FRICTION_FAULT', 'MASS_FAULT', 'JOINT_FAULT', 'INERTIA_FAULT']
    metrics = evaluate_classifier(model, X_test, y_test, class_names)

    print(f"Classifier Test Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")

    return model, scaler, metrics