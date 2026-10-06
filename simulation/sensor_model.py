import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from physics import FourBarParams


@dataclass
class SensorConfig:
    noise_level: float = 0.01
    bias: float = 0.0
    sampling_frequency: int = 1000
    missing_samples_prob: float = 0.0
    noise_type: str = "gaussian"

    def __post_init__(self):
        if self.noise_level < 0:
            raise ValueError("noise_level must be non-negative")
        if not 0 <= self.missing_samples_prob <= 1:
            raise ValueError("missing_samples_prob must be in [0, 1]")


DEFAULT_SENSOR_CONFIGS = {
    'encoder': SensorConfig(noise_level=0.001, bias=0.0, sampling_frequency=1000),
    'tachometer': SensorConfig(noise_level=0.01, bias=0.0, sampling_frequency=1000),
    'accelerometer': SensorConfig(noise_level=0.02, bias=0.0, sampling_frequency=1000),
    'torque_sensor': SensorConfig(noise_level=0.01, bias=0.0, sampling_frequency=1000)
}


class VirtualSensor:
    def __init__(self, config: SensorConfig, signal_name: str):
        self.config = config
        self.signal_name = signal_name
        self.rng = np.random.default_rng()

    def measure(self, true_signal: np.ndarray, time: np.ndarray) -> np.ndarray:
        measured = true_signal.copy()

        if self.config.noise_type == "gaussian":
            noise = self.rng.normal(0, self.config.noise_level * np.std(true_signal) if np.std(true_signal) > 0 else self.config.noise_level, size=true_signal.shape)
            measured += noise
        elif self.config.noise_type == "uniform":
            noise = self.rng.uniform(-self.config.noise_level, self.config.noise_level, size=true_signal.shape)
            measured += noise

        measured += self.config.bias

        if self.config.missing_samples_prob > 0:
            missing_mask = self.rng.random(size=true_signal.shape) < self.config.missing_samples_prob
            measured[missing_mask] = np.nan

        return measured


class SensorSuite:
    def __init__(self, configs: Optional[Dict[str, SensorConfig]] = None):
        self.configs = configs or DEFAULT_SENSOR_CONFIGS
        self.sensors = {
            name: VirtualSensor(config, name)
            for name, config in self.configs.items()
        }

    def measure_all(self, simulation_result) -> Dict[str, np.ndarray]:
        measurements = {}

        measurements['theta2_measured'] = self.sensors['encoder'].measure(simulation_result.theta2, simulation_result.time)
        measurements['omega2_measured'] = self.sensors['tachometer'].measure(simulation_result.omega2, simulation_result.time)
        measurements['alpha2_measured'] = self.sensors['accelerometer'].measure(simulation_result.alpha2, simulation_result.time)
        measurements['torque_measured'] = self.sensors['torque_sensor'].measure(simulation_result.input_torque, simulation_result.time)

        return measurements

    def add_noise_to_result(self, simulation_result) -> Dict[str, np.ndarray]:
        return self.measure_all(simulation_result)


def downsample_signal(signal: np.ndarray, time: np.ndarray, target_fs: int, original_fs: int) -> Tuple[np.ndarray, np.ndarray]:
    if target_fs >= original_fs:
        return signal, time

    factor = original_fs // target_fs
    if factor <= 1:
        return signal, time

    indices = np.arange(0, len(signal), factor)
    indices = indices[indices < len(signal)]

    return signal[indices], time[indices]


def add_sensor_noise(
    signal: np.ndarray,
    noise_level: float,
    bias: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(0, noise_level * (np.std(signal) if np.std(signal) > 0 else 1.0), size=signal.shape)
    return signal + noise + bias


def quantize_signal(signal: np.ndarray, resolution: float) -> np.ndarray:
    return np.round(signal / resolution) * resolution


def simulate_encoder(
    theta: np.ndarray,
    time: np.ndarray,
    resolution: float = 2 * np.pi / 4096,
    noise_level: float = 0.001,
    bias: float = 0.0
) -> np.ndarray:
    measured = quantize_signal(theta, resolution)
    return add_sensor_noise(measured, noise_level, bias)


def simulate_tachometer(
    omega: np.ndarray,
    time: np.ndarray,
    noise_level: float = 0.01,
    bias: float = 0.0
) -> np.ndarray:
    return add_sensor_noise(omega, noise_level, bias)


def simulate_accelerometer(
    alpha: np.ndarray,
    time: np.ndarray,
    noise_level: float = 0.02,
    bias: float = 0.0
) -> np.ndarray:
    return add_sensor_noise(alpha, noise_level, bias)


def simulate_torque_sensor(
    torque: np.ndarray,
    time: np.ndarray,
    noise_level: float = 0.01,
    bias: float = 0.0
) -> np.ndarray:
    return add_sensor_noise(torque, noise_level, bias)