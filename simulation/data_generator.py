import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import yaml

from physics import FourBarParams, FaultConfig, DEFAULT_FRICTION, FrictionModel, DEFAULT_FAULT_CONFIGS, FAULT_CLASS_TO_IDX
from simulation.simulator import FourBarSimulator, run_simulation
from simulation.sensor_model import SensorSuite, DEFAULT_SENSOR_CONFIGS


@dataclass
class DataGenerationConfig:
    duration: float = 10.0
    timestep: float = 0.001
    n_runs_normal: int = 200
    n_runs_per_fault: int = 100
    omega2_range: Tuple[float, float] = (5.0, 15.0)
    initial_theta2_range: Tuple[float, float] = (0.0, 2 * np.pi)
    noise_level: float = 0.01
    random_seed: int = 42
    output_dir: str = "data/raw"
    save_format: str = "csv"


class DataGenerator:
    def __init__(self, config: DataGenerationConfig, params: FourBarParams):
        self.config = config
        self.params = params
        self.rng = np.random.default_rng(config.random_seed)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.friction_model = DEFAULT_FRICTION
        self.sensor_suite = SensorSuite(DEFAULT_SENSOR_CONFIGS)

    def generate_single_run(
        self,
        fault: FaultConfig,
        run_id: int,
        omega2: Optional[float] = None,
        initial_theta2: Optional[float] = None
    ) -> Dict:
        if omega2 is None:
            omega2 = self.rng.uniform(*self.config.omega2_range)
        if initial_theta2 is None:
            initial_theta2 = self.rng.uniform(*self.config.initial_theta2_range)

        simulator = FourBarSimulator(self.params, self.friction_model, fault)
        result = simulator.simulate_constant_speed(
            self.config.duration,
            self.config.timestep,
            omega2,
            initial_theta2
        )
        result.simulation_id = run_id

        measurements = self.sensor_suite.measure_all(result)

        data = {
            'time': result.time,
            'theta2': result.theta2,
            'theta3': result.theta3,
            'theta4': result.theta4,
            'omega2': result.omega2,
            'omega3': result.omega3,
            'omega4': result.omega4,
            'alpha2': result.alpha2,
            'alpha3': result.alpha3,
            'alpha4': result.alpha4,
            'input_torque': result.input_torque,
            'theta2_measured': measurements['theta2_measured'],
            'omega2_measured': measurements['omega2_measured'],
            'alpha2_measured': measurements['alpha2_measured'],
            'torque_measured': measurements['torque_measured'],
            'fault_type': result.fault_type,
            'fault_severity': result.fault_severity,
            'simulation_id': run_id,
            'fault_class': FAULT_CLASS_TO_IDX.get(result.fault_type.upper(), 0)
        }

        return data

    def generate_healthy_data(self, n_runs: Optional[int] = None) -> List[Dict]:
        n_runs = n_runs or self.config.n_runs_normal
        fault = FaultConfig("normal", 1.0)
        data_list = []

        for i in tqdm(range(n_runs), desc="Generating healthy data"):
            data = self.generate_single_run(fault, i)
            data_list.append(data)

        return data_list

    def generate_fault_data(self, fault_configs: List[FaultConfig], n_runs_per_fault: Optional[int] = None) -> List[Dict]:
        n_runs = n_runs_per_fault or self.config.n_runs_per_fault
        data_list = []
        run_id = self.config.n_runs_normal

        for fault in fault_configs:
            for i in tqdm(range(n_runs), desc=f"Generating {fault.fault_type} data"):
                data = self.generate_single_run(fault, run_id)
                data_list.append(data)
                run_id += 1

        return data_list

    def generate_dataset(
        self,
        fault_configs: Optional[List[FaultConfig]] = None,
        n_runs_normal: Optional[int] = None,
        n_runs_per_fault: Optional[int] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        if fault_configs is None:
            fault_configs = DEFAULT_FAULT_CONFIGS[1:]

        healthy_data = self.generate_healthy_data(n_runs_normal)
        fault_data = self.generate_fault_data(fault_configs, n_runs_per_fault)

        return healthy_data, fault_data

    def save_data(self, data_list: List[Dict], subdir: str, prefix: str = "sim"):
        save_dir = self.output_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, data in enumerate(tqdm(data_list, desc=f"Saving {subdir}")):
            df = pd.DataFrame(data)
            df['simulation_id'] = data['simulation_id']
            df['fault_type'] = data['fault_type']
            df['fault_severity'] = data['fault_severity']
            df['fault_class'] = data['fault_class']

            if self.config.save_format == "csv":
                filepath = save_dir / f"{prefix}_{data['fault_type']}_{data['simulation_id']:04d}.csv"
                df.to_csv(filepath, index=False)
            elif self.config.save_format == "parquet":
                filepath = save_dir / f"{prefix}_{data['fault_type']}_{data['simulation_id']:04d}.parquet"
                df.to_parquet(filepath, index=False)

    def save_consolidated(self, data_list: List[Dict], filepath: str):
        all_dfs = []
        for data in data_list:
            df = pd.DataFrame(data)
            df['simulation_id'] = data['simulation_id']
            df['fault_type'] = data['fault_type']
            df['fault_severity'] = data['fault_severity']
            df['fault_class'] = data['fault_class']
            all_dfs.append(df)

        consolidated = pd.concat(all_dfs, ignore_index=True)
        consolidated.to_parquet(filepath, index=False)
        return consolidated


def load_consolidated_data(filepath: str) -> pd.DataFrame:
    return pd.read_parquet(filepath)


def load_config_from_yaml(config_path: str) -> DataGenerationConfig:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    sim = config.get('simulation', {})
    sensors = config.get('sensors', {})

    return DataGenerationConfig(
        duration=sim.get('duration', 10.0),
        timestep=sim.get('timestep', 0.001),
        n_runs_normal=sim.get('number_of_runs', 200),
        n_runs_per_fault=sim.get('number_of_runs', 100),
        noise_level=sensors.get('noise_level', 0.01),
        random_seed=config.get('random_seed', 42),
        output_dir=config.get('paths', {}).get('data_raw', 'data/raw')
    )


def create_default_generator(config_path: str = "config.yaml") -> DataGenerator:
    config = load_config_from_yaml(config_path)
    params = FourBarParams()
    return DataGenerator(config, params)