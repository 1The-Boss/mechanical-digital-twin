from dataclasses import dataclass, replace
from typing import Optional
import numpy as np


@dataclass
class FourBarParams:
    L1: float = 0.20
    L2: float = 0.08
    L3: float = 0.30
    L4: float = 0.25
    m2: float = 0.5
    m3: float = 1.0
    m4: float = 0.7
    I2: float = 0.0005
    I3: float = 0.003
    I4: float = 0.0015
    g: float = 9.81

    def __post_init__(self):
        self.validate()

    def validate(self) -> bool:
        if any(getattr(self, attr) <= 0 for attr in ['L1', 'L2', 'L3', 'L4']):
            raise ValueError("All link lengths must be positive")
        if any(getattr(self, attr) <= 0 for attr in ['m2', 'm3', 'm4']):
            raise ValueError("All masses must be positive")
        if any(getattr(self, attr) < 0 for attr in ['I2', 'I3', 'I4']):
            raise ValueError("Moments of inertia must be non-negative")
        return True

    def is_grashof(self) -> bool:
        links = [self.L1, self.L2, self.L3, self.L4]
        s = min(links)
        l = max(links)
        p, q = sorted(links)[1], sorted(links)[2]
        return s + l <= p + q

    def with_fault(self, fault_type: str, severity: float) -> 'FourBarParams':
        new_params = replace(self)
        if fault_type == "friction_fault":
            pass
        elif fault_type == "mass_fault":
            new_params.m3 = self.m3 * severity
            new_params.I3 = self.I3 * severity
        elif fault_type == "joint_fault":
            pass
        elif fault_type == "inertia_fault":
            new_params.I3 = self.I3 * severity
        return new_params


DEFAULT_PARAMS = FourBarParams()


def load_params_from_config(config_path: str) -> FourBarParams:
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    mech = config.get('mechanism', {})
    return FourBarParams(
        L1=mech.get('L1', 0.20),
        L2=mech.get('L2', 0.08),
        L3=mech.get('L3', 0.30),
        L4=mech.get('L4', 0.25),
        m2=mech.get('m2', 0.5),
        m3=mech.get('m3', 1.0),
        m4=mech.get('m4', 0.7),
        I2=mech.get('I2', 0.0005),
        I3=mech.get('I3', 0.003),
        I4=mech.get('I4', 0.0015),
    )