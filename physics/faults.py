import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from .mechanism import FourBarParams


@dataclass
class FaultConfig:
    fault_type: str
    severity: float
    affected_joint: int = 1
    description: str = ""

    def __post_init__(self):
        valid_types = ["normal", "friction_fault", "mass_fault", "joint_fault", "inertia_fault"]
        if self.fault_type not in valid_types:
            raise ValueError(f"Invalid fault type: {self.fault_type}. Must be one of {valid_types}")
        if self.severity <= 1.0 and self.fault_type != "normal":
            raise ValueError(f"Fault severity must be > 1.0 for fault types, got {self.severity}")


def apply_fault(params: FourBarParams, fault: FaultConfig) -> FourBarParams:
    from copy import deepcopy
    new_params = deepcopy(params)

    if fault.fault_type == "normal":
        return new_params

    elif fault.fault_type == "friction_fault":
        return new_params

    elif fault.fault_type == "mass_fault":
        joint = fault.affected_joint
        if joint == 2:
            new_params.m2 = params.m2 * fault.severity
            new_params.I2 = params.I2 * fault.severity
        elif joint == 3:
            new_params.m3 = params.m3 * fault.severity
            new_params.I3 = params.I3 * fault.severity
        elif joint == 4:
            new_params.m4 = params.m4 * fault.severity
            new_params.I4 = params.I4 * fault.severity
        else:
            new_params.m3 = params.m3 * fault.severity
            new_params.I3 = params.I3 * fault.severity

    elif fault.fault_type == "joint_fault":
        return new_params

    elif fault.fault_type == "inertia_fault":
        joint = fault.affected_joint
        if joint == 2:
            new_params.I2 = params.I2 * fault.severity
        elif joint == 3:
            new_params.I3 = params.I3 * fault.severity
        elif joint == 4:
            new_params.I4 = params.I4 * fault.severity
        else:
            new_params.I3 = params.I3 * fault.severity

    return new_params


def get_fault_friction_params(fault: FaultConfig, base_coulomb: np.ndarray, base_viscous: np.ndarray) -> tuple:
    coulomb = base_coulomb.copy()
    viscous = base_viscous.copy()

    if fault.fault_type == "friction_fault":
        joint = fault.affected_joint - 1
        if 0 <= joint < 3:
            coulomb[joint] *= fault.severity
            viscous[joint] *= fault.severity
        else:
            coulomb *= fault.severity
            viscous *= fault.severity

    elif fault.fault_type == "joint_fault":
        joint = fault.affected_joint - 1
        if 0 <= joint < 3:
            coulomb[joint] *= fault.severity
            viscous[joint] *= fault.severity

    return coulomb, viscous


def get_fault_description(fault: FaultConfig) -> str:
    descriptions = {
        "normal": "Normal operation - no faults",
        "friction_fault": f"Increased friction at joint {fault.affected_joint} (severity: {fault.severity:.1f}x)",
        "mass_fault": f"Increased mass at link {fault.affected_joint} (severity: {fault.severity:.1f}x)",
        "joint_fault": f"Joint degradation at joint {fault.affected_joint} (severity: {fault.severity:.1f}x)",
        "inertia_fault": f"Increased inertia at link {fault.affected_joint} (severity: {fault.severity:.1f}x)"
    }
    return descriptions.get(fault.fault_type, "Unknown fault")


FAULT_CLASSES = {
    0: "NORMAL",
    1: "FRICTION_FAULT",
    2: "MASS_FAULT",
    3: "JOINT_FAULT",
    4: "INERTIA_FAULT"
}

FAULT_CLASS_TO_IDX = {v: k for k, v in FAULT_CLASSES.items()}


def create_fault_configs(
    fault_types: list,
    severities: list,
    affected_joints: list = None
) -> list:
    if affected_joints is None:
        affected_joints = [3] * len(fault_types)

    configs = []
    for i, (fault_type, severity) in enumerate(zip(fault_types, severities)):
        joint = affected_joints[i] if i < len(affected_joints) else 3
        configs.append(FaultConfig(
            fault_type=fault_type,
            severity=severity,
            affected_joint=joint,
            description=get_fault_description(FaultConfig(fault_type=fault_type, severity=severity, affected_joint=joint))
        ))
    return configs


DEFAULT_FAULT_CONFIGS = [
    FaultConfig("normal", 1.0),
    FaultConfig("friction_fault", 3.0, 2),
    FaultConfig("mass_fault", 2.0, 3),
    FaultConfig("joint_fault", 5.0, 3),
    FaultConfig("inertia_fault", 2.5, 3)
]