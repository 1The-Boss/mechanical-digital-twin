import numpy as np
import pytest
from physics import FourBarParams, FaultConfig
from simulation import FourBarSimulator, SensorSuite, DEFAULT_SENSOR_CONFIGS
from physics.friction import DEFAULT_FRICTION


class TestSimulation:
    def setup_method(self):
        self.params = FourBarParams(
            L1=0.20, L2=0.08, L3=0.18, L4=0.15,
            m2=0.5, m3=1.0, m4=0.7,
            I2=0.0005, I3=0.003, I4=0.0015
        )

    def test_simulator_creation(self):
        simulator = FourBarSimulator(self.params)
        assert simulator.params == self.params
        assert simulator.friction_model is not None

    def test_simulate_constant_speed(self):
        simulator = FourBarSimulator(self.params)
        result = simulator.simulate_constant_speed(
            duration=1.0,
            timestep=0.001,
            omega2_nominal=10.0,
            initial_theta2=0.0
        )

        assert len(result.time) == 1001
        assert np.all(np.isfinite(result.theta2))
        assert np.all(np.isfinite(result.omega2))
        assert np.all(np.isfinite(result.alpha2))
        assert np.all(np.isfinite(result.input_torque))

    def test_simulation_result_structure(self):
        simulator = FourBarSimulator(self.params)
        result = simulator.simulate_constant_speed(0.5, 0.001, 10.0, 0.0)

        assert hasattr(result, 'time')
        assert hasattr(result, 'theta2')
        assert hasattr(result, 'theta3')
        assert hasattr(result, 'theta4')
        assert hasattr(result, 'omega2')
        assert hasattr(result, 'omega3')
        assert hasattr(result, 'omega4')
        assert hasattr(result, 'alpha2')
        assert hasattr(result, 'alpha3')
        assert hasattr(result, 'alpha4')
        assert hasattr(result, 'input_torque')
        assert hasattr(result, 'fault_type')
        assert hasattr(result, 'fault_severity')
        assert hasattr(result, 'simulation_id')

    def test_fault_injection_friction(self):
        fault = FaultConfig("friction_fault", 3.0, affected_joint=2)
        simulator = FourBarSimulator(self.params, fault=fault)

        assert simulator.fault.fault_type == "friction_fault"
        assert simulator.fault.severity == 3.0

    def test_fault_injection_mass(self):
        fault = FaultConfig("mass_fault", 2.0, affected_joint=3)
        simulator = FourBarSimulator(self.params, fault=fault)

        assert simulator.fault_params.m3 == 2.0
        assert simulator.fault_params.I3 == 0.006

    def test_fault_injection_inertia(self):
        fault = FaultConfig("inertia_fault", 2.5, affected_joint=3)
        simulator = FourBarSimulator(self.params, fault=fault)

        assert simulator.fault_params.I3 == 0.0075

    def test_sensor_suite(self):
        suite = SensorSuite(DEFAULT_SENSOR_CONFIGS)

        time = np.linspace(0, 1, 1000)
        theta2 = 10.0 * time
        omega2 = np.full_like(time, 10.0)
        alpha2 = np.zeros_like(time)
        torque = np.full_like(time, 5.0)

        class MockResult:
            pass

        result = MockResult()
        result.time = time
        result.theta2 = theta2
        result.omega2 = omega2
        result.alpha2 = alpha2
        result.input_torque = torque

        measurements = suite.measure_all(result)

        assert 'theta2_measured' in measurements
        assert 'omega2_measured' in measurements
        assert 'alpha2_measured' in measurements
        assert 'torque_measured' in measurements

        assert len(measurements['theta2_measured']) == 1000

    def test_sensor_noise(self):
        suite = SensorSuite(DEFAULT_SENSOR_CONFIGS)

        signal = np.ones(100) * 5.0
        time = np.linspace(0, 1, 100)

        class MockResult:
            pass

        result = MockResult()
        result.time = time
        result.theta2 = signal
        result.omega2 = signal
        result.alpha2 = signal
        result.input_torque = signal

        measurements = suite.measure_all(result)

        for key, measured in measurements.items():
            assert np.std(measured) > 0
            assert np.isclose(np.mean(measured), 5.0, atol=0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])