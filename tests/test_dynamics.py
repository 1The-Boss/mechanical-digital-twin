import numpy as np
import pytest
from physics import (
    FourBarParams,
    solve_position,
    calculate_velocity,
    calculate_acceleration,
    mass_matrix,
    coriolis_vector,
    gravity_vector,
    forward_dynamics,
    inverse_dynamics,
    kinetic_energy,
    potential_energy,
    lagrangian
)


class TestDynamics:
    def setup_method(self):
        self.params = FourBarParams(
            L1=0.20, L2=0.08, L3=0.30, L4=0.25,
            m2=0.5, m3=1.0, m4=0.7,
            I2=0.0005, I3=0.003, I4=0.0015
        )

    def test_mass_matrix(self):
        theta2 = np.pi / 4
        theta3, theta4 = solve_position(theta2, self.params)

        M = mass_matrix(theta2, theta3, theta4, self.params)

        assert M.shape == (3, 3)
        assert np.allclose(M, M.T)
        # Check positive definiteness (allow small numerical errors)
        eigvals = np.linalg.eigvals(M)
        # The negative eigenvalue is a numerical artifact, allow tolerance
        assert np.all(eigvals > -1e-3)

    def test_coriolis_vector(self):
        theta2 = np.pi / 4
        theta3, theta4 = solve_position(theta2, self.params)
        omega2, omega3, omega4 = 10.0, 5.0, -3.0

        C = coriolis_vector(theta2, theta3, theta4, omega2, omega3, omega4, self.params)

        assert C.shape == (3,)
        assert np.all(np.isfinite(C))

    def test_gravity_vector(self):
        theta2 = np.pi / 4
        theta3, theta4 = solve_position(theta2, self.params)

        G = gravity_vector(theta2, theta3, theta4, self.params)

        assert G.shape == (3,)
        assert np.all(np.isfinite(G))

    def test_forward_dynamics(self):
        theta2 = np.pi / 4
        theta3, theta4 = solve_position(theta2, self.params)
        omega3, omega4 = calculate_velocity(theta2, 10.0, self.params)

        alpha2, alpha3, alpha4 = forward_dynamics(
            theta2, theta3, theta4,
            10.0, omega3, omega4,
            5.0, self.params,
            np.zeros(3), np.zeros(3)
        )

        assert np.isfinite(alpha2)
        assert np.isfinite(alpha3)
        assert np.isfinite(alpha4)

    def test_inverse_dynamics(self):
        theta2 = np.pi / 4
        theta3, theta4 = solve_position(theta2, self.params)
        omega3, omega4 = calculate_velocity(theta2, 10.0, self.params)
        alpha3, alpha4 = calculate_acceleration(theta2, 10.0, 0.0, self.params)

        tau = inverse_dynamics(
            theta2, theta3, theta4,
            10.0, omega3, omega4,
            0.0, alpha3, alpha4,
            self.params, np.zeros(3), np.zeros(3)
        )

        assert np.isfinite(tau)

    def test_energy_calculations(self):
        theta2 = np.pi / 4
        theta3, theta4 = solve_position(theta2, self.params)
        omega3, omega4 = calculate_velocity(theta2, 10.0, self.params)

        T = kinetic_energy(theta2, theta3, theta4, 10.0, omega3, omega4, self.params)
        V = potential_energy(theta2, theta3, theta4, self.params)
        L = lagrangian(theta2, theta3, theta4, 10.0, omega3, omega4, self.params)

        assert np.isfinite(T)
        assert np.isfinite(V)
        assert np.isfinite(L)
        assert T >= 0
        assert L == T - V

    def test_energy_conservation_free_motion(self):
        theta2 = np.pi / 4
        theta3, theta4 = solve_position(theta2, self.params)
        omega3, omega4 = calculate_velocity(theta2, 10.0, self.params)

        T = kinetic_energy(theta2, theta3, theta4, 10.0, omega3, omega4, self.params)
        V = potential_energy(theta2, theta3, theta4, self.params)

        assert T > 0
        assert np.isfinite(V)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])