import numpy as np
import pytest
from physics import (
    FourBarParams,
    solve_position,
    calculate_velocity,
    calculate_acceleration,
    calculate_joint_positions,
    vector_loop_error,
    check_mechanism_validity
)


class TestKinematics:
    def setup_method(self):
        # Valid Grashof crank-rocker parameters
        # Links: [0.20, 0.08, 0.30, 0.25] -> sorted: [0.08, 0.20, 0.25, 0.30]
        # s=0.08, l=0.30, p=0.20, q=0.25 -> s+l=0.38 <= p+q=0.45 ✓
        # L2 is shortest and adjacent to ground -> crank-rocker
        self.params = FourBarParams(
            L1=0.20, L2=0.08, L3=0.30, L4=0.25,
            m2=0.5, m3=1.0, m4=0.7,
            I2=0.0005, I3=0.003, I4=0.0015
        )

    def test_position_analysis_valid(self):
        theta2 = np.pi / 4
        theta3, theta4 = solve_position(theta2, self.params)

        assert isinstance(theta3, float)
        assert isinstance(theta4, float)
        assert -np.pi <= theta3 <= np.pi
        assert -np.pi <= theta4 <= np.pi

    def test_position_analysis_vector_loop_closure(self):
        theta2 = np.pi / 3
        theta3, theta4 = solve_position(theta2, self.params)

        error = vector_loop_error(theta2, theta3, theta4, self.params)
        assert error < 1e-10, f"Vector loop not closed: error = {error}"

    def test_position_multiple_angles(self):
        for theta2 in np.linspace(0, 2*np.pi, 20):
            theta3, theta4 = solve_position(theta2, self.params)
            error = vector_loop_error(theta2, theta3, theta4, self.params)
            assert error < 1e-9, f"Vector loop not closed at θ₂={theta2}: error = {error}"

    def test_velocity_analysis(self):
        theta2 = np.pi / 4
        omega2 = 10.0

        theta3, theta4 = solve_position(theta2, self.params)
        omega3, omega4 = calculate_velocity(theta2, omega2, self.params)

        assert isinstance(omega3, float)
        assert isinstance(omega4, float)
        assert np.isfinite(omega3)
        assert np.isfinite(omega4)

    def test_acceleration_analysis(self):
        theta2 = np.pi / 4
        omega2 = 10.0
        alpha2 = 0.0

        theta3, theta4 = solve_position(theta2, self.params)
        omega3, omega4 = calculate_velocity(theta2, omega2, self.params)
        alpha3, alpha4 = calculate_acceleration(theta2, omega2, alpha2, self.params)

        assert isinstance(alpha3, float)
        assert isinstance(alpha4, float)
        assert np.isfinite(alpha3)
        assert np.isfinite(alpha4)

    def test_joint_positions(self):
        theta2 = np.pi / 4
        theta3, theta4 = solve_position(theta2, self.params)

        positions = calculate_joint_positions(theta2, theta3, theta4, self.params)

        assert 'O2' in positions
        assert 'A' in positions
        assert 'B' in positions
        assert 'O4' in positions

        assert np.allclose(positions['O2'], [0, 0])
        assert np.allclose(positions['O4'], [0.20, 0])

        L2_check = np.linalg.norm(positions['A'] - positions['O2'])
        L3_check = np.linalg.norm(positions['B'] - positions['A'])
        L4_check = np.linalg.norm(positions['B'] - positions['O4'])

        assert abs(L2_check - 0.08) < 1e-9
        assert abs(L3_check - 0.30) < 1e-9
        assert abs(L4_check - 0.25) < 1e-9

    def test_mechanism_validity(self):
        valid, msg = check_mechanism_validity(self.params)
        assert valid == True
        assert "Grashof" in msg

    def test_non_grashof_mechanism(self):
        # Non-Grashof: s+l > p+q
        params = FourBarParams(L1=0.10, L2=0.05, L3=0.05, L4=0.15)
        # links = [0.10, 0.05, 0.05, 0.15] -> s=0.05, l=0.15, p=0.05, q=0.10
        # s+l = 0.20 > p+q = 0.15
        valid, msg = check_mechanism_validity(params)
        assert valid == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])