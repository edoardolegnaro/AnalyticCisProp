import analcisprop.utils.variable_changes as vc
import numpy as np
import pytest
from analcisprop.constants import GML
from numpy.testing import assert_allclose


class TestVariableChanges:
    """Test suite for variable conversion functions."""

    def setup_method(self):
        """Set up test parameters."""
        self.mu = 398600.4418  # Earth's gravitational parameter [km^3/s^2]
        self.tolerance = 1e-10

        # Standard test orbit (elliptical)
        self.test_kep = [
            7000.0,
            0.1,
            np.radians(30),
            np.radians(45),
            np.radians(60),
            np.radians(90),
        ]

        # Circular orbit
        self.circular_kep = [
            7000.0,
            0.0,
            np.radians(30),
            np.radians(45),
            np.radians(60),
            np.radians(90),
        ]

        # Equatorial orbit
        self.equatorial_kep = [7000.0, 0.1, 0.0, 0.0, np.radians(60), np.radians(90)]

        # High eccentricity orbit
        self.eccentric_kep = [
            10000.0,
            0.8,
            np.radians(60),
            np.radians(120),
            np.radians(45),
            np.radians(180),
        ]

    def test_wrap_to_2pi(self):
        """Test angle wrapping function."""
        assert_allclose(vc.wrap_to_2pi(3 * np.pi), np.pi, rtol=self.tolerance)
        assert_allclose(vc.wrap_to_2pi(-np.pi), np.pi, rtol=self.tolerance)
        assert_allclose(vc.wrap_to_2pi(0), 0, rtol=self.tolerance)
        assert_allclose(vc.wrap_to_2pi(2 * np.pi), 0, rtol=self.tolerance)

    def test_M2E_E2M_roundtrip(self):
        """Test mean to eccentric anomaly conversion round trip."""
        eccentricities = [0.0, 0.1, 0.5, 0.9]
        # Avoid exactly 2π to prevent wrapping issues
        mean_anomalies = [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]

        for e in eccentricities:
            for M in mean_anomalies:
                E = vc.M2E(M, e)
                M_recovered = vc.E2M(E, e)
                # Handle angle wrapping - both should be equivalent mod 2π
                diff = abs(vc.wrap_to_2pi(M_recovered) - vc.wrap_to_2pi(M))
                # Also check if they differ by 2π
                diff_2pi = abs(diff - 2 * np.pi)
                assert min(diff, diff_2pi) < self.tolerance

    def test_ic2par_par2ic_roundtrip_standard(self):
        """Test cartesian to Keplerian and back for standard orbit."""
        # Convert to cartesian
        state = vc.par2ic(self.test_kep, self.mu)

        # Convert back to Keplerian
        kep_recovered = vc.ic2par(state, self.mu)

        assert_allclose(kep_recovered, self.test_kep, rtol=1e-8)

    def test_ic2par_par2ic_roundtrip_circular(self):
        """Test cartesian to Keplerian and back for circular orbit."""
        state = vc.par2ic(self.circular_kep, self.mu)
        kep_recovered = vc.ic2par(state, self.mu)

        # For circular orbits, argument of perigee is undefined
        assert_allclose(kep_recovered[0], self.circular_kep[0], rtol=1e-8)  # a
        assert_allclose(kep_recovered[1], self.circular_kep[1], atol=1e-10)  # e
        assert_allclose(kep_recovered[2], self.circular_kep[2], rtol=1e-8)  # i
        assert_allclose(kep_recovered[3], self.circular_kep[3], rtol=1e-8)  # OM
        # Don't check om and M for circular orbits as they may be redefined

    def test_ic2par_par2ic_roundtrip_equatorial(self):
        """Test cartesian to Keplerian and back for equatorial orbit."""
        state = vc.par2ic(self.equatorial_kep, self.mu)
        kep_recovered = vc.ic2par(state, self.mu)

        # For equatorial orbits, RAAN is undefined
        assert_allclose(kep_recovered[0], self.equatorial_kep[0], rtol=1e-8)  # a
        assert_allclose(kep_recovered[1], self.equatorial_kep[1], rtol=1e-8)  # e
        assert_allclose(kep_recovered[2], self.equatorial_kep[2], atol=1e-10)  # i

    def test_ic2par_par2ic_roundtrip_eccentric(self):
        """Test cartesian to Keplerian and back for high eccentricity orbit."""
        state = vc.par2ic(self.eccentric_kep, self.mu)
        kep_recovered = vc.ic2par(state, self.mu)

        assert_allclose(kep_recovered, self.eccentric_kep, rtol=1e-8)

    def test_kep2equinox_equinox2kep_roundtrip(self):
        """Test Keplerian to equinoctial and back."""
        equinox = vc.kep2equinox(self.test_kep)
        kep_recovered = vc.equinox2kep(equinox)

        assert_allclose(kep_recovered, self.test_kep, rtol=self.tolerance)

    def test_kep2equinox_equinox2kep_roundtrip_circular(self):
        """Test Keplerian to equinoctial and back for circular orbit."""
        equinox = vc.kep2equinox(self.circular_kep)
        kep_recovered = vc.equinox2kep(equinox)

        # For circular orbits, some angles may be undefined/wrapped differently
        assert_allclose(kep_recovered[0], self.circular_kep[0], rtol=1e-8)  # a
        assert_allclose(kep_recovered[1], self.circular_kep[1], atol=1e-10)  # e
        assert_allclose(kep_recovered[2], self.circular_kep[2], rtol=1e-8)  # i
        assert_allclose(kep_recovered[3], self.circular_kep[3], rtol=1e-8)  # OM
        # Don't check om and M for circular orbits as they may be redefined

    def test_del2kep_kep2del_roundtrip(self):
        """Test Delaunay to Keplerian and back."""
        # Use first 5 elements for Delaunay conversion
        kep_for_del = self.test_kep[:5]  # [a, e, i, OM, om]

        del_vars = vc.kep2del(kep_for_del)
        L = del_vars[0]  # Extract L for del2kep
        DEL = del_vars[1:]  # [G, H, g, h] where g=om, h=OM

        kep_recovered = vc.del2kep(DEL, L)

        # Now the functions should be consistent:
        # kep2del: [a, e, i, OM, om] -> [L, G, H, om, OM]
        # del2kep: [G, H, om, OM], L -> [a, e, i, OM, om]
        # So kep_recovered should match kep_for_del exactly

        assert_allclose(kep_recovered, kep_for_del, rtol=self.tolerance)

    def test_mod2del_del2mod_roundtrip(self):
        """Test modified to Delaunay and back."""
        # Create test modified variables
        L = 1000.0
        mod_vars = [100.0, 50.0, 0.1, 0.2]  # [P, Q, p, q]

        del_vars = vc.mod2del(mod_vars, L)
        mod_recovered = vc.del2mod(del_vars, L)

        assert_allclose(mod_recovered, mod_vars, rtol=self.tolerance)

    def test_kep2spcvars_consistency(self):
        """Test kep2spcvars returns reasonable values."""
        spc_vars = vc.kep2spcvars(self.test_kep)

        # Check that we get 10 variables
        assert len(spc_vars) == 10

        # Check that radius is positive
        assert spc_vars[0] > 0

        # Check that mean motion is positive
        assert spc_vars[1] > 0

        # Check that eta (sqrt(1-e^2)) is reasonable
        expected_eta = np.sqrt(1 - self.test_kep[1] ** 2)
        assert_allclose(spc_vars[4], expected_eta, rtol=self.tolerance)

    def test_physical_constraints(self):
        """Test that conversions maintain physical constraints."""
        state = vc.par2ic(self.test_kep, self.mu)

        # Check energy conservation
        rv = state[:3]
        vv = state[3:]
        r = np.linalg.norm(rv)
        v = np.linalg.norm(vv)

        # Specific energy
        energy = v**2 / 2 - self.mu / r
        expected_energy = -self.mu / (2 * self.test_kep[0])

        assert_allclose(energy, expected_energy, rtol=1e-8)

        # Check angular momentum conservation
        h_vec = np.cross(rv, vv)
        h_mag = np.linalg.norm(h_vec)
        expected_h = np.sqrt(self.mu * self.test_kep[0] * (1 - self.test_kep[1] ** 2))

        assert_allclose(h_mag, expected_h, rtol=1e-8)

    def test_edge_case_zero_eccentricity(self):
        """Test handling of exactly zero eccentricity."""
        zero_ecc_kep = [
            7000.0,
            0.0,
            np.radians(30),
            np.radians(45),
            0.0,
            np.radians(90),
        ]

        state = vc.par2ic(zero_ecc_kep, self.mu)
        kep_recovered = vc.ic2par(state, self.mu)

        # Semi-major axis and inclination should be preserved
        assert_allclose(kep_recovered[0], zero_ecc_kep[0], rtol=1e-8)
        assert_allclose(kep_recovered[1], 0.0, atol=1e-12)
        assert_allclose(kep_recovered[2], zero_ecc_kep[2], rtol=1e-8)

    def test_edge_case_zero_inclination(self):
        """Test handling of exactly zero inclination."""
        zero_inc_kep = [7000.0, 0.1, 0.0, 0.0, np.radians(60), np.radians(90)]

        state = vc.par2ic(zero_inc_kep, self.mu)
        kep_recovered = vc.ic2par(state, self.mu)

        # Semi-major axis and eccentricity should be preserved
        assert_allclose(kep_recovered[0], zero_inc_kep[0], rtol=1e-8)
        assert_allclose(kep_recovered[1], zero_inc_kep[1], rtol=1e-8)
        assert_allclose(kep_recovered[2], 0.0, atol=1e-12)

    @pytest.mark.parametrize("e", [0.0, 0.1, 0.5, 0.9])
    def test_anomaly_conversions_various_eccentricities(self, e):
        """Test anomaly conversions for various eccentricities."""
        # Avoid exactly 2π to prevent wrapping issues
        mean_anomalies = np.linspace(0, 2 * np.pi, 8, endpoint=False)

        for M in mean_anomalies:
            E = vc.M2E(M, e)
            M_recovered = vc.E2M(E, e)
            # Handle angle wrapping
            diff = abs(vc.wrap_to_2pi(M_recovered) - vc.wrap_to_2pi(M))
            diff_2pi = abs(diff - 2 * np.pi)
            assert min(diff, diff_2pi) < self.tolerance


if __name__ == "__main__":
    # Simple way to run tests directly
    test_instance = TestVariableChanges()
    test_instance.setup_method()

    # Run individual tests
    try:
        test_instance.test_wrap_to_2pi()
        print("✓ test_wrap_to_2pi passed")
    except Exception as e:
        print(f"✗ test_wrap_to_2pi failed: {e}")

    try:
        test_instance.test_M2E_E2M_roundtrip()
        print("✓ test_M2E_E2M_roundtrip passed")
    except Exception as e:
        print(f"✗ test_M2E_E2M_roundtrip failed: {e}")

    try:
        test_instance.test_M2E_E2M_roundtrip()
        print("✓ test_M2E_E2M_roundtrip passed")
    except Exception as e:
        print(f"✗ test_M2E_E2M_roundtrip failed: {e}")

    # Add more tests as needed...
    print("Tests completed!")
