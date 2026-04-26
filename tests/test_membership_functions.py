import numpy as np
import pytest

print("TEST MODULE: tests/test_membership_functions.py loaded")

from core.membership_functions import (
    triangular, trapezoidal, gaussian, generalized_bell,
    sigmoid, s_shaped, z_shaped, pi_shaped, singleton,
    make_mf, MembershipFunction, MFType,
)
from tests.helpers import assert_in_range


def test_triangular_peak_and_feet():
    assert float(triangular(25, 15, 25, 35)) == 1.0
    assert float(triangular(15, 15, 25, 35)) == 0.0
    assert float(triangular(35, 15, 25, 35)) == 0.0


def test_trapezoidal_flat_top_and_slopes():
    for x in [50, 60, 70, 80]:
        assert float(trapezoidal(x, 40, 50, 80, 90)) == 1.0
    assert float(trapezoidal(45, 40, 50, 80, 90)) == 0.5
    assert float(trapezoidal(85, 40, 50, 80, 90)) == 0.5


def test_gaussian_symmetry_and_range():
    assert float(gaussian(50, 50, 10)) == 1.0
    assert float(gaussian(40, 50, 10)) == float(gaussian(60, 50, 10))
    x = np.linspace(-100, 100, 500)
    assert_in_range(gaussian(x, 0, 15))


def test_generalized_bell_behavior():
    assert float(generalized_bell(50, 10, 2, 50)) == 1.0
    x = np.linspace(0, 100, 300)
    assert_in_range(generalized_bell(x, 10, 2, 50))


def test_sigmoid_crossing_and_direction():
    assert pytest.approx(float(sigmoid(50, 50, 0.2)), rel=1e-5) == 0.5
    assert float(sigmoid(0, 50, 0.2)) < 0.5
    assert float(sigmoid(100, 50, 0.2)) > 0.5
    assert float(sigmoid(0, 50, -0.2)) > 0.5
    assert float(sigmoid(100, 50, -0.2)) < 0.5


def test_s_and_z_shaped_complementarity():
    assert float(s_shaped(0, 0, 10)) == 0.0
    assert float(s_shaped(10, 0, 10)) == 1.0
    assert float(z_shaped(0, 0, 10)) == 1.0
    assert float(z_shaped(10, 0, 10)) == 0.0
    x = np.linspace(0, 10, 100)
    assert np.allclose(s_shaped(x, 0, 10) + z_shaped(x, 0, 10), 1.0)


def test_pi_shaped_topology():
    x = np.linspace(-10, 10, 300)
    y = pi_shaped(x, 0, 3, 7, 10)
    assert_in_range(y)
    assert np.all(y[(x >= 3) & (x <= 7)] > 0.99)


def test_singleton_behavior():
    assert float(singleton(5.0, center=5.0)) == 1.0
    assert float(singleton(5.1, center=5.0, tolerance=1e-9)) == 0.0


def test_make_mf_factory_and_compute():
    mf = make_mf("young", "triangular", {"a": 15, "b": 25, "c": 35}, 0, 100)
    assert isinstance(mf, MembershipFunction)
    assert mf.name == "young"
    assert mf.mf_type == MFType.TRIANGULAR
    assert float(mf.compute(25)) == 1.0
    assert float(mf.compute(0)) == 0.0
