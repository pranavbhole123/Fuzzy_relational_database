import numpy as np

print("TEST MODULE: tests/test_defuzzification.py loaded")

from core.defuzzification import (
    centroid, bisector, mean_of_maxima,
    smallest_of_maxima, largest_of_maxima,
    defuzzify, compare_methods,
)


def setup_symmetric_triangular():
    u = np.linspace(0, 10, 1000)
    mu = np.maximum(0.0, np.minimum(1.0, (10 - np.abs(u - 5)) / 5))
    return u, mu


def test_centroid_and_bisector():
    u, mu = setup_symmetric_triangular()
    assert abs(centroid(u, mu) - 5.0) < 0.2
    assert abs(bisector(u, mu) - 5.0) < 0.5


def test_mom_som_lom_ordering():
    u = np.linspace(0, 10, 1000)
    mu = np.maximum(0.0, np.minimum(1.0, (10 - np.abs(u - 5)) / 5))
    som = smallest_of_maxima(u, mu)
    lom = largest_of_maxima(u, mu)
    mom = mean_of_maxima(u, mu)
    assert som <= mom + 1e-9
    assert mom <= lom + 1e-9


def test_defuzzify_all_methods():
    u, mu = setup_symmetric_triangular()
    for method in ["centroid", "bisector", "mom", "som", "lom"]:
        val = defuzzify(u, mu, method=method)
        assert isinstance(val, float)
        assert 0.0 <= val <= 10.0


def test_defuzzify_invalid_method_raises():
    u, mu = setup_symmetric_triangular()
    try:
        defuzzify(u, mu, method="bogus")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_compare_methods_returns_summary():
    u, mu = setup_symmetric_triangular()
    result = compare_methods(u, mu)
    assert isinstance(result, dict)
    assert len(result) > 0
