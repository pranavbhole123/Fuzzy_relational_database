import numpy as np

print("TEST MODULE: tests/test_alpha_cut.py loaded")

from core.alpha_cut import (
    alpha_cut, alpha_cut_interval, get_support, get_core,
    get_height, is_normal, normalize, get_crossover_points,
    get_cardinality, get_relative_cardinality, is_convex,
    analyze,
)
from core.membership_functions import triangular


def setup_triangular():
    u = np.linspace(0, 10, 1001)
    mu = triangular(u, 2, 5, 8)
    return u, mu


def test_alpha_cut_at_zero_returns_all_points():
    u, mu = setup_triangular()
    elems, vals = alpha_cut(u, mu, 0.0)
    assert len(elems) == len(u)
    assert np.all(vals >= 0.0)


def test_alpha_cut_at_one_returns_peak():
    u, mu = setup_triangular()
    elems, vals = alpha_cut(u, mu, 1.0)
    assert len(elems) >= 1
    assert np.all(vals >= 1.0 - 1e-9)


def test_support_and_core():
    u, mu = setup_triangular()
    supp, _ = get_support(u, mu)
    core, cvals = get_core(u, mu)
    assert np.all(supp >= 2.0 - 1e-9)
    assert np.all(supp <= 8.0 + 1e-9)
    assert np.all(cvals >= 1.0 - 1e-9)


def test_normality_and_normalize():
    u, mu = setup_triangular()
    assert is_normal(mu)
    mu_sub = mu * 0.5
    assert not is_normal(mu_sub)
    mu_norm = normalize(mu_sub)
    assert is_normal(mu_norm)


def test_crossover_points_and_convexity():
    u, mu = setup_triangular()
    pts = get_crossover_points(u, mu)
    assert len(pts) == 2
    assert is_convex(u, mu)


def test_cardinality_and_relative_cardinality():
    u, mu = setup_triangular()
    assert get_cardinality(mu) > 0
    rel = get_relative_cardinality(mu)
    assert 0.0 <= rel <= 1.0


def test_analyze_reports_expected_fields():
    u, mu = setup_triangular()
    report = analyze(u, mu, name="test_set")
    assert isinstance(report, dict)
    assert report["name"] == "test_set"
    assert "height" in report
    assert "is_normal" in report
    assert "cardinality" in report
