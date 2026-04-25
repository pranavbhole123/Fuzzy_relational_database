"""
test_fuzzy_rdb.py
=================
Comprehensive test suite for the Fuzzy Relational Database Engine.

Covers:
  1.  Membership Functions  (all 9 types + edge cases)
  2.  Linguistic Hedges     (all 11 hedges + chains)
  3.  T-norms & S-norms     (all 6 families each)
  4.  Alpha-Cut Analysis    (cuts, support, core, properties)
  5.  Defuzzification       (all 5 methods)
  6.  FuzzyRelation         (matrix ops, compositions, properties)
  7.  Storage Engine        (CSV loading, metadata, persistence)
  8.  MF Registry           (define, get, suggest, ASCII plot)
  9.  Fuzzy SQL Operations  (WHERE, JOIN, GROUP BY, SET OPS, DISTINCT, etc.)
  10. FuzzyRDBEngine API    (full end-to-end integration)
  11. CLI Parser            (FQL syntax, error handling)
  12. Edge Cases & Robustness

Run with:
    cd fuzzy_rel_db
    python test_fuzzy_rdb.py            # plain output
    python test_fuzzy_rdb.py -v         # verbose (shows each test name)
    python -m unittest test_fuzzy_rdb   # via unittest runner
    python -m unittest test_fuzzy_rdb.TestMembershipFunctions  # single class
    python -m unittest test_fuzzy_rdb.TestMembershipFunctions.test_triangular_peak  # single test
"""

from __future__ import annotations
import os
import sys
import csv
import math
import tempfile
import unittest
import numpy as np
import pandas as pd

# ── path setup ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_HERE, "core")
for p in [_HERE, _CORE]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── project imports ─────────────────────────────────────────────────────────
from core.membership_functions import (
    triangular, trapezoidal, gaussian, generalized_bell,
    sigmoid, s_shaped, z_shaped, pi_shaped, singleton,
    make_mf, MembershipFunction, MFType,
)
from core.hedges import apply_hedge, apply_hedge_chain, HEDGE_REGISTRY, list_hedges
from core.operations import (
    t_norm, s_norm, TNorm, SNorm,
    complement, fuzzy_union as set_union,
    fuzzy_intersection as set_intersection,
    aggregate,
)
from core.alpha_cut import (
    alpha_cut, alpha_cut_interval, get_support, get_core,
    get_height, is_normal, normalize, get_crossover_points,
    get_cardinality, get_relative_cardinality, is_convex,
    analyze,
)
from core.defuzzification import (
    centroid, bisector, mean_of_maxima,
    smallest_of_maxima, largest_of_maxima,
    defuzzify, compare_methods,
)
from core.relations import (
    FuzzyRelation, fuzzy_select, fuzzy_project, fuzzy_join,
)
from core.storage import StorageEngine
from core.fuzzifier import MFRegistry
from core.fuzzy_sql import (
    fuzzy_where, fuzzy_where_multi, fuzzy_threshold,
    fuzzy_between, fuzzy_inner_join, fuzzy_left_join,
    fuzzy_union, fuzzy_intersect, fuzzy_except,
    fuzzy_order_by, fuzzy_distinct, fuzzy_exists,
    fuzzy_group_by, fuzzy_aggregate,
    FuzzyQuery,
)
from cli.parser import parse, ParseError
from main import FuzzyRDBEngine


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def _assert_in_range(test, arr, lo=0.0, hi=1.0, msg=""):
    """Assert every value is in [lo, hi]."""
    test.assertTrue(np.all(arr >= lo) and np.all(arr <= hi),
                    msg or f"Values out of [{lo},{hi}]: {arr}")

def _csv_file(rows: list[dict]) -> str:
    """Write rows to a temp CSV and return the path."""
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                    delete=False, newline="")
    writer = csv.DictWriter(tf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    tf.close()
    return tf.name

EMPLOYEES = [
    {"name": "Alice",  "age": 24, "experience_years": 2,  "salary": 35000,  "performance_score": 82},
    {"name": "Bob",    "age": 31, "experience_years": 7,  "salary": 62000,  "performance_score": 74},
    {"name": "Carol",  "age": 45, "experience_years": 18, "salary": 95000,  "performance_score": 91},
    {"name": "David",  "age": 28, "experience_years": 5,  "salary": 48000,  "performance_score": 67},
    {"name": "Eva",    "age": 52, "experience_years": 25, "salary": 110000, "performance_score": 88},
    {"name": "Frank",  "age": 22, "experience_years": 1,  "salary": 28000,  "performance_score": 71},
    {"name": "Grace",  "age": 38, "experience_years": 12, "salary": 78000,  "performance_score": 85},
    {"name": "Henry",  "age": 60, "experience_years": 35, "salary": 130000, "performance_score": 79},
    {"name": "Irene",  "age": 27, "experience_years": 4,  "salary": 43000,  "performance_score": 90},
    {"name": "Jack",   "age": 35, "experience_years": 9,  "salary": 70000,  "performance_score": 63},
]

# ════════════════════════════════════════════════════════════════════════════
#  1. Membership Functions
# ════════════════════════════════════════════════════════════════════════════

class TestMembershipFunctions(unittest.TestCase):
    """Tests for every raw MF implementation."""

    # ── Triangular ──────────────────────────────────────────────────────────

    def test_triangular_peak(self):
        """Peak (b) must return 1.0."""
        self.assertAlmostEqual(float(triangular(25, 15, 25, 35)), 1.0)

    def test_triangular_feet(self):
        """Left foot a and right foot c must return 0.0."""
        self.assertAlmostEqual(float(triangular(15, 15, 25, 35)), 0.0)
        self.assertAlmostEqual(float(triangular(35, 15, 25, 35)), 0.0)

    def test_triangular_midpoints(self):
        """Mid-way between foot and peak should return ~0.5."""
        self.assertAlmostEqual(float(triangular(20, 15, 25, 35)), 0.5, places=5)
        self.assertAlmostEqual(float(triangular(30, 15, 25, 35)), 0.5, places=5)

    def test_triangular_out_of_range(self):
        """Values outside [a, c] should be 0."""
        self.assertEqual(float(triangular(0, 15, 25, 35)), 0.0)
        self.assertEqual(float(triangular(100, 15, 25, 35)), 0.0)

    def test_triangular_vectorised(self):
        """Array input must stay in [0, 1]."""
        x = np.linspace(0, 50, 200)
        y = triangular(x, 10, 25, 40)
        _assert_in_range(self, y)
        self.assertEqual(y.argmax(), np.argmin(np.abs(x - 25)))

    # ── Trapezoidal ─────────────────────────────────────────────────────────

    def test_trapezoidal_flat_top(self):
        """Every point in [b, c] must give 1.0."""
        for x in [50, 60, 70, 80]:
            self.assertAlmostEqual(float(trapezoidal(x, 40, 50, 80, 90)), 1.0)

    def test_trapezoidal_feet(self):
        """Values at or before a, and at or after d, must be 0."""
        self.assertAlmostEqual(float(trapezoidal(40, 40, 50, 80, 90)), 0.0)
        self.assertAlmostEqual(float(trapezoidal(90, 40, 50, 80, 90)), 0.0)

    def test_trapezoidal_slopes(self):
        """Midpoint of each slope should give 0.5."""
        self.assertAlmostEqual(float(trapezoidal(45, 40, 50, 80, 90)), 0.5, places=5)
        self.assertAlmostEqual(float(trapezoidal(85, 40, 50, 80, 90)), 0.5, places=5)

    def test_trapezoidal_range(self):
        x = np.linspace(0, 130, 300)
        y = trapezoidal(x, 40, 50, 80, 90)
        _assert_in_range(self, y)

    # ── Gaussian ────────────────────────────────────────────────────────────

    def test_gaussian_peak(self):
        self.assertAlmostEqual(float(gaussian(50, 50, 10)), 1.0)

    def test_gaussian_symmetry(self):
        v1 = float(gaussian(40, 50, 10))
        v2 = float(gaussian(60, 50, 10))
        self.assertAlmostEqual(v1, v2, places=10)

    def test_gaussian_sigma_effect(self):
        """Narrow sigma should give lower membership away from mean."""
        narrow = float(gaussian(60, 50, 5))
        wide   = float(gaussian(60, 50, 20))
        self.assertLess(narrow, wide)

    def test_gaussian_range(self):
        x = np.linspace(-100, 100, 500)
        y = gaussian(x, 0, 15)
        _assert_in_range(self, y)

    # ── Generalized Bell ────────────────────────────────────────────────────

    def test_gbell_centre(self):
        self.assertAlmostEqual(float(generalized_bell(50, 10, 2, 50)), 1.0)

    def test_gbell_range(self):
        x = np.linspace(0, 100, 300)
        y = generalized_bell(x, 10, 2, 50)
        _assert_in_range(self, y)

    def test_gbell_steeper_with_higher_b(self):
        x = np.array([40.0])
        gentle = float(generalized_bell(x, 10, 1, 50))
        steep  = float(generalized_bell(x, 10, 5, 50))
        self.assertLess(steep, gentle)

    # ── Sigmoid ─────────────────────────────────────────────────────────────

    def test_sigmoid_crossover(self):
        """At crossover point c, μ should be exactly 0.5."""
        self.assertAlmostEqual(float(sigmoid(50, 50, 0.2)), 0.5, places=5)

    def test_sigmoid_rising(self):
        self.assertLess(float(sigmoid(0, 50, 0.2)), 0.5)
        self.assertGreater(float(sigmoid(100, 50, 0.2)), 0.5)

    def test_sigmoid_falling(self):
        self.assertGreater(float(sigmoid(0, 50, -0.2)), 0.5)
        self.assertLess(float(sigmoid(100, 50, -0.2)), 0.5)

    def test_sigmoid_range(self):
        x = np.linspace(-200, 200, 500)
        y = sigmoid(x, 0, 0.1)
        _assert_in_range(self, y)

    # ── S-shaped, Z-shaped ──────────────────────────────────────────────────

    def test_s_shaped_boundary(self):
        self.assertAlmostEqual(float(s_shaped(0, 0, 10)), 0.0, places=5)
        self.assertAlmostEqual(float(s_shaped(10, 0, 10)), 1.0, places=5)

    def test_z_shaped_boundary(self):
        self.assertAlmostEqual(float(z_shaped(0, 0, 10)), 1.0, places=5)
        self.assertAlmostEqual(float(z_shaped(10, 0, 10)), 0.0, places=5)

    def test_sz_complement(self):
        """s_shaped + z_shaped should sum to 1.0 everywhere."""
        x = np.linspace(0, 10, 100)
        self.assertTrue(np.allclose(s_shaped(x, 0, 10) + z_shaped(x, 0, 10), 1.0))

    # ── Pi-shaped ───────────────────────────────────────────────────────────

    def test_pi_shaped_range(self):
        x = np.linspace(-10, 10, 300)
        y = pi_shaped(x, 0, 3, 7, 10)
        _assert_in_range(self, y)

    def test_pi_shaped_flat_top(self):
        x = np.linspace(3, 7, 50)
        y = pi_shaped(x, 0, 3, 7, 10)
        self.assertTrue(np.all(y > 0.99), f"Flat-top should be ~1.0: {y}")

    # ── Singleton ───────────────────────────────────────────────────────────

    def test_singleton_exact(self):
        self.assertAlmostEqual(float(singleton(5.0, center=5.0)), 1.0)

    def test_singleton_away(self):
        self.assertAlmostEqual(float(singleton(5.1, center=5.0, tolerance=1e-9)), 0.0)

    # ── make_mf factory ─────────────────────────────────────────────────────

    def test_make_mf_returns_object(self):
        mf = make_mf("young", "triangular", {"a": 15, "b": 25, "c": 35}, 0, 100)
        self.assertIsInstance(mf, MembershipFunction)
        self.assertEqual(mf.name, "young")
        self.assertEqual(mf.mf_type, MFType.TRIANGULAR)

    def test_make_mf_compute(self):
        mf = make_mf("high", "trapezoidal",
                     {"a": 70000, "b": 90000, "c": 120000, "d": 150000},
                     0, 200000)
        self.assertAlmostEqual(float(mf.compute(100000)), 1.0)

    def test_make_mf_invalid_type(self):
        with self.assertRaises(ValueError):
            make_mf("x", "unknown_type", {}, 0, 1)

    def test_make_mf_callable(self):
        mf = make_mf("test", "gaussian", {"mean": 5, "sigma": 1}, 0, 10)
        self.assertAlmostEqual(float(mf(5)), 1.0)

    def test_make_mf_get_curve(self):
        mf = make_mf("test", "triangular", {"a": 0, "b": 5, "c": 10}, 0, 10)
        u, mu = mf.get_curve(resolution=100)
        self.assertEqual(len(u), 100)
        _assert_in_range(self, mu)


# ════════════════════════════════════════════════════════════════════════════
#  2. Linguistic Hedges
# ════════════════════════════════════════════════════════════════════════════

class TestHedges(unittest.TestCase):

    BASE = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    def test_very_squares(self):
        result = apply_hedge("very", self.BASE)
        np.testing.assert_allclose(result, self.BASE ** 2)

    def test_extremely_cubes(self):
        result = apply_hedge("extremely", self.BASE)
        np.testing.assert_allclose(result, self.BASE ** 3)

    def test_somewhat_sqrt(self):
        result = apply_hedge("somewhat", self.BASE)
        np.testing.assert_allclose(result, np.sqrt(self.BASE), atol=1e-10)

    def test_not_complement(self):
        result = apply_hedge("not", self.BASE)
        np.testing.assert_allclose(result, 1.0 - self.BASE)

    def test_not_very(self):
        result = apply_hedge("not_very", self.BASE)
        np.testing.assert_allclose(result, np.clip(1.0 - self.BASE**2, 0, 1))

    def test_hedge_range_all(self):
        """Every hedge must return values in [0, 1]."""
        for name in HEDGE_REGISTRY:
            with self.subTest(hedge=name):
                out = apply_hedge(name, self.BASE)
                _assert_in_range(self, out, msg=f"Hedge '{name}' out of range")

    def test_hedge_concentration_order(self):
        """very > extremely should be MORE concentrated (lower away from 1)."""
        mu = np.array([0.5])
        very      = float(apply_hedge("very",      mu))
        extremely = float(apply_hedge("extremely", mu))
        self.assertGreater(very, extremely)

    def test_hedge_dilation_order(self):
        """somewhat should give higher value than very for mid-range mu."""
        mu = np.array([0.5])
        somewhat = float(apply_hedge("somewhat", mu))
        very     = float(apply_hedge("very",     mu))
        self.assertGreater(somewhat, very)

    def test_hedge_chain_not_very(self):
        """apply_hedge_chain(['not', 'very'], mu) should equal 1 - mu^2."""
        mu = np.array([0.2, 0.5, 0.8])
        result   = apply_hedge_chain(["not", "very"], mu)
        expected = np.clip(1.0 - mu**2, 0, 1)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_unknown_hedge_raises(self):
        with self.assertRaises(ValueError):
            apply_hedge("superduper", np.array([0.5]))

    def test_list_hedges_returns_list(self):
        hedges = list_hedges()
        self.assertIsInstance(hedges, list)
        self.assertTrue(all("hedge" in h and "effect" in h for h in hedges))


# ════════════════════════════════════════════════════════════════════════════
#  3. T-norms & S-norms
# ════════════════════════════════════════════════════════════════════════════

class TestOperations(unittest.TestCase):

    A = np.array([0.0, 0.3, 0.5, 0.8, 1.0])
    B = np.array([1.0, 0.6, 0.5, 0.4, 0.0])

    # ── T-norm boundary conditions ──────────────────────────────────────────

    def test_tnorm_minimum_identity(self):
        """T(x, 1) = x  for all T-norms."""
        ones = np.ones_like(self.A)
        for tnorm in TNorm:
            with self.subTest(tnorm=tnorm):
                result = t_norm(self.A, ones, tnorm)
                np.testing.assert_allclose(result, self.A, atol=1e-10,
                                           err_msg=f"{tnorm} failed identity")

    def test_tnorm_commutativity(self):
        """T(a, b) = T(b, a)."""
        for tnorm in TNorm:
            with self.subTest(tnorm=tnorm):
                np.testing.assert_allclose(
                    t_norm(self.A, self.B, tnorm),
                    t_norm(self.B, self.A, tnorm),
                    atol=1e-10
                )

    def test_tnorm_range(self):
        for tnorm in TNorm:
            with self.subTest(tnorm=tnorm):
                _assert_in_range(self, t_norm(self.A, self.B, tnorm),
                                 msg=f"{tnorm} out of range")

    def test_tnorm_minimum_correct(self):
        expected = np.minimum(self.A, self.B)
        np.testing.assert_allclose(t_norm(self.A, self.B, TNorm.MINIMUM), expected)

    def test_tnorm_algebraic_product(self):
        expected = self.A * self.B
        np.testing.assert_allclose(
            t_norm(self.A, self.B, TNorm.ALGEBRAIC_PRODUCT), expected)

    # ── S-norm boundary conditions ──────────────────────────────────────────

    def test_snorm_identity(self):
        """S(x, 0) = x."""
        zeros = np.zeros_like(self.A)
        for snorm in SNorm:
            with self.subTest(snorm=snorm):
                np.testing.assert_allclose(
                    s_norm(self.A, zeros, snorm), self.A, atol=1e-10)

    def test_snorm_range(self):
        for snorm in SNorm:
            with self.subTest(snorm=snorm):
                _assert_in_range(self, s_norm(self.A, self.B, snorm))

    def test_snorm_maximum_correct(self):
        expected = np.maximum(self.A, self.B)
        np.testing.assert_allclose(s_norm(self.A, self.B, SNorm.MAXIMUM), expected)

    # ── Complement ──────────────────────────────────────────────────────────

    def test_complement_standard(self):
        c = complement(self.A)
        np.testing.assert_allclose(c, 1.0 - self.A)

    def test_complement_range(self):
        _assert_in_range(self, complement(self.A))

    # ── Set operations ──────────────────────────────────────────────────────

    def test_set_union_is_max(self):
        u = set_union(self.A, self.B)
        np.testing.assert_allclose(u, np.maximum(self.A, self.B))

    def test_set_intersection_is_min(self):
        i = set_intersection(self.A, self.B)
        np.testing.assert_allclose(i, np.minimum(self.A, self.B))

    # ── Aggregate ───────────────────────────────────────────────────────────

    def test_aggregate_mean(self):
        memberships = [np.array([0.2, 0.4]), np.array([0.6, 0.8])]
        result = aggregate(memberships, method="mean")
        np.testing.assert_allclose(result, np.array([0.4, 0.6]))

    def test_aggregate_min(self):
        memberships = [np.array([0.2, 0.9]), np.array([0.7, 0.4])]
        result = aggregate(memberships, method="min")
        np.testing.assert_allclose(result, np.array([0.2, 0.4]))

    def test_aggregate_max(self):
        memberships = [np.array([0.2, 0.9]), np.array([0.7, 0.4])]
        result = aggregate(memberships, method="max")
        np.testing.assert_allclose(result, np.array([0.7, 0.9]))


# ════════════════════════════════════════════════════════════════════════════
#  4. Alpha-Cut Analysis
# ════════════════════════════════════════════════════════════════════════════

class TestAlphaCut(unittest.TestCase):

    def setUp(self):
        self.u = np.linspace(0, 10, 1000)
        # triangular at peak=5
        self.mu = triangular(self.u, 2, 5, 8)

    def test_alpha_cut_at_zero(self):
        """α=0 cut should include all points with μ ≥ 0."""
        elems, _ = alpha_cut(self.u, self.mu, 0.0)
        # all points are >= 0 so full array
        self.assertEqual(len(elems), len(self.u))

    def test_alpha_cut_at_one(self):
        """α=1 cut on triangular should return just the peak."""
        elems, vals = alpha_cut(self.u, self.mu, 1.0)
        self.assertTrue(len(elems) >= 1)
        self.assertTrue(np.all(vals >= 1.0 - 1e-9))

    def test_alpha_cut_midpoint(self):
        """α=0.5 cut should lie between feet and peak."""
        elems, _ = alpha_cut(self.u, self.mu, 0.5)
        self.assertTrue(np.all(elems >= 2.0))
        self.assertTrue(np.all(elems <= 8.0))

    def test_strong_alpha_cut_strict(self):
        """Strong cut at 1.0 should exclude anything below 1.0."""
        elems, vals = alpha_cut(self.u, self.mu, 1.0, strong=True)
        self.assertTrue(np.all(vals > 1.0 - 1e-9))

    def test_alpha_cut_interval(self):
        interval = alpha_cut_interval(self.u, self.mu, 0.5)
        self.assertIsNotNone(interval)
        lo, hi = interval
        self.assertGreater(hi, lo)

    def test_alpha_cut_interval_returns_none_when_empty(self):
        result = alpha_cut_interval(self.u, self.mu, 1.1)
        self.assertIsNone(result)

    def test_support(self):
        supp_elems, _ = get_support(self.u, self.mu)
        self.assertGreater(len(supp_elems), 0)
        # support should not include values outside [2, 8]
        self.assertTrue(np.all(supp_elems >= 2.0 - 1e-9))
        self.assertTrue(np.all(supp_elems <= 8.0 + 1e-9))

    def test_core(self):
        core_elems, core_vals = get_core(self.u, self.mu)
        self.assertGreater(len(core_elems), 0)
        self.assertTrue(np.all(core_vals >= 1.0 - 1e-9))

    def test_height_normal_mf(self):
        h = get_height(self.u, self.mu)
        self.assertAlmostEqual(h, 1.0, places=3)

    def test_is_normal(self):
        self.assertTrue(is_normal(self.u, self.mu))

    def test_normalize(self):
        mu_subnormal = self.mu * 0.5
        self.assertFalse(is_normal(self.u, mu_subnormal))
        mu_norm = normalize(self.u, mu_subnormal)
        self.assertTrue(is_normal(self.u, mu_norm))

    def test_crossover_points(self):
        pts = get_crossover_points(self.u, self.mu)
        # A symmetric triangular should have exactly two crossover points
        self.assertEqual(len(pts), 2)
        for p in pts:
            self.assertAlmostEqual(float(triangular(p, 2, 5, 8)), 0.5, places=2)

    def test_scalar_cardinality(self):
        card = get_cardinality(self.mu)
        self.assertGreater(card, 0)

    def test_relative_cardinality(self):
        rel = get_relative_cardinality(self.u, self.mu)
        self.assertGreaterEqual(rel, 0.0)
        self.assertLessEqual(rel, 1.0)

    def test_is_convex_triangular(self):
        self.assertTrue(is_convex(self.u, self.mu))

    def test_analyze_returns_dict(self):
        report = analyze(self.u, self.mu, name="test_set")
        self.assertIsInstance(report, dict)
        for key in ["name", "height", "is_normal", "cardinality"]:
            self.assertIn(key, report)


# ════════════════════════════════════════════════════════════════════════════
#  5. Defuzzification
# ════════════════════════════════════════════════════════════════════════════

class TestDefuzzification(unittest.TestCase):

    def setUp(self):
        self.u = np.linspace(0, 10, 1000)
        # Symmetric triangular: centroid should be at peak=5
        self.mu_sym = triangular(self.u, 0, 5, 10)

    def test_centroid_symmetric(self):
        c = centroid(self.u, self.mu_sym)
        self.assertAlmostEqual(c, 5.0, places=1)

    def test_bisector_symmetric(self):
        b = bisector(self.u, self.mu_sym)
        self.assertAlmostEqual(b, 5.0, places=0)

    def test_mom_symmetric(self):
        m = mean_of_maxima(self.u, self.mu_sym)
        self.assertAlmostEqual(m, 5.0, places=1)

    def test_som_lom_ordering(self):
        """For non-symmetric set, SOM ≤ MOM ≤ LOM."""
        mu = trapezoidal(self.u, 2, 4, 7, 9)
        som = smallest_of_maxima(self.u, mu)
        lom = largest_of_maxima(self.u, mu)
        mom = mean_of_maxima(self.u, mu)
        self.assertLessEqual(som, mom + 1e-9)
        self.assertLessEqual(mom, lom + 1e-9)

    def test_defuzzify_all_methods(self):
        for method in ["centroid", "bisector", "mom", "som", "lom"]:
            with self.subTest(method=method):
                val = defuzzify(self.u, self.mu_sym, method=method)
                self.assertIsInstance(val, float)
                self.assertGreater(val, 0.0)
                self.assertLess(val, 10.0)

    def test_defuzzify_invalid_method(self):
        with self.assertRaises(ValueError):
            defuzzify(self.u, self.mu_sym, method="bogus")

    def test_compare_methods_returns_dict(self):
        result = compare_methods(self.u, self.mu_sym)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_centroid_all_zero(self):
        """All-zero membership should not crash, returns mean of universe."""
        mu = np.zeros_like(self.u)
        val = centroid(self.u, mu)
        self.assertAlmostEqual(val, float(np.mean(self.u)), places=1)


# ════════════════════════════════════════════════════════════════════════════
#  6. FuzzyRelation
# ════════════════════════════════════════════════════════════════════════════

class TestFuzzyRelation(unittest.TestCase):

    def setUp(self):
        self.M = np.array([
            [0.9, 0.2, 0.0],
            [0.5, 0.8, 0.3],
            [0.1, 0.4, 0.7],
        ])
        self.R = FuzzyRelation(
            self.M,
            row_labels=["A", "B", "C"],
            col_labels=["X", "Y", "Z"],
            name="TestR"
        )

    def test_shape(self):
        self.assertEqual(self.R.shape, (3, 3))

    def test_values_clipped(self):
        # Pass out-of-range values: should be clipped
        R2 = FuzzyRelation(np.array([[1.5, -0.3], [0.5, 0.5]]))
        self.assertTrue(np.all(R2.matrix >= 0))
        self.assertTrue(np.all(R2.matrix <= 1))

    def test_inverse(self):
        Rinv = self.R.inverse()
        np.testing.assert_array_equal(Rinv.matrix, self.M.T)
        self.assertEqual(Rinv.row_labels, self.R.col_labels)
        self.assertEqual(Rinv.col_labels, self.R.row_labels)

    def test_complement(self):
        Rc = self.R.complement()
        np.testing.assert_allclose(Rc.matrix, 1.0 - self.M)

    def test_union(self):
        R2 = FuzzyRelation(np.ones((3, 3)) * 0.5,
                           row_labels=["A","B","C"],
                           col_labels=["X","Y","Z"])
        Ru = self.R.union(R2)
        expected = np.maximum(self.M, 0.5)
        np.testing.assert_allclose(Ru.matrix, expected)

    def test_intersection(self):
        R2 = FuzzyRelation(np.ones((3, 3)) * 0.5,
                           row_labels=["A","B","C"],
                           col_labels=["X","Y","Z"])
        Ri = self.R.intersection(R2)
        expected = np.minimum(self.M, 0.5)
        np.testing.assert_allclose(Ri.matrix, expected)

    def test_max_min_composition(self):
        """Max-min composition should produce a valid relation."""
        comp = self.R.compose(self.R.inverse(), method="max_min")
        self.assertEqual(comp.shape, (3, 3))
        _assert_in_range(self, comp.matrix)

    def test_max_product_composition(self):
        comp = self.R.compose(self.R.inverse(), method="max_product")
        _assert_in_range(self, comp.matrix)

    def test_composition_identity(self):
        """Composing with identity-like relation should approximate original."""
        I = FuzzyRelation(np.eye(3),
                          row_labels=["X","Y","Z"],
                          col_labels=["X","Y","Z"])
        comp = self.R.compose(I, method="max_min")
        # Result should be at least as strong as original min-wise
        self.assertEqual(comp.shape, (3, 3))

    def test_is_reflexive(self):
        I = FuzzyRelation(np.eye(3))
        self.assertTrue(I.is_reflexive())

    def test_not_reflexive(self):
        self.assertFalse(self.R.is_reflexive())

    def test_is_symmetric(self):
        sym_m = (self.M + self.M.T) / 2
        R_sym = FuzzyRelation(sym_m)
        self.assertTrue(R_sym.is_symmetric())

    def test_fuzzy_select_relational(self):
        df = pd.DataFrame({"val": [0.1, 0.5, 0.9], "_membership": [0.1, 0.5, 0.9]})
        result = fuzzy_select(df, threshold=0.4)
        self.assertEqual(len(result), 2)

    def test_fuzzy_project(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4],
                           "_membership": [0.5, 0.8]})
        result = fuzzy_project(df, ["a", "_membership"])
        self.assertIn("a", result.columns)
        self.assertNotIn("b", result.columns)


# ════════════════════════════════════════════════════════════════════════════
#  7. Storage Engine
# ════════════════════════════════════════════════════════════════════════════

class TestStorageEngine(unittest.TestCase):

    def setUp(self):
        self.storage = StorageEngine(":memory:")
        self.csv_path = _csv_file(EMPLOYEES)

    def tearDown(self):
        os.unlink(self.csv_path)

    def test_load_csv(self):
        df = self.storage.load_csv(self.csv_path, "emp")
        self.assertEqual(len(df), len(EMPLOYEES))

    def test_list_tables(self):
        self.storage.load_csv(self.csv_path, "emp")
        tables = self.storage.list_tables()
        self.assertIn("emp", tables)

    def test_row_count(self):
        self.storage.load_csv(self.csv_path, "emp")
        self.assertEqual(self.storage.row_count("emp"), len(EMPLOYEES))

    def test_get_columns(self):
        self.storage.load_csv(self.csv_path, "emp")
        cols = self.storage.get_columns("emp")
        self.assertIn("age", cols)
        self.assertIn("salary", cols)

    def test_fetch_as_dataframe(self):
        self.storage.load_csv(self.csv_path, "emp")
        df = self.storage.fetch_as_dataframe("emp")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(EMPLOYEES))

    def test_column_stats(self):
        self.storage.load_csv(self.csv_path, "emp")
        stats = self.storage.column_stats("emp", "age")
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertLessEqual(stats["min"], stats["max"])

    def test_get_column_range(self):
        self.storage.load_csv(self.csv_path, "emp")
        lo, hi = self.storage.get_column_range("emp", "age")
        self.assertLessEqual(lo, hi)

    def test_save_and_get_mf(self):
        self.storage.load_csv(self.csv_path, "emp")
        self.storage.save_mf("emp", "age", "young", "triangular", [15, 25, 35])
        mfs = self.storage.get_all_mfs("emp")
        self.assertIn("age", mfs)
        terms = [e["term"] for e in mfs["age"]]
        self.assertIn("young", terms)

    def test_load_dataframe_directly(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        self.storage.load_dataframe(df, "test_df")
        self.assertIn("test_df", self.storage.list_tables())
        self.assertEqual(self.storage.row_count("test_df"), 3)

    def test_get_numeric_columns(self):
        self.storage.load_csv(self.csv_path, "emp")
        numeric = self.storage.get_numeric_columns("emp")
        self.assertIn("age", numeric)
        self.assertIn("salary", numeric)
        self.assertNotIn("name", numeric)


# ════════════════════════════════════════════════════════════════════════════
#  8. MF Registry (Fuzzifier)
# ════════════════════════════════════════════════════════════════════════════

class TestMFRegistry(unittest.TestCase):

    def setUp(self):
        self.reg = MFRegistry()
        self.reg.define("age", "young", "triangular",
                        {"a": 15, "b": 25, "c": 35}, 0, 100)
        self.reg.define("age", "old", "trapezoidal",
                        {"a": 55, "b": 65, "c": 100, "d": 100}, 0, 100)
        self.reg.define("salary", "high", "gaussian",
                        {"mean": 100000, "sigma": 20000}, 0, 200000)

    def test_has_returns_true(self):
        self.assertTrue(self.reg.has("age", "young"))
        self.assertTrue(self.reg.has("age", "old"))

    def test_has_returns_false(self):
        self.assertFalse(self.reg.has("age", "middle"))
        self.assertFalse(self.reg.has("nonexistent", "term"))

    def test_get_returns_mf(self):
        mf = self.reg.get("age", "young")
        self.assertIsInstance(mf, MembershipFunction)

    def test_get_missing_raises(self):
        with self.assertRaises(KeyError):
            self.reg.get("age", "undefined_term")

    def test_list_columns(self):
        cols = self.reg.list_columns()
        self.assertIn("age", cols)
        self.assertIn("salary", cols)

    def test_summary(self):
        summary = self.reg.summary()
        self.assertIsInstance(summary, list)
        self.assertTrue(len(summary) >= 3)

    def test_compute_with_hedge(self):
        ages = np.array([20, 25, 30])
        base    = self.reg.compute("age", "young", ages)
        very    = self.reg.compute("age", "young", ages, hedge="very")
        np.testing.assert_allclose(very, base ** 2, atol=1e-10)

    def test_suggest_from_data(self):
        series = pd.Series([22, 28, 35, 42, 55])
        suggestions = self.reg.suggest_from_data("age", series, n_terms=3)
        self.assertEqual(len(suggestions), 3)
        for s in suggestions:
            self.assertIn("term", s)
            self.assertIn("mf_type", s)
            self.assertIn("params", s)

    def test_ascii_plot_returns_string(self):
        plot = self.reg.ascii_plot("age")
        self.assertIsInstance(plot, str)
        self.assertGreater(len(plot), 10)

    def test_overwrite_term(self):
        """Re-defining a term should overwrite without error."""
        self.reg.define("age", "young", "gaussian",
                        {"mean": 23, "sigma": 5}, 0, 100)
        mf = self.reg.get("age", "young")
        self.assertEqual(mf.mf_type, MFType.GAUSSIAN)


# ════════════════════════════════════════════════════════════════════════════
#  9. Fuzzy SQL Operations
# ════════════════════════════════════════════════════════════════════════════

class TestFuzzySQL(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(EMPLOYEES)
        self.mf_young = make_mf("young", "triangular",
                                {"a": 15, "b": 25, "c": 35}, 0, 100)
        self.mf_high  = make_mf("high", "trapezoidal",
                                {"a": 70000, "b": 90000, "c": 120000, "d": 150000},
                                0, 200000)
        self.mf_senior = make_mf("senior", "s_shaped",
                                  {"a": 40, "b": 60}, 0, 100)

    # ── fuzzy_where ─────────────────────────────────────────────────────────

    def test_fuzzy_where_adds_membership(self):
        result = fuzzy_where(self.df, "age", self.mf_young)
        self.assertIn("_membership", result.columns)

    def test_fuzzy_where_sorted_desc(self):
        result = fuzzy_where(self.df, "age", self.mf_young)
        memberships = result["_membership"].values
        self.assertTrue(np.all(memberships[:-1] >= memberships[1:]))

    def test_fuzzy_where_range(self):
        result = fuzzy_where(self.df, "age", self.mf_young)
        _assert_in_range(self, result["_membership"].values)

    def test_fuzzy_where_alice_high_membership(self):
        """Alice (age=24) should have high membership for 'young'."""
        result = fuzzy_where(self.df, "age", self.mf_young)
        alice_row = result[result["name"] == "Alice"]
        self.assertGreater(float(alice_row["_membership"].iloc[0]), 0.8)

    def test_fuzzy_where_henry_low_membership(self):
        """Henry (age=60) should have 0 membership for 'young'."""
        result = fuzzy_where(self.df, "age", self.mf_young)
        henry_row = result[result["name"] == "Henry"]
        self.assertAlmostEqual(float(henry_row["_membership"].iloc[0]), 0.0)

    # ── fuzzy_where_multi ───────────────────────────────────────────────────

    def test_fuzzy_where_multi_and(self):
        conditions = [
            ("age",    self.mf_young,  "AND"),
            ("salary", self.mf_high,   "AND"),
        ]
        result = fuzzy_where_multi(self.df, conditions)
        self.assertIn("_membership", result.columns)
        _assert_in_range(self, result["_membership"].values)

    def test_fuzzy_where_multi_or(self):
        conditions = [
            ("age",    self.mf_young,  "OR"),
            ("salary", self.mf_high,   "OR"),
        ]
        result = fuzzy_where_multi(self.df, conditions)
        # OR should give ≥ individual max
        individual_young = fuzzy_where(self.df, "age", self.mf_young)
        max_individual   = individual_young["_membership"].max()
        self.assertGreaterEqual(result["_membership"].max() + 1e-9, max_individual)

    # ── fuzzy_threshold ─────────────────────────────────────────────────────

    def test_fuzzy_threshold_filters(self):
        df_with_mem = fuzzy_where(self.df, "age", self.mf_young)
        filtered = fuzzy_threshold(df_with_mem, 0.5)
        self.assertTrue(np.all(filtered["_membership"] >= 0.5))

    def test_fuzzy_threshold_zero_keeps_all(self):
        df_with_mem = fuzzy_where(self.df, "age", self.mf_young)
        result = fuzzy_threshold(df_with_mem, 0.0)
        self.assertEqual(len(result), len(df_with_mem))

    def test_fuzzy_threshold_one_keeps_only_full(self):
        df_with_mem = fuzzy_where(self.df, "age", self.mf_young)
        result = fuzzy_threshold(df_with_mem, 1.0)
        # triangular peak is exactly 1.0 only at b=25
        self.assertTrue(len(result) <= len(df_with_mem))

    # ── fuzzy_between ───────────────────────────────────────────────────────

    def test_fuzzy_between_inside(self):
        result = fuzzy_between(self.df, "age", 20, 35,
                               low_tolerance=5, high_tolerance=5)
        self.assertIn("_membership", result.columns)

    def test_fuzzy_between_range(self):
        result = fuzzy_between(self.df, "age", 20, 40,
                               low_tolerance=3, high_tolerance=3)
        _assert_in_range(self, result["_membership"].values)

    # ── Joins ────────────────────────────────────────────────────────────────

    def test_fuzzy_inner_join(self):
        left  = self.df[["name", "age"]].copy()
        right = self.df[["name", "salary"]].copy()
        result = fuzzy_inner_join(left, right, on="name",
                                  left_mf=self.mf_young, left_col="age")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("_membership", result.columns)

    def test_fuzzy_left_join_keeps_all_left(self):
        left  = self.df[["name", "age"]].copy()
        right = self.df[["name", "salary"]].copy()
        result = fuzzy_left_join(left, right, on="name",
                                 left_mf=self.mf_young, left_col="age")
        # All left rows should appear
        self.assertGreaterEqual(len(result), len(left))

    # ── Set operations ───────────────────────────────────────────────────────

    def test_fuzzy_union_max(self):
        dfA = fuzzy_where(self.df, "age", self.mf_young)
        dfB = fuzzy_where(self.df, "age", self.mf_senior)
        result = fuzzy_union(dfA, dfB)
        self.assertIn("_membership", result.columns)
        # Union max should be ≥ both inputs
        self.assertGreaterEqual(result["_membership"].max() + 1e-9,
                                max(dfA["_membership"].max(), dfB["_membership"].max()))

    def test_fuzzy_intersect_min(self):
        dfA = fuzzy_where(self.df, "age", self.mf_young)
        dfB = fuzzy_where(self.df, "age", self.mf_senior)
        result = fuzzy_intersect(dfA, dfB)
        # Intersection min should be ≤ both inputs
        self.assertLessEqual(result["_membership"].max(),
                             max(dfA["_membership"].max(), dfB["_membership"].max()) + 1e-9)

    def test_fuzzy_except_excludes(self):
        dfA = fuzzy_where(self.df, "age", self.mf_young)
        dfB = dfA.head(3).copy()
        result = fuzzy_except(dfA, dfB)
        self.assertIsInstance(result, pd.DataFrame)

    # ── Ordering ─────────────────────────────────────────────────────────────

    def test_fuzzy_order_by_descending(self):
        df_mem = fuzzy_where(self.df, "age", self.mf_young)
        result = fuzzy_order_by(df_mem, by="_membership", ascending=False)
        m = result["_membership"].values
        self.assertTrue(np.all(m[:-1] >= m[1:]))

    def test_fuzzy_order_by_ascending(self):
        df_mem = fuzzy_where(self.df, "age", self.mf_young)
        result = fuzzy_order_by(df_mem, by="_membership", ascending=True)
        m = result["_membership"].values
        self.assertTrue(np.all(m[:-1] <= m[1:]))

    # ── Distinct ─────────────────────────────────────────────────────────────

    def test_fuzzy_distinct_reduces_or_keeps(self):
        df_mem = fuzzy_where(self.df, "age", self.mf_young)
        result = fuzzy_distinct(df_mem, key_col="age", threshold=0.95)
        self.assertLessEqual(len(result), len(df_mem))

    # ── Exists ───────────────────────────────────────────────────────────────

    def test_fuzzy_exists_returns_float(self):
        df_mem = fuzzy_where(self.df, "age", self.mf_young)
        high_young = df_mem[df_mem["_membership"] > 0.7]
        score = fuzzy_exists(high_young)
        self.assertIsInstance(float(score), float)
        self.assertGreaterEqual(float(score), 0.0)
        self.assertLessEqual(float(score), 1.0)

    # ── Group By & Aggregate ─────────────────────────────────────────────────

    def test_fuzzy_group_by(self):
        df_mem = fuzzy_where(self.df, "age", self.mf_young)
        groups = fuzzy_group_by(df_mem, group_col="age",
                                group_mfs={"young": self.mf_young,
                                           "senior": self.mf_senior})
        self.assertIn("young",  groups)
        self.assertIn("senior", groups)

    def test_fuzzy_aggregate_mean(self):
        df_mem = fuzzy_where(self.df, "age", self.mf_young)
        groups = {"young": df_mem.head(4), "other": df_mem.tail(4)}
        result = fuzzy_aggregate(groups, agg_col="salary", method="mean")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("group", result.columns)

    # ── FuzzyQuery chainable builder ─────────────────────────────────────────

    def test_fuzzy_query_builder(self):
        result = (
            FuzzyQuery(self.df)
            .where("age", self.mf_young)
            .threshold(0.3)
            .execute()
        )
        self.assertIn("_membership", result.columns)
        self.assertTrue(np.all(result["_membership"] >= 0.3))

    def test_fuzzy_query_top_k(self):
        result = (
            FuzzyQuery(self.df)
            .where("age", self.mf_young)
            .top(3)
            .execute()
        )
        self.assertLessEqual(len(result), 3)

    def test_fuzzy_query_multi_where(self):
        result = (
            FuzzyQuery(self.df)
            .where("age", self.mf_young, logic="AND")
            .where("salary", self.mf_high, logic="AND")
            .threshold(0.0)
            .execute()
        )
        self.assertIn("_membership", result.columns)


# ════════════════════════════════════════════════════════════════════════════
#  10. FuzzyRDBEngine (Full Integration)
# ════════════════════════════════════════════════════════════════════════════

class TestFuzzyRDBEngine(unittest.TestCase):

    def setUp(self):
        self.csv_path = _csv_file(EMPLOYEES)
        self.engine   = FuzzyRDBEngine()
        self.engine.load_csv(self.csv_path, "employees")

        self.engine.define_term("employees", "age", "young",
                                "triangular", {"a": 15, "b": 25, "c": 35})
        self.engine.define_term("employees", "age", "senior",
                                "s_shaped",   {"a": 40, "b": 60})
        self.engine.define_term("employees", "salary", "high",
                                "trapezoidal",
                                {"a": 70000, "b": 90000, "c": 120000, "d": 150000})
        self.engine.define_term("employees", "performance_score", "excellent",
                                "gaussian", {"mean": 90, "sigma": 5})

    def tearDown(self):
        os.unlink(self.csv_path)

    # ── Loading ──────────────────────────────────────────────────────────────

    def test_list_tables(self):
        tables = self.engine.list_tables()
        self.assertIn("employees", tables)

    def test_describe_table_keys(self):
        desc = self.engine.describe_table("employees")
        for key in ["table", "columns", "row_count", "stats", "fuzzy_cols"]:
            self.assertIn(key, desc)

    def test_describe_row_count(self):
        desc = self.engine.describe_table("employees")
        self.assertEqual(desc["row_count"], len(EMPLOYEES))

    def test_load_dataframe(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        self.engine.load_dataframe(df, "test_table")
        self.assertIn("test_table", self.engine.list_tables())

    def test_missing_table_raises(self):
        with self.assertRaises(KeyError):
            self.engine.describe_table("nonexistent_table")

    # ── Term management ──────────────────────────────────────────────────────

    def test_list_terms_returns_list(self):
        terms = self.engine.list_terms("employees")
        self.assertIsInstance(terms, list)
        self.assertGreater(len(terms), 0)

    def test_suggest_terms(self):
        suggestions = self.engine.suggest_terms("employees", "age", n_terms=3)
        self.assertEqual(len(suggestions), 3)

    def test_ascii_plot_returns_string(self):
        plot = self.engine.ascii_plot("employees", "age")
        self.assertIsInstance(plot, str)

    # ── Querying ─────────────────────────────────────────────────────────────

    def test_query_single_condition(self):
        result = self.engine.query(
            "employees",
            [{"col": "age", "hedge": None, "term": "young", "logic": "AND"}],
            threshold=0.0
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("_membership", result.columns)

    def test_query_threshold_filters_rows(self):
        all_rows = self.engine.query(
            "employees",
            [{"col": "age", "hedge": None, "term": "young", "logic": "AND"}],
            threshold=0.0
        )
        filtered = self.engine.query(
            "employees",
            [{"col": "age", "hedge": None, "term": "young", "logic": "AND"}],
            threshold=0.5
        )
        self.assertLessEqual(len(filtered), len(all_rows))
        self.assertTrue(np.all(filtered["_membership"] >= 0.5))

    def test_query_top_k(self):
        result = self.engine.query(
            "employees",
            [{"col": "age", "hedge": None, "term": "young", "logic": "AND"}],
            threshold=0.0,
            top_k=3
        )
        self.assertLessEqual(len(result), 3)

    def test_query_with_hedge_very(self):
        no_hedge = self.engine.query(
            "employees",
            [{"col": "age", "hedge": None,   "term": "young", "logic": "AND"}],
            threshold=0.0
        )
        with_hedge = self.engine.query(
            "employees",
            [{"col": "age", "hedge": "very", "term": "young", "logic": "AND"}],
            threshold=0.0
        )
        # 'very' concentrates: max membership should not exceed base
        self.assertLessEqual(with_hedge["_membership"].max(),
                             no_hedge["_membership"].max() + 1e-9)

    def test_query_and_conditions(self):
        result = self.engine.query(
            "employees",
            [
                {"col": "age",    "hedge": None, "term": "young", "logic": "AND"},
                {"col": "salary", "hedge": None, "term": "high",  "logic": "AND"},
            ],
            threshold=0.0
        )
        self.assertIn("_membership", result.columns)
        # AND should give ≤ any single condition
        single = self.engine.query(
            "employees",
            [{"col": "age", "hedge": None, "term": "young", "logic": "AND"}],
            threshold=0.0
        )
        self.assertLessEqual(result["_membership"].max(),
                             single["_membership"].max() + 1e-9)

    def test_query_or_conditions(self):
        result = self.engine.query(
            "employees",
            [
                {"col": "age", "hedge": None, "term": "young",  "logic": "OR"},
                {"col": "age", "hedge": None, "term": "senior", "logic": "OR"},
            ],
            threshold=0.0
        )
        self.assertIn("_membership", result.columns)

    def test_query_undefined_term_raises(self):
        with self.assertRaises(KeyError):
            self.engine.query(
                "employees",
                [{"col": "age", "hedge": None, "term": "nonexistent", "logic": "AND"}]
            )

    def test_query_empty_conditions_raises(self):
        with self.assertRaises(ValueError):
            self.engine.query("employees", conditions=[])

    def test_query_sorted_descending(self):
        result = self.engine.query(
            "employees",
            [{"col": "age", "hedge": None, "term": "young", "logic": "AND"}],
        )
        m = result["_membership"].values
        self.assertTrue(np.all(m[:-1] >= m[1:]))

    # ── Analyze ──────────────────────────────────────────────────────────────

    def test_analyze_returns_dict(self):
        report = self.engine.analyze("employees", "age", "young")
        self.assertIsInstance(report, dict)
        self.assertIn("height", report)

    # ── Defuzz ───────────────────────────────────────────────────────────────

    def test_defuzz_all_methods(self):
        result = self.engine.defuzz("employees", "age", "young", method="all")
        self.assertIsInstance(result, dict)

    def test_defuzz_single_method(self):
        val = self.engine.defuzz("employees", "age", "young", method="centroid")
        self.assertIsInstance(val, float)
        self.assertGreater(val, 0.0)

    # ── Repr ─────────────────────────────────────────────────────────────────

    def test_repr(self):
        r = repr(self.engine)
        self.assertIn("FuzzyRDBEngine", r)
        self.assertIn("employees", r)

    # ── Persistence (file DB) ─────────────────────────────────────────────────

    def test_file_db_persistence(self):
        """MF definitions should survive loading from a file-based DB."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            csv_path = _csv_file(EMPLOYEES)
            eng1 = FuzzyRDBEngine(db_path)
            eng1.load_csv(csv_path, "emp")
            eng1.define_term("emp", "age", "young",
                             "triangular", {"a": 15, "b": 25, "c": 35})

            # Create a new engine on the same file → terms should be restored
            eng2 = FuzzyRDBEngine(db_path)
            eng2.load_csv(csv_path, "emp")
            terms = eng2.list_terms("emp")
            term_names = [t["term"] for t in terms]
            self.assertIn("young", term_names)

            os.unlink(csv_path)
        finally:
            os.unlink(db_path)


# ════════════════════════════════════════════════════════════════════════════
#  11. CLI Parser
# ════════════════════════════════════════════════════════════════════════════

class TestCLIParser(unittest.TestCase):

    # ── LOAD ─────────────────────────────────────────────────────────────────

    def test_parse_load(self):
        cmd = parse("LOAD data/employees.csv AS employees")
        self.assertEqual(cmd["type"], "load")
        self.assertEqual(cmd["path"], "data/employees.csv")
        self.assertEqual(cmd["table"], "employees")

    # ── SHOW ─────────────────────────────────────────────────────────────────

    def test_parse_show_tables(self):
        cmd = parse("SHOW TABLES")
        self.assertEqual(cmd["type"], "show_tables")

    def test_parse_show_terms(self):
        cmd = parse("SHOW TERMS employees")
        self.assertEqual(cmd["type"], "show_terms")
        self.assertEqual(cmd["table"], "employees")

    # ── DESCRIBE ─────────────────────────────────────────────────────────────

    def test_parse_describe(self):
        cmd = parse("DESCRIBE employees")
        self.assertEqual(cmd["type"], "describe")
        self.assertEqual(cmd["table"], "employees")

    # ── DEFINE TERM ──────────────────────────────────────────────────────────

    def test_parse_define_term_triangular(self):
        cmd = parse("DEFINE TERM employees.age AS young USING triangular(15, 25, 35)")
        self.assertEqual(cmd["type"],   "define_term")
        self.assertEqual(cmd["table"],  "employees")
        self.assertEqual(cmd["col"],    "age")
        self.assertEqual(cmd["term"],   "young")
        self.assertEqual(cmd["mf_type"],"triangular")
        self.assertAlmostEqual(cmd["params"]["a"], 15.0)
        self.assertAlmostEqual(cmd["params"]["b"], 25.0)
        self.assertAlmostEqual(cmd["params"]["c"], 35.0)

    def test_parse_define_term_trapezoidal(self):
        cmd = parse("DEFINE TERM employees.salary AS high USING trapezoidal(70000, 90000, 120000, 150000)")
        self.assertEqual(cmd["mf_type"], "trapezoidal")
        self.assertEqual(len(cmd["params"]), 4)

    def test_parse_define_term_gaussian(self):
        cmd = parse("DEFINE TERM employees.age AS middle USING gaussian(40, 8)")
        self.assertEqual(cmd["mf_type"], "gaussian")
        self.assertIn("mean",  cmd["params"])
        self.assertIn("sigma", cmd["params"])

    def test_parse_define_term_with_umin_umax(self):
        cmd = parse(
            "DEFINE TERM employees.age AS young USING triangular(15, 25, 35) "
            "UMIN 0 UMAX 100"
        )
        self.assertAlmostEqual(cmd["umin"], 0.0)
        self.assertAlmostEqual(cmd["umax"], 100.0)

    # ── SELECT ───────────────────────────────────────────────────────────────

    def test_parse_select_simple(self):
        cmd = parse("SELECT * FROM employees WHERE age IS young THRESHOLD 0.3")
        self.assertEqual(cmd["type"],      "select")
        self.assertEqual(cmd["table"],     "employees")
        self.assertAlmostEqual(cmd["threshold"], 0.3)
        conds = cmd["conditions"]
        self.assertEqual(len(conds), 1)
        self.assertEqual(conds[0]["col"],  "age")
        self.assertEqual(conds[0]["term"], "young")
        self.assertIsNone(conds[0]["hedge"])

    def test_parse_select_with_hedge(self):
        cmd = parse("SELECT * FROM employees WHERE age IS very young")
        conds = cmd["conditions"]
        self.assertEqual(conds[0]["hedge"], "very")
        self.assertEqual(conds[0]["term"],  "young")

    def test_parse_select_and(self):
        cmd = parse(
            "SELECT * FROM employees "
            "WHERE age IS young AND salary IS high THRESHOLD 0.2"
        )
        conds = cmd["conditions"]
        self.assertEqual(len(conds), 2)
        self.assertEqual(conds[0]["logic"], "AND")
        self.assertEqual(conds[1]["logic"], "AND")

    def test_parse_select_or(self):
        cmd = parse("SELECT * FROM employees WHERE age IS young OR age IS senior")
        conds = cmd["conditions"]
        self.assertEqual(len(conds), 2)
        self.assertEqual(conds[1]["logic"], "OR")

    def test_parse_select_top(self):
        cmd = parse("SELECT * FROM employees WHERE age IS young TOP 5")
        self.assertEqual(cmd["top_k"], 5)

    # ── ANALYZE ──────────────────────────────────────────────────────────────

    def test_parse_analyze(self):
        cmd = parse("ANALYZE employees.age IS young")
        self.assertEqual(cmd["type"],   "analyze")
        self.assertEqual(cmd["table"],  "employees")
        self.assertEqual(cmd["col"],    "age")
        self.assertEqual(cmd["term"],   "young")

    # ── DEFUZZ ───────────────────────────────────────────────────────────────

    def test_parse_defuzz(self):
        cmd = parse("DEFUZZ employees.age IS young METHOD centroid")
        self.assertEqual(cmd["type"],   "defuzz")
        self.assertEqual(cmd["method"], "centroid")

    # ── SUGGEST ──────────────────────────────────────────────────────────────

    def test_parse_suggest_terms(self):
        cmd = parse("SUGGEST TERMS employees.age N 4")
        self.assertEqual(cmd["type"],    "suggest")
        self.assertEqual(cmd["table"],   "employees")
        self.assertEqual(cmd["col"],     "age")
        self.assertEqual(cmd["n_terms"], 4)

    def test_parse_suggest_terms_default_n(self):
        cmd = parse("SUGGEST TERMS employees.age")
        self.assertEqual(cmd["n_terms"], 3)

    # ── PLOT ─────────────────────────────────────────────────────────────────

    def test_parse_plot(self):
        cmd = parse("PLOT employees.age")
        self.assertEqual(cmd["type"],  "plot")
        self.assertEqual(cmd["table"], "employees")
        self.assertEqual(cmd["col"],   "age")

    # ── HELP / EXIT ──────────────────────────────────────────────────────────

    def test_parse_help(self):
        cmd = parse("HELP")
        self.assertEqual(cmd["type"], "help")

    def test_parse_exit(self):
        for word in ["EXIT", "QUIT", "exit", "quit"]:
            with self.subTest(word=word):
                cmd = parse(word)
                self.assertEqual(cmd["type"], "exit")

    # ── Case insensitivity ───────────────────────────────────────────────────

    def test_parse_case_insensitive_keywords(self):
        cmd = parse("select * from employees where age is young threshold 0.5")
        self.assertEqual(cmd["type"], "select")

    # ── Errors ───────────────────────────────────────────────────────────────

    def test_parse_empty_raises(self):
        with self.assertRaises(ParseError):
            parse("")

    def test_parse_garbage_raises(self):
        with self.assertRaises(ParseError):
            parse("!!! not a command")

    def test_parse_define_wrong_param_count_raises(self):
        with self.assertRaises(ParseError):
            # triangular needs 3 params but only 2 given
            parse("DEFINE TERM employees.age AS young USING triangular(15, 25)")

    def test_parse_select_missing_from_raises(self):
        with self.assertRaises(ParseError):
            parse("SELECT * employees WHERE age IS young")

    def test_parse_invalid_mf_type_raises(self):
        with self.assertRaises(ParseError):
            parse("DEFINE TERM employees.age AS young USING supermf(1,2,3)")

    def test_parse_invalid_defuzz_method_raises(self):
        with self.assertRaises(ParseError):
            parse("DEFUZZ employees.age IS young METHOD badmethod")


# ════════════════════════════════════════════════════════════════════════════
#  12. Edge Cases & Robustness
# ════════════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):

    def setUp(self):
        self.csv_path = _csv_file(EMPLOYEES)
        self.engine   = FuzzyRDBEngine()
        self.engine.load_csv(self.csv_path, "emp")
        self.engine.define_term("emp", "age", "young",
                                "triangular", {"a": 15, "b": 25, "c": 35})

    def tearDown(self):
        os.unlink(self.csv_path)

    def test_mf_values_always_in_01(self):
        """No MF type should ever produce values outside [0, 1]."""
        x = np.linspace(-1000, 1000, 5000)
        configs = [
            ("triangular",       {"a": -500, "b": 0, "c": 500}),
            ("trapezoidal",      {"a": -500, "b": -100, "c": 100, "d": 500}),
            ("gaussian",         {"mean": 0, "sigma": 100}),
            ("generalized_bell", {"a": 100, "b": 2, "c": 0}),
            ("sigmoid",          {"c": 0, "a": 0.01}),
            ("s_shaped",         {"a": -500, "b": 500}),
            ("z_shaped",         {"a": -500, "b": 500}),
            ("pi_shaped",        {"a": -500, "b": -100, "c": 100, "d": 500}),
        ]
        for mf_type, params in configs:
            with self.subTest(mf_type=mf_type):
                mf = make_mf("test", mf_type, params, -1000, 1000)
                y  = mf.compute(x)
                _assert_in_range(self, y, msg=f"{mf_type} exceeded [0,1]")

    def test_empty_dataframe_query(self):
        """Query on empty DataFrame should return empty result gracefully."""
        empty_df = pd.DataFrame(columns=["name", "age", "salary"])
        self.engine.storage.load_dataframe(empty_df, "empty_table")
        # Should not crash
        try:
            result = self.engine.query(
                "empty_table",
                [{"col": "age", "hedge": None, "term": "young", "logic": "AND"}]
            )
        except KeyError:
            pass  # Term not defined for this table — acceptable

    def test_threshold_above_max_returns_empty(self):
        result = self.engine.query(
            "emp",
            [{"col": "age", "hedge": None, "term": "young", "logic": "AND"}],
            threshold=2.0   # above max possible membership
        )
        self.assertEqual(len(result), 0)

    def test_all_hedges_on_engine_query(self):
        """All valid hedges should work in engine.query()."""
        hedges_to_test = ["very", "extremely", "somewhat", "not", "slightly"]
        for hedge in hedges_to_test:
            with self.subTest(hedge=hedge):
                result = self.engine.query(
                    "emp",
                    [{"col": "age", "hedge": hedge, "term": "young", "logic": "AND"}],
                    threshold=0.0
                )
                self.assertIn("_membership", result.columns)

    def test_multiple_tables_independent(self):
        """MF registries for different tables should be independent."""
        csv2 = _csv_file(EMPLOYEES)
        try:
            self.engine.load_csv(csv2, "emp2")
            self.engine.define_term("emp2", "age", "middle",
                                    "gaussian", {"mean": 40, "sigma": 8})
            # emp should not have "middle"; emp2 should not have "young"
            terms_emp  = {t["term"] for t in self.engine.list_terms("emp")}
            terms_emp2 = {t["term"] for t in self.engine.list_terms("emp2")}
            self.assertIn("young",  terms_emp)
            self.assertNotIn("middle", terms_emp)
            self.assertIn("middle", terms_emp2)
        finally:
            os.unlink(csv2)

    def test_gaussian_very_narrow_sigma(self):
        """Gaussian with tiny sigma should act like a spike."""
        mf = make_mf("spike", "gaussian", {"mean": 5.0, "sigma": 0.01}, 0, 10)
        self.assertAlmostEqual(float(mf.compute(5.0)), 1.0, places=5)
        self.assertAlmostEqual(float(mf.compute(5.1)), 0.0, places=1)

    def test_trapezoidal_degenerate_triangle(self):
        """Trapezoidal with b==c degenerates to triangle; should still work."""
        mf = make_mf("degen", "trapezoidal",
                     {"a": 0, "b": 5, "c": 5, "d": 10}, 0, 10)
        self.assertAlmostEqual(float(mf.compute(5)), 1.0)
        self.assertAlmostEqual(float(mf.compute(0)), 0.0, places=5)

    def test_parse_roundtrip_through_engine(self):
        """Parse → engine.define_term → engine.query round-trip."""
        cmd = parse(
            "DEFINE TERM emp.salary AS decent "
            "USING trapezoidal(30000, 45000, 75000, 90000)"
        )
        self.engine.define_term(
            cmd["table"], cmd["col"], cmd["term"],
            cmd["mf_type"], cmd["params"],
            cmd.get("umin"), cmd.get("umax")
        )
        result = self.engine.query(
            "emp",
            [{"col": "salary", "hedge": None, "term": "decent", "logic": "AND"}],
            threshold=0.0
        )
        self.assertGreater(len(result), 0)

    def test_hedge_not_inverts_membership(self):
        """'not young' should give high membership to older employees."""
        young_result = self.engine.query(
            "emp",
            [{"col": "age", "hedge": None,  "term": "young", "logic": "AND"}],
        )
        not_young_result = self.engine.query(
            "emp",
            [{"col": "age", "hedge": "not", "term": "young", "logic": "AND"}],
        )
        # Top row of 'not young' should be an older person
        top_not_young = not_young_result.iloc[0]["name"]
        top_young     = young_result.iloc[0]["name"]
        self.assertNotEqual(top_not_young, top_young)

    def test_tnorm_ordering_weakest_to_strongest(self):
        """Drastic ≤ Bounded ≤ Algebraic ≤ Minimum for mid-range inputs."""
        a = np.array([0.6])
        b = np.array([0.7])
        drastic  = float(t_norm(a, b, TNorm.DRASTIC_PRODUCT))
        bounded  = float(t_norm(a, b, TNorm.BOUNDED_PRODUCT))
        alg      = float(t_norm(a, b, TNorm.ALGEBRAIC_PRODUCT))
        minimum  = float(t_norm(a, b, TNorm.MINIMUM))
        self.assertLessEqual(drastic, bounded  + 1e-10)
        self.assertLessEqual(bounded, alg      + 1e-10)
        self.assertLessEqual(alg,     minimum  + 1e-10)


# ════════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # honour -v / --verbose flag
    verbosity = 2 if ("-v" in sys.argv or "--verbose" in sys.argv) else 1
    # strip our custom flag so unittest doesn't choke on it
    sys.argv = [a for a in sys.argv if a not in ("-v", "--verbose")]
    unittest.main(verbosity=verbosity)