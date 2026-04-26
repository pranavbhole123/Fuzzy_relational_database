import numpy as np

print("TEST MODULE: tests/test_operations.py loaded")

from core.operations import (
    t_norm, s_norm, TNorm, SNorm,
    complement, fuzzy_union as set_union, fuzzy_intersection as set_intersection,
    aggregate,
)
from tests.helpers import assert_in_range


A = np.array([0.0, 0.3, 0.5, 0.8, 1.0])
B = np.array([1.0, 0.6, 0.5, 0.4, 0.0])


def test_tnorm_minimum_identity():
    ones = np.ones_like(A)
    for tnorm in TNorm:
        assert np.allclose(t_norm(A, ones, tnorm), A)


def test_tnorm_commutativity():
    for tnorm in TNorm:
        assert np.allclose(t_norm(A, B, tnorm), t_norm(B, A, tnorm))


def test_tnorm_range():
    for tnorm in TNorm:
        assert_in_range(t_norm(A, B, tnorm))


def test_tnorm_minimum():
    assert np.allclose(t_norm(A, B, TNorm.MINIMUM), np.minimum(A, B))


def test_snorm_identity():
    zeros = np.zeros_like(A)
    for snorm in SNorm:
        assert np.allclose(s_norm(A, zeros, snorm), A)


def test_snorm_range():
    for snorm in SNorm:
        assert_in_range(s_norm(A, B, snorm))


def test_snorm_maximum():
    assert np.allclose(s_norm(A, B, SNorm.MAXIMUM), np.maximum(A, B))


def test_complement_is_one_minus():
    assert np.allclose(complement(A), 1.0 - A)


def test_set_union_and_intersection():
    assert np.allclose(set_union(A, B), np.maximum(A, B))
    assert np.allclose(set_intersection(A, B), np.minimum(A, B))


def test_aggregate_methods():
    memberships = [np.array([0.2, 0.4]), np.array([0.6, 0.8])]
    assert np.isclose(aggregate(memberships, method="mean"), 0.5)
    assert np.isclose(aggregate(memberships, method="min"), 0.2)
    assert np.isclose(aggregate(memberships, method="max"), 0.8)
