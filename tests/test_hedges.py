import numpy as np

print("TEST MODULE: tests/test_hedges.py loaded")

from core.hedges import apply_hedge, apply_hedge_chain, HEDGE_REGISTRY, list_hedges
from tests.helpers import assert_in_range


def test_apply_known_hedges():
    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(apply_hedge("very", values), values ** 2)
    assert np.allclose(apply_hedge("extremely", values), values ** 3)
    assert np.allclose(apply_hedge("somewhat", values), np.sqrt(values))
    assert np.allclose(apply_hedge("not", values), 1.0 - values)


def test_apply_hedge_chain():
    values = np.array([0.2, 0.5, 0.8])
    result = apply_hedge_chain(["not", "very"], values)
    expected = np.clip(1.0 - values**2, 0, 1)
    assert np.allclose(result, expected)


def test_all_hedges_stay_in_range():
    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    for hedge in HEDGE_REGISTRY:
        assert_in_range(apply_hedge(hedge, values))


def test_list_hedges_returns_descriptions():
    hedges = list_hedges()
    assert isinstance(hedges, list)
    assert all("hedge" in h and "effect" in h for h in hedges)
