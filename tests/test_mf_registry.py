import numpy as np
import pandas as pd

print("TEST MODULE: tests/test_mf_registry.py loaded")

from core.fuzzifier import MFRegistry
from core.membership_functions import MembershipFunction, MFType


def test_mf_registry_define_and_lookup():
    reg = MFRegistry()
    reg.define("age", "young", "triangular", {"a": 15, "b": 25, "c": 35}, 0, 100)
    reg.define("age", "old", "trapezoidal", {"a": 55, "b": 65, "c": 100, "d": 100}, 0, 100)
    reg.define("salary", "high", "gaussian", {"mean": 100000, "sigma": 20000}, 0, 200000)

    assert reg.has("age", "young")
    assert not reg.has("age", "middle")
    mf = reg.get("age", "young")
    assert mf.name == "young"
    assert mf.mf_type.value == "triangular"


def test_mf_registry_compute_with_hedge():
    reg = MFRegistry()
    reg.define("age", "young", "triangular", {"a": 15, "b": 25, "c": 35}, 0, 100)
    ages = np.array([20, 25, 30])
    base = reg.compute("age", "young", ages)
    very = reg.compute("age", "young", ages, hedge="very")
    assert np.allclose(very, base ** 2)


def test_mf_registry_suggest_from_data():
    reg = MFRegistry()
    series = pd.Series([22, 28, 35, 42, 55])
    suggestions = reg.suggest_from_data("age", series, n_terms=3)
    assert len(suggestions) == 3
    assert all("term" in s and "mf_type" in s and "params" in s for s in suggestions)


def test_mf_registry_ascii_plot_returns_text():
    reg = MFRegistry()
    reg.define("age", "young", "triangular", {"a": 15, "b": 25, "c": 35}, 0, 100)
    plot = reg.ascii_plot("age")
    assert isinstance(plot, str)
    assert len(plot) > 10
