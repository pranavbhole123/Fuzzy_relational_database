import os
import numpy as np
import pandas as pd
import pytest

print("TEST MODULE: tests/test_edge_cases.py loaded")

from core.membership_functions import make_mf
from cli.parser import parse, ParseError
from main import FuzzyRDBEngine
from tests.helpers import EMPLOYEES, _csv_file


def test_all_mf_types_stay_in_unit_interval():
    x = np.linspace(-1000, 1000, 5000)
    configs = [
        ("triangular", {"a": -500, "b": 0, "c": 500}),
        ("trapezoidal", {"a": -500, "b": -100, "c": 100, "d": 500}),
        ("gaussian", {"mean": 0, "sigma": 100}),
        ("generalized_bell", {"a": 100, "b": 2, "c": 0}),
        ("sigmoid", {"c": 0, "a": 0.01}),
        ("s_shaped", {"a": -500, "b": 500}),
        ("z_shaped", {"a": -500, "b": 500}),
        ("pi_shaped", {"a": -500, "b": -100, "c": 100, "d": 500}),
    ]
    for mf_type, params in configs:
        mf = make_mf("test", mf_type, params, -1000, 1000)
        values = mf.compute(x)
        assert np.all(values >= 0.0) and np.all(values <= 1.0)


def test_empty_dataframe_query_returns_empty():
    csv_path = _csv_file(EMPLOYEES)
    engine = FuzzyRDBEngine()
    try:
        engine.load_csv(csv_path, "emp")
        empty_df = pd.DataFrame(columns=["name", "age", "salary"])
        engine.storage.load_dataframe(empty_df, "empty_table")
        result = engine.query("empty_table", [{"col": "age", "hedge": None, "term": "young", "logic": "AND"}])
        assert result.empty
    finally:
        os.unlink(csv_path)


def test_query_threshold_above_max_returns_empty():
    csv_path = _csv_file(EMPLOYEES)
    engine = FuzzyRDBEngine()
    try:
        engine.load_csv(csv_path, "emp")
        engine.define_term("emp", "age", "young", "triangular", {"a": 15, "b": 25, "c": 35})
        result = engine.query("emp", [{"col": "age", "hedge": None, "term": "young", "logic": "AND"}], threshold=2.0)
        assert len(result) == 0
    finally:
        os.unlink(csv_path)


def test_parse_invalid_syntax_raises():
    with pytest.raises(ParseError):
        parse("SELECT * employees WHERE age IS young")
