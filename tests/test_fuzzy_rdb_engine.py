import os
import pandas as pd

print("TEST MODULE: tests/test_fuzzy_rdb_engine.py loaded")

from main import FuzzyRDBEngine
from tests.helpers import EMPLOYEES, _csv_file


def setup_engine():
    csv_path = _csv_file(EMPLOYEES)
    engine = FuzzyRDBEngine()
    engine.load_csv(csv_path, "employees")
    engine.define_term("employees", "age", "young", "triangular", {"a": 15, "b": 25, "c": 35})
    engine.define_term("employees", "age", "senior", "s_shaped", {"a": 40, "b": 60})
    engine.define_term("employees", "salary", "high", "trapezoidal", {"a": 70000, "b": 90000, "c": 120000, "d": 150000})
    return engine, csv_path


def test_engine_list_tables_and_describe():
    engine, csv_path = setup_engine()
    try:
        assert "employees" in engine.list_tables()
        desc = engine.describe_table("employees")
        assert desc["row_count"] == len(EMPLOYEES)
    finally:
        engine.storage.close()
        os.unlink(csv_path)


def test_engine_query_threshold_and_top_k():
    engine, csv_path = setup_engine()
    try:
        result_all = engine.query("employees", [{"col": "age", "hedge": None, "term": "young", "logic": "AND"}], threshold=0.0)
        result_top = engine.query("employees", [{"col": "age", "hedge": None, "term": "young", "logic": "AND"}], threshold=0.0, top_k=3)
        assert len(result_top) <= 3
        assert "_membership" in result_all.columns
    finally:
        engine.storage.close()
        os.unlink(csv_path)


def test_engine_suggest_and_ascii_plot():
    engine, csv_path = setup_engine()
    try:
        suggestions = engine.suggest_terms("employees", "age", n_terms=3)
        assert len(suggestions) == 3
        assert isinstance(engine.ascii_plot("employees", "age"), str)
    finally:
        engine.storage.close()
        os.unlink(csv_path)


def test_engine_defuzz_and_analyze():
    engine, csv_path = setup_engine()
    try:
        analysis = engine.analyze("employees", "age", "young")
        assert isinstance(analysis, dict)
        result = engine.defuzz("employees", "age", "young", method="centroid")
        assert isinstance(result, float)
    finally:
        engine.storage.close()
        os.unlink(csv_path)


def test_engine_persistence_loads_saved_mfs():
    csv_path = _csv_file(EMPLOYEES)
    db_path = os.path.join(os.path.dirname(csv_path), "temp_engine.db")
    try:
        engine = FuzzyRDBEngine(db_path)
        engine.load_csv(csv_path, "employees")
        engine.define_term("employees", "age", "young", "triangular", {"a": 15, "b": 25, "c": 35})
        engine.define_term("employees", "salary", "high", "trapezoidal", {"a": 70000, "b": 90000, "c": 120000, "d": 150000})
        engine2 = FuzzyRDBEngine(db_path)
        engine2.load_csv(csv_path, "employees")
        terms = [item["term"] for item in engine2.list_terms("employees")]
        assert "young" in terms
        assert "high" in terms
    finally:
        engine.storage.close()
        engine2.storage.close()
        if os.path.exists(db_path):
            os.unlink(db_path)
        os.unlink(csv_path)
