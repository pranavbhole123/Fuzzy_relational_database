import os
import pandas as pd

print("TEST MODULE: tests/test_storage_engine.py loaded")

from core.storage import StorageEngine
from tests.helpers import EMPLOYEES, _csv_file


def test_storage_engine_load_csv_and_list_tables():
    storage = StorageEngine(":memory:")
    csv_path = _csv_file(EMPLOYEES)
    try:
        df = storage.load_csv(csv_path, "emp")
        assert len(df) == len(EMPLOYEES)
        assert "emp" in storage.list_tables()
    finally:
        os.unlink(csv_path)


def test_storage_engine_row_count_and_columns():
    storage = StorageEngine(":memory:")
    csv_path = _csv_file(EMPLOYEES)
    try:
        storage.load_csv(csv_path, "emp")
        assert storage.row_count("emp") == len(EMPLOYEES)
        cols = storage.get_columns("emp")
        assert "age" in cols
        assert "salary" in cols
    finally:
        os.unlink(csv_path)


def test_fetch_as_dataframe_and_numeric_columns():
    storage = StorageEngine(":memory:")
    csv_path = _csv_file(EMPLOYEES)
    try:
        storage.load_csv(csv_path, "emp")
        df = storage.fetch_as_dataframe("emp")
        assert isinstance(df, pd.DataFrame)
        assert storage.get_column_range("emp", "age")[0] <= storage.get_column_range("emp", "age")[1]
        numeric = storage.get_numeric_columns("emp")
        assert "age" in numeric
        assert "salary" in numeric
    finally:
        os.unlink(csv_path)


def test_save_and_get_mf_metadata():
    storage = StorageEngine(":memory:")
    csv_path = _csv_file(EMPLOYEES)
    try:
        storage.load_csv(csv_path, "emp")
        storage.save_mf("emp", "age", "young", "triangular", [15, 25, 35])
        mfs = storage.get_all_mfs("emp")
        assert "age" in mfs
        assert any(item["term"] == "young" for item in mfs["age"])
    finally:
        os.unlink(csv_path)
