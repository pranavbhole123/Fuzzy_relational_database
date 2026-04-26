import csv
import tempfile
from typing import List

import numpy as np

print("TEST MODULE: tests/helpers.py loaded")

EMPLOYEES = [
    {"name": "Alice",  "age": 24, "experience_years": 2,  "salary": 35000,  "performance_score": 82},
    {"name": "Bob",    "age": 31, "experience_years": 7,  "salary": 62000,  "performance_score": 74},
    {"name": "Carol",  "age": 45, "experience_years": 18, "salary": 95000,  "performance_score": 91},
    {"name": "David",  "age": 28, "experience_years": 5,  "salary": 48000,  "performance_score": 67},
    {"name": "Eva",    "age": 52, "experience_years": 25, "salary": 110000, "performance_score": 88},
    {"name": "Frank",  "age": 22, "experience_years": 1,  "salary": 28000, "performance_score": 71},
    {"name": "Grace",  "age": 38, "experience_years": 12, "salary": 78000, "performance_score": 85},
    {"name": "Henry",  "age": 60, "experience_years": 35, "salary": 130000, "performance_score": 79},
    {"name": "Irene",  "age": 27, "experience_years": 4,  "salary": 43000, "performance_score": 90},
    {"name": "Jack",   "age": 35, "experience_years": 9,  "salary": 70000, "performance_score": 63},
]

def _csv_file(rows: List[dict]) -> str:
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    writer = csv.DictWriter(tf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    tf.close()
    return tf.name


def assert_in_range(arr, lo=0.0, hi=1.0):
    arr = np.asarray(arr, dtype=float)
    assert np.all(arr >= lo) and np.all(arr <= hi), f"Values out of [{lo},{hi}]: {arr}"
