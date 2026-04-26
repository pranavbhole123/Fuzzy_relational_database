import numpy as np
import pandas as pd

print("TEST MODULE: tests/test_fuzzy_relation.py loaded")

from core.relations import FuzzyRelation, fuzzy_select, fuzzy_project


def test_fuzzy_relation_basic_properties():
    M = np.array([
        [0.9, 0.2, 0.0],
        [0.5, 0.8, 0.3],
        [0.1, 0.4, 0.7],
    ])
    R = FuzzyRelation(M, row_labels=["A", "B", "C"], col_labels=["X", "Y", "Z"], name="TestR")
    assert R.shape == (3, 3)
    assert np.all(R.matrix >= 0.0)
    assert np.all(R.matrix <= 1.0)


def test_fuzzy_relation_inverse_and_complement():
    M = np.array([[0.1, 0.9], [0.7, 0.4]])
    R = FuzzyRelation(M, row_labels=["A", "B"], col_labels=["X", "Y"])
    Ri = R.inverse()
    assert np.array_equal(Ri.matrix, M.T)
    Rc = R.complement()
    assert np.array_equal(Rc.matrix, 1.0 - M)


def test_fuzzy_relation_union_and_intersection():
    M1 = np.array([[0.2, 0.8], [0.5, 0.4]])
    M2 = np.full((2, 2), 0.5)
    R1 = FuzzyRelation(M1)
    R2 = FuzzyRelation(M2)
    assert np.array_equal(R1.union(R2).matrix, np.maximum(M1, M2))
    assert np.array_equal(R1.intersection(R2).matrix, np.minimum(M1, M2))


def test_fuzzy_relation_composition_methods():
    M = np.array([[1.0, 0.0], [0.0, 1.0]])
    R = FuzzyRelation(M)
    comp_mm = R.compose(R, method="max_min")
    comp_mp = R.compose(R, method="max_product")
    assert comp_mm.shape == (2, 2)
    assert comp_mp.shape == (2, 2)
    assert np.all(comp_mm.matrix >= 0.0)
    assert np.all(comp_mp.matrix >= 0.0)


def test_fuzzy_select_and_project():
    df = pd.DataFrame({"val": [0.1, 0.5, 0.9], "_membership": [0.1, 0.5, 0.9]})
    selected = fuzzy_select(df, "_membership", threshold=0.4)
    assert len(selected) == 2
    projected = fuzzy_project(df, ["val", "_membership"])
    assert list(projected.columns) == ["val", "_membership"]
