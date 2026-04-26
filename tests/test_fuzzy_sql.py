import pandas as pd

print("TEST MODULE: tests/test_fuzzy_sql.py loaded")

from core.membership_functions import make_mf
from core.fuzzy_sql import (
    FuzzyQuery, fuzzy_where, fuzzy_where_multi, fuzzy_threshold,
    fuzzy_between, fuzzy_inner_join, fuzzy_left_join,
    fuzzy_union, fuzzy_intersect, fuzzy_except,
    fuzzy_order_by, fuzzy_distinct, fuzzy_exists,
    fuzzy_group_by, fuzzy_aggregate,
)
from tests.helpers import EMPLOYEES


def setup_memberships():
    df = pd.DataFrame(EMPLOYEES)
    mf_young = make_mf("young", "triangular", {"a": 15, "b": 25, "c": 35}, 0, 100)
    mf_high = make_mf("high", "trapezoidal", {"a": 70000, "b": 90000, "c": 120000, "d": 150000}, 0, 200000)
    mf_senior = make_mf("senior", "s_shaped", {"a": 40, "b": 60}, 0, 100)
    return df, mf_young, mf_high, mf_senior


def test_fuzzy_where_adds_membership():
    df, mf_young, *_ = setup_memberships()
    result = fuzzy_where(df, "age", mf_young)
    assert "_membership" in result.columns
    assert result["_membership"].iloc[0] >= result["_membership"].iloc[-1]


def test_fuzzy_where_multi_and_or():
    df, mf_young, mf_high, _ = setup_memberships()
    result_and = fuzzy_where_multi(df, [("age", mf_young), ("salary", mf_high)], logic="AND")
    assert "_membership" in result_and.columns
    result_or = fuzzy_where_multi(df, [("age", mf_young), ("salary", mf_high)], logic="OR")
    assert "_membership" in result_or.columns


def test_fuzzy_threshold_filters():
    df, mf_young, *_ = setup_memberships()
    with_mem = fuzzy_where(df, "age", mf_young)
    filtered = fuzzy_threshold(with_mem, 0.5)
    assert filtered["_membership"].min() >= 0.5


def test_fuzzy_between_creates_membership():
    df, *_ = setup_memberships()
    result = fuzzy_between(df, "age", 20, 35, softness=0.2)
    assert "_membership" in result.columns


def test_fuzzy_join_functions():
    df, mf_young, *_ = setup_memberships()
    left = df[["name", "age"]].copy()
    right = df[["name", "salary"]].copy()
    inner = fuzzy_inner_join(left, right, on="name")
    assert isinstance(inner, pd.DataFrame)
    assert "_membership" in inner.columns
    left_join = fuzzy_left_join(left, right, on="name")
    assert len(left_join) >= len(left)


def test_fuzzy_set_operations_with_key():
    df, mf_young, _, mf_senior = setup_memberships()
    dfA = fuzzy_where(df, "age", mf_young)
    dfB = fuzzy_where(df, "age", mf_senior)
    union = fuzzy_union(dfA, dfB, key="name")
    intersect = fuzzy_intersect(dfA, dfB, key="name")
    assert "_membership" in union.columns
    assert "_membership" in intersect.columns


def test_fuzzy_except_and_order_by():
    df, mf_young, *_ = setup_memberships()
    dfA = fuzzy_where(df, "age", mf_young)
    dfB = dfA.head(3).copy()
    result = fuzzy_except(dfA, dfB, key="name")
    assert isinstance(result, pd.DataFrame)
    ordered = fuzzy_order_by(dfA, by="_membership", ascending=False)
    assert ordered["_membership"].iloc[0] >= ordered["_membership"].iloc[-1]


def test_fuzzy_distinct_exists_group_and_aggregate():
    df, mf_young, _, mf_senior = setup_memberships()
    df_mem = fuzzy_where(df, "age", mf_young)
    distinct = fuzzy_distinct(df_mem, columns=["age", "salary"], similarity_threshold=0.95)
    assert len(distinct) <= len(df_mem)
    groups = fuzzy_group_by(df_mem, "age", {"young": mf_young, "senior": mf_senior})
    assert "young" in groups
    agg = fuzzy_aggregate({"young": df_mem.head(2)}, agg_column="salary", funcs=["avg"])
    assert "group" in agg.columns
    assert "avg" in agg.columns
    exists_df = fuzzy_exists(df_mem, df_mem.head(1), key="name")
    assert "_membership" in exists_df.columns


def test_fuzzy_query_builder_top_k():
    df, mf_young, mf_high, _ = setup_memberships()
    result = FuzzyQuery(df).where("age", mf_young).threshold(0.3).top(3).execute()
    assert len(result) <= 3
    assert "_membership" in result.columns
