"""
test_fuzzy_sql.py
=================
Demonstrates all fuzzy SQL operations against a sample employee dataset.
Run this to verify the engine works end to end.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from membership_functions import make_mf
from fuzzy_sql import (
    FuzzyQuery, fuzzy_where, fuzzy_where_multi,
    fuzzy_between, fuzzy_group_by, fuzzy_aggregate,
    fuzzy_having, fuzzy_inner_join, fuzzy_left_join,
    fuzzy_union, fuzzy_intersect, fuzzy_except,
    fuzzy_distinct, fuzzy_exists, fuzzy_in,
    compare_with_classical
)

DIVIDER = "─" * 60


def header(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


# ─────────────────────────────────────────────────────────────
#  Sample data
# ─────────────────────────────────────────────────────────────
employees = pd.DataFrame({
    "id"        : [1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
    "name"      : ["Alice","Bob","Carol","Dave","Eve",
                   "Frank","Grace","Hank","Iris","Jack"],
    "age"       : [24, 45, 31, 52, 27, 38, 61, 29, 47, 35],
    "salary"    : [35000, 90000, 55000, 105000, 40000,
                   72000, 120000, 42000, 88000, 67000],
    "experience": [2, 20, 8, 30, 3, 14, 38, 5, 22, 11],
    "dept"      : ["IT","HR","IT","Mgmt","IT",
                   "HR","Mgmt","IT","HR","IT"],
})

departments = pd.DataFrame({
    "dept"   : ["IT","HR","Mgmt"],
    "budget" : [500000, 300000, 800000],
    "size"   : [20, 12, 5],
})

# ─────────────────────────────────────────────────────────────
#  Membership functions
# ─────────────────────────────────────────────────────────────
mf_young    = make_mf("young",    "triangular",  {"a":15, "b":25, "c":35},   0, 100)
mf_middle   = make_mf("middle",   "triangular",  {"a":30, "b":42, "c":55},   0, 100)
mf_senior   = make_mf("senior",   "trapezoidal", {"a":50, "b":60, "c":100, "d":100}, 0, 100)

mf_low_sal  = make_mf("low",      "z_shaped",    {"a":40000, "b":65000},  0, 150000)
mf_mid_sal  = make_mf("medium",   "pi_shaped",   {"a":40000, "b":60000, "c":80000, "d":100000}, 0, 150000)
mf_high_sal = make_mf("high",     "s_shaped",    {"a":70000, "b":100000}, 0, 150000)

mf_junior   = make_mf("junior",   "z_shaped",    {"a":5,  "b":10},   0, 40)
mf_exp      = make_mf("experienced","s_shaped",  {"a":10, "b":20},   0, 40)


# ─────────────────────────────────────────────────────────────
#  TEST 1: Basic Fuzzy WHERE
# ─────────────────────────────────────────────────────────────
header("TEST 1: Fuzzy WHERE — age IS young")
result = fuzzy_where(employees, "age", mf_young)
print(result[["name","age","salary","_membership"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  TEST 2: Multi-condition AND
# ─────────────────────────────────────────────────────────────
header("TEST 2: WHERE age IS young AND experience IS junior")
result = fuzzy_where_multi(employees, [
    ("age",        mf_young),
    ("experience", mf_junior),
], logic="AND")
print(result[["name","age","experience","salary","_membership"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  TEST 3: Multi-condition OR
# ─────────────────────────────────────────────────────────────
header("TEST 3: WHERE age IS senior OR salary IS high")
result = fuzzy_where_multi(employees, [
    ("age",    mf_senior),
    ("salary", mf_high_sal),
], logic="OR")
print(result[["name","age","salary","_membership"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  TEST 4: Fuzzy BETWEEN
# ─────────────────────────────────────────────────────────────
header("TEST 4: Fuzzy BETWEEN salary 50000 AND 85000 (softness=0.15)")
result = fuzzy_between(employees, "salary", 50000, 85000, softness=0.15)
print(result[["name","salary","_membership"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  TEST 5: FuzzyQuery — chainable builder
# ─────────────────────────────────────────────────────────────
header("TEST 5: FuzzyQuery — age IS young AND salary IS medium, top 5")
q = (FuzzyQuery(employees)
     .where("age",    mf_young)
     .where("salary", mf_mid_sal, logic="AND")
     .threshold(0.1)
     .top(5))
print(q.explain())
print()
result = q.execute()
print(result[["name","age","salary","experience","_membership"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  TEST 6: Fuzzy GROUP BY + AGGREGATE
# ─────────────────────────────────────────────────────────────
header("TEST 6: Fuzzy GROUP BY age groups + weighted salary aggregate")
groups = fuzzy_group_by(employees, "age",
                         {"young": mf_young, "middle": mf_middle, "senior": mf_senior},
                         min_membership=0.1)

print("  Group sizes:")
for name, gdf in groups.items():
    print(f"    {name}: {len(gdf)} rows (membership ≥ 0.1)")

agg = fuzzy_aggregate(groups, "salary",
                       funcs=["count","fuzzy_count","avg","weighted_avg","min","max"])
print("\n  Aggregation result:")
print(agg.to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  TEST 7: Fuzzy HAVING
# ─────────────────────────────────────────────────────────────
header("TEST 7: HAVING weighted_avg salary IS high")
result = fuzzy_having(agg, "weighted_avg", mf_high_sal, threshold=0.3)
print(result[["group","weighted_avg","_having_membership"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  TEST 8: Fuzzy INNER JOIN (similarity-based)
# ─────────────────────────────────────────────────────────────
header("TEST 8: Fuzzy INNER JOIN employees × departments (exact key)")
result = fuzzy_inner_join(employees, departments, on="dept", tolerance=0.0)
print(result[["name","dept","budget_r","_membership"]].head(6).to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  TEST 9: Fuzzy LEFT JOIN
# ─────────────────────────────────────────────────────────────
header("TEST 9: Fuzzy LEFT JOIN — all employees matched to dept budget")
result = fuzzy_left_join(employees, departments, on="dept", tolerance=0.0)
print(result[["name","dept","budget_r","size_r","_membership"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  TEST 10: SET OPERATIONS — UNION
# ─────────────────────────────────────────────────────────────
header("TEST 10: Fuzzy UNION — (young employees) ∪ (high salary employees)")
young_set = fuzzy_where(employees.copy(), "age",    mf_young)
hsal_set  = fuzzy_where(employees.copy(), "salary", mf_high_sal)
result = fuzzy_union(young_set, hsal_set, key="id")
print(result[["name","age","salary","_membership"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  TEST 11: SET OPERATIONS — INTERSECT
# ─────────────────────────────────────────────────────────────
header("TEST 11: Fuzzy INTERSECT — young ∩ medium salary")
mid_set = fuzzy_where(employees.copy(), "salary", mf_mid_sal)
result  = fuzzy_intersect(young_set, mid_set, key="id")
print(result[["name","age","salary","_membership"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  TEST 12: SET OPERATIONS — EXCEPT
# ─────────────────────────────────────────────────────────────
header("TEST 12: Fuzzy EXCEPT — experienced NOT high salary")
exp_set = fuzzy_where(employees.copy(), "experience", mf_exp)
result  = fuzzy_except(exp_set, hsal_set, key="id")
print(result[["name","experience","salary","_membership"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  TEST 13: Fuzzy DISTINCT
# ─────────────────────────────────────────────────────────────
header("TEST 13: Fuzzy DISTINCT — remove near-duplicate age/salary rows")
result = fuzzy_distinct(employees, columns=["age","salary"],
                         similarity_threshold=0.92)
print(f"  Original rows: {len(employees)}, After fuzzy DISTINCT: {len(result)}")
print(result[["name","age","salary"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  TEST 14: Fuzzy IN
# ─────────────────────────────────────────────────────────────
header("TEST 14: Fuzzy IN — salary approximately IN [40000, 70000, 100000]")
result = fuzzy_in(employees, "salary", [40000, 70000, 100000], tolerance=10000)
print(result[["name","salary","_membership"]].to_string(index=False))


# ─────────────────────────────────────────────────────────────
#  TEST 15: THE VIVA PROOF — compare_with_classical
# ─────────────────────────────────────────────────────────────
header("TEST 15: VIVA PROOF — Fuzzy vs Classical SQL")

fuzzy_q = (FuzzyQuery(employees)
           .where("age",    mf_young)
           .where("salary", mf_mid_sal, logic="AND")
           .threshold(0.2)
           .execute())

comparison = compare_with_classical(
    df=employees,
    fuzzy_result=fuzzy_q,
    classical_conditions={"age": ("<=", 30), "salary": (">=", 35000)},
    id_col="id"
)

print(f"\n  Classical SQL found  : {comparison['classical_count']} rows")
print(f"  Fuzzy query found   : {comparison['fuzzy_count']} rows (ranked)")
print(f"\n  {comparison['fuzzy_advantage']}")
print("\n  Rows ONLY fuzzy found (classical missed these):")
if len(comparison["only_in_fuzzy"]):
    print(comparison["only_in_fuzzy"][["name","age","salary","_membership"]].to_string(index=False))
else:
    print("  (none — classical was equally broad for this dataset)")
print("\n  Classical result (binary — no ranking):")
print(comparison["classical_result"][["name","age","salary"]].to_string(index=False))
print("\n  Fuzzy result (ranked — borderline cases visible):")
print(comparison["fuzzy_result"][["name","age","salary","_membership"]].to_string(index=False))