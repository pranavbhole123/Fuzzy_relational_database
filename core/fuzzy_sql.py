"""
fuzzy_sql.py
============
Fuzzy extensions of all major relational SQL operations.

Every operation returns a DataFrame that always carries a
'_membership' column — the degree to which each row satisfies
the query.  Results are always ranked (highest membership first).

Operations implemented
----------------------
DML / Query
  FuzzyQuery            — chainable query builder (SELECT / WHERE / HAVING /
                          ORDER BY / LIMIT / THRESHOLD)
Filtering
  fuzzy_where           — WHERE col IS <linguistic_term>  (single condition)
  fuzzy_where_multi     — multi-condition WHERE with AND / OR / MIXED
  fuzzy_threshold       — keep rows above a minimum membership
  fuzzy_between         — crisp BETWEEN extended to soft boundaries
Joins  (5 types)
  fuzzy_inner_join      — only well-matching pairs survive
  fuzzy_left_join       — all left rows, matched to best right partner
  fuzzy_right_join      — mirror of left
  fuzzy_full_join       — all rows from both sides
  fuzzy_similarity_join — join on fuzzy equality of a numeric column
Grouping & Aggregation
  fuzzy_group_by        — soft grouping (row can belong to multiple groups)
  fuzzy_aggregate       — weighted AVG / SUM / COUNT / MIN / MAX per group
  fuzzy_having          — post-group filter on aggregate membership
Set Operations
  fuzzy_union           — max(μA, μB)
  fuzzy_intersect       — min(μA, μB)
  fuzzy_except          — rows in A not matched in B
Ordering & Deduplication
  fuzzy_order_by        — rank by membership (default) or any column
  fuzzy_distinct        — remove near-duplicate rows above similarity threshold
Subquery helpers
  fuzzy_exists          — degree to which a match exists in a sub-result
  fuzzy_in              — degree of membership in a fuzzy set of values
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

# Internal imports — these live in the same fuzzy/ package
from membership_functions import MembershipFunction, make_mf
from operations import aggregate as fuzzy_agg, TNorm, SNorm, t_norm, s_norm


# ======================================================================= #
#  Type aliases                                                            #
# ======================================================================= #

MFDict   = Dict[str, Dict[str, MembershipFunction]]  # {col: {term: MF}}
AggFuncs = Dict[str, str]                             # {col: 'mean'|'sum'|...}


# ======================================================================= #
#  Internal helpers                                                        #
# ======================================================================= #

def _ensure_membership(df: pd.DataFrame) -> pd.DataFrame:
    """Add a _membership column of 1.0 if it doesn't exist yet."""
    df = df.copy()
    if "_membership" not in df.columns:
        df["_membership"] = 1.0
    return df


def _compute_membership(series: pd.Series,
                         mf: MembershipFunction) -> np.ndarray:
    """Compute per-row membership degree for a column using an MF."""
    return mf.compute(series.values)


def _rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("_membership", ascending=False).reset_index(drop=True)


# ======================================================================= #
#  1. FUZZY WHERE — single & multi-condition                               #
# ======================================================================= #

def fuzzy_where(df: pd.DataFrame,
                column: str,
                mf: MembershipFunction,
                existing_membership: Optional[str] = None,
                tnorm: TNorm = TNorm.MINIMUM) -> pd.DataFrame:
    """
    Fuzzy WHERE:  SELECT * FROM df WHERE <column> IS <linguistic_term>

    Parameters
    ----------
    df                  : input DataFrame
    column              : which column to evaluate
    mf                  : the MembershipFunction for the linguistic term
    existing_membership : if the df already carries a '_membership' column,
                          combine with it using the given T-norm
    tnorm               : how to combine with existing membership (AND)

    Returns
    -------
    DataFrame with '_membership' column added/updated, sorted descending.
    """
    df = _ensure_membership(df)
    new_membership = _compute_membership(df[column], mf)

    if existing_membership or "_membership" in df.columns:
        old = df["_membership"].values
        combined = t_norm(old, new_membership, tnorm)
    else:
        combined = new_membership

    df["_membership"] = combined
    return _rank(df)


def fuzzy_where_multi(df: pd.DataFrame,
                      conditions: List[Tuple[str, MembershipFunction]],
                      logic: str = "AND",
                      tnorm: TNorm = TNorm.MINIMUM,
                      snorm: SNorm = SNorm.MAXIMUM,
                      weights: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Multi-condition fuzzy WHERE.

    Parameters
    ----------
    conditions  : list of (column, MembershipFunction) pairs
    logic       : "AND"     → combine with T-norm  (min by default)
                  "OR"      → combine with S-norm  (max by default)
                  "WEIGHTED"→ weighted average of all membership degrees
                              (requires `weights` list of same length)
    Example
    -------
    fuzzy_where_multi(df, [
        ("age",        mf_young),
        ("experience", mf_high),
    ], logic="AND")
    """
    df = df.copy()
    # Compute each condition's membership
    memberships = np.stack(
        [_compute_membership(df[col], mf) for col, mf in conditions],
        axis=1
    )  # shape: (n_rows, n_conditions)

    if logic == "AND":
        combined = memberships[:, 0]
        for k in range(1, memberships.shape[1]):
            combined = t_norm(combined, memberships[:, k], tnorm)

    elif logic == "OR":
        combined = memberships[:, 0]
        for k in range(1, memberships.shape[1]):
            combined = s_norm(combined, memberships[:, k], snorm)

    elif logic == "WEIGHTED":
        if weights is None:
            weights = [1.0] * len(conditions)
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        combined = memberships @ w  # dot product

    else:
        raise ValueError(f"Unknown logic '{logic}'. Use AND / OR / WEIGHTED.")

    df["_membership"] = combined
    return _rank(df)


# ======================================================================= #
#  2. THRESHOLD  (like a HAVING on individual rows)                       #
# ======================================================================= #

def fuzzy_threshold(df: pd.DataFrame,
                    alpha: float = 0.0,
                    strong: bool = False) -> pd.DataFrame:
    """
    Keep only rows whose membership degree satisfies:
      ≥ alpha  (soft cut)
      >  alpha  (strong cut, when strong=True)
    """
    df = _ensure_membership(df)
    m = df["_membership"].values
    mask = m > alpha if strong else m >= alpha
    return df[mask].reset_index(drop=True)


# ======================================================================= #
#  3. FUZZY BETWEEN  (soft range query)                                   #
# ======================================================================= #

def fuzzy_between(df: pd.DataFrame,
                  column: str,
                  low: float,
                  high: float,
                  softness: float = 0.1) -> pd.DataFrame:
    """
    Fuzzy version of  WHERE col BETWEEN low AND high.

    `softness` controls how much the boundaries blur as a fraction of range.
    Classical BETWEEN is a special case when softness=0.

    Membership function used:
      0 outside [low - margin, high + margin]
      1 strictly inside [low, high]
      ramps at both ends of width = softness × (high - low)

    Example
    -------
    fuzzy_between(df, "salary", 40000, 80000, softness=0.15)
    """
    margin = softness * (high - low)
    mf = make_mf(
        name=f"{column}_between_{low}_{high}",
        mf_type="trapezoidal",
        params={"a": low - margin, "b": low,
                "c": high,         "d": high + margin},
        universe_min=df[column].min(),
        universe_max=df[column].max(),
    )
    return fuzzy_where(df, column, mf)


# ======================================================================= #
#  4. FUZZY JOINS                                                          #
# ======================================================================= #

def _similarity_degree(v1: float, v2: float,
                        tolerance: float, method: str = "linear") -> float:
    """Compute similarity between two scalar values."""
    diff = abs(v1 - v2)
    if method == "linear":
        return float(max(0.0, 1.0 - diff / tolerance))
    elif method == "gaussian":
        sigma = tolerance / 3.0
        return float(np.exp(-0.5 * (diff / sigma) ** 2))
    return 1.0 if diff == 0 else 0.0


def fuzzy_inner_join(left: pd.DataFrame,
                     right: pd.DataFrame,
                     on: str,
                     tolerance: float = 0.0,
                     similarity_method: str = "linear",
                     min_membership: float = 0.0) -> pd.DataFrame:
    """
    Fuzzy INNER JOIN.

    Classical inner join requires  left[on] == right[on]  exactly.
    Fuzzy inner join computes a similarity degree for every pair and
    keeps all pairs above min_membership.

    When tolerance=0 this reduces to the classical INNER JOIN.

    Returns a combined DataFrame with '_membership' = similarity degree.
    """
    rows = []
    for _, lrow in left.iterrows():
        for _, rrow in right.iterrows():
            if tolerance == 0.0:
                sim = 1.0 if lrow[on] == rrow[on] else 0.0
            else:
                sim = _similarity_degree(
                    float(lrow[on]), float(rrow[on]),
                    tolerance, similarity_method
                )
            if sim > min_membership:
                combined = {**lrow.to_dict(), **{f"{k}_r": v
                            for k, v in rrow.to_dict().items() if k != on}}
                combined["_membership"] = sim
                rows.append(combined)

    if not rows:
        return pd.DataFrame()
    return _rank(pd.DataFrame(rows))


def fuzzy_left_join(left: pd.DataFrame,
                    right: pd.DataFrame,
                    on: str,
                    tolerance: float = 0.0,
                    similarity_method: str = "linear") -> pd.DataFrame:
    """
    Fuzzy LEFT JOIN.

    Every row in `left` is preserved.
    It is matched to its best-matching row in `right` (max similarity).
    If no match exists, right columns are NaN and membership = 0.
    """
    rows = []
    right_cols = [c for c in right.columns if c != on]

    for _, lrow in left.iterrows():
        best_sim = 0.0
        best_rrow = None

        for _, rrow in right.iterrows():
            if tolerance == 0.0:
                sim = 1.0 if lrow[on] == rrow[on] else 0.0
            else:
                sim = _similarity_degree(
                    float(lrow[on]), float(rrow[on]),
                    tolerance, similarity_method
                )
            if sim > best_sim:
                best_sim = sim
                best_rrow = rrow

        combined = lrow.to_dict()
        if best_rrow is not None:
            for c in right_cols:
                combined[f"{c}_r"] = best_rrow[c]
        else:
            for c in right_cols:
                combined[f"{c}_r"] = np.nan
        combined["_membership"] = best_sim
        rows.append(combined)

    return _rank(pd.DataFrame(rows))


def fuzzy_right_join(left: pd.DataFrame,
                     right: pd.DataFrame,
                     on: str,
                     tolerance: float = 0.0,
                     similarity_method: str = "linear") -> pd.DataFrame:
    """Fuzzy RIGHT JOIN — mirror of left join with sides swapped."""
    result = fuzzy_left_join(right, left, on, tolerance, similarity_method)
    # Rename columns back to expected convention
    return result


def fuzzy_full_join(left: pd.DataFrame,
                    right: pd.DataFrame,
                    on: str,
                    tolerance: float = 0.0,
                    similarity_method: str = "linear") -> pd.DataFrame:
    """
    Fuzzy FULL OUTER JOIN.

    All rows from both sides appear. Unmatched rows get membership = 0.
    """
    left_result  = fuzzy_left_join(left, right, on, tolerance, similarity_method)
    right_result = fuzzy_left_join(right, left, on, tolerance, similarity_method)

    combined = pd.concat([left_result, right_result], ignore_index=True)
    # Drop exact duplicates keeping highest membership
    combined = combined.sort_values("_membership", ascending=False)
    combined = combined.drop_duplicates(subset=[on]).reset_index(drop=True)
    return _rank(combined)


def fuzzy_similarity_join(left: pd.DataFrame,
                          right: pd.DataFrame,
                          left_col: str,
                          right_col: str,
                          mf: MembershipFunction,
                          threshold: float = 0.0) -> pd.DataFrame:
    """
    Join two tables where the membership of the combination of
    (left_col value, right_col value) under a given MF is above threshold.

    Use case: join employees to jobs where salary IS compatible
    (salary difference modelled by an MF).
    """
    rows = []
    for _, lrow in left.iterrows():
        for _, rrow in right.iterrows():
            combined_val = abs(float(lrow[left_col]) - float(rrow[right_col]))
            sim = float(mf.compute(np.array([combined_val]))[0])
            if sim >= threshold:
                entry = {**{f"L_{k}": v for k, v in lrow.to_dict().items()},
                         **{f"R_{k}": v for k, v in rrow.to_dict().items()},
                         "_membership": sim}
                rows.append(entry)
    if not rows:
        return pd.DataFrame()
    return _rank(pd.DataFrame(rows))


# ======================================================================= #
#  5. FUZZY GROUP BY — soft grouping                                      #
# ======================================================================= #

def fuzzy_group_by(df: pd.DataFrame,
                   column: str,
                   mf_dict: Dict[str, MembershipFunction],
                   min_membership: float = 0.0) -> Dict[str, pd.DataFrame]:
    """
    Fuzzy GROUP BY.

    Classical GROUP BY puts each row into exactly ONE group.
    Fuzzy GROUP BY allows a row to belong to MULTIPLE groups
    with different degrees of membership.

    Parameters
    ----------
    df             : input DataFrame
    column         : the column to group on (e.g. "age")
    mf_dict        : {group_name: MembershipFunction}
                     e.g. {"young": mf_young, "middle": mf_middle}
    min_membership : ignore memberships below this (prune weak assignments)

    Returns
    -------
    dict {group_name: DataFrame}   — each df has a '_membership' column
                                     for that group.

    Example
    -------
    groups = fuzzy_group_by(employees, "age",
                            {"young": mf_young, "old": mf_old})
    groups["young"]   # all employees with their degree of being "young"
    """
    groups = {}
    for group_name, mf in mf_dict.items():
        group_df = df.copy()
        group_df["_membership"] = _compute_membership(df[column], mf)
        group_df = group_df[group_df["_membership"] >= min_membership]
        groups[group_name] = _rank(group_df)
    return groups


# ======================================================================= #
#  6. FUZZY AGGREGATE FUNCTIONS                                            #
# ======================================================================= #

def fuzzy_aggregate(groups: Dict[str, pd.DataFrame],
                    agg_column: str,
                    funcs: List[str] = None) -> pd.DataFrame:
    """
    Fuzzy-weighted aggregate functions applied per group.

    Each row contributes proportionally to its membership degree.

    Supported funcs: 'count', 'sum', 'avg', 'min', 'max',
                     'weighted_avg', 'fuzzy_count'

    fuzzy_count   = Σ μ(x)   (sigma-count, not crisp row count)
    weighted_avg  = Σ μ(x)·v(x) / Σ μ(x)

    Returns
    -------
    DataFrame with one row per group and one column per function.
    """
    if funcs is None:
        funcs = ["fuzzy_count", "weighted_avg", "avg", "min", "max"]

    results = []
    for group_name, gdf in groups.items():
        if gdf.empty:
            continue
        m = gdf["_membership"].values
        v = gdf[agg_column].values.astype(float)
        row = {"group": group_name}

        for func in funcs:
            if func == "count":
                row["count"] = len(gdf)
            elif func == "fuzzy_count":
                row["fuzzy_count"] = float(np.sum(m))
            elif func == "sum":
                row["sum"] = float(np.sum(v))
            elif func == "weighted_sum":
                row["weighted_sum"] = float(np.sum(m * v))
            elif func == "avg":
                row["avg"] = float(np.mean(v))
            elif func == "weighted_avg":
                denom = np.sum(m)
                row["weighted_avg"] = float(np.sum(m * v) / denom) if denom > 0 else 0.0
            elif func == "min":
                row["min"] = float(np.min(v))
            elif func == "max":
                row["max"] = float(np.max(v))
            elif func == "fuzzy_min":
                # Soft minimum: weighted by (1 - μ) to emphasise lower-membership outliers
                row["fuzzy_min"] = float(np.sum((1.0 - m) * v) / np.sum(1.0 - m + 1e-10))
            elif func == "fuzzy_max":
                row["fuzzy_max"] = float(np.sum(m * v) / np.sum(m + 1e-10))
            elif func == "std":
                row["std"] = float(np.std(v))

        row["group_membership_avg"] = float(np.mean(m))
        results.append(row)

    return pd.DataFrame(results)


# ======================================================================= #
#  7. FUZZY HAVING — filter groups on their aggregate membership          #
# ======================================================================= #

def fuzzy_having(agg_df: pd.DataFrame,
                 column: str,
                 mf: MembershipFunction,
                 threshold: float = 0.0) -> pd.DataFrame:
    """
    HAVING equivalent for fuzzy aggregated results.

    Example
    -------
    # Keep only groups whose weighted_avg salary IS "high"
    fuzzy_having(agg_result, "weighted_avg", mf_high_salary, threshold=0.5)
    """
    agg_df = agg_df.copy()
    agg_df["_having_membership"] = _compute_membership(agg_df[column], mf)
    result = agg_df[agg_df["_having_membership"] >= threshold]
    return result.sort_values("_having_membership", ascending=False).reset_index(drop=True)


# ======================================================================= #
#  8. SET OPERATIONS                                                       #
# ======================================================================= #

def fuzzy_union(df_A: pd.DataFrame,
                df_B: pd.DataFrame,
                key: str = None) -> pd.DataFrame:
    """
    Fuzzy UNION of two query results.

    If key is provided: for rows present in both, take max(μA, μB).
    Rows present in only one side keep their degree.
    If key is None: simple vertical concatenation, keep max per duplicate.
    """
    df_A = _ensure_membership(df_A)
    df_B = _ensure_membership(df_B)

    if key is None:
        combined = pd.concat([df_A, df_B], ignore_index=True)
        # If duplicate rows exist keep the highest membership
        combined = combined.sort_values("_membership", ascending=False)
        combined = combined.drop_duplicates(
            subset=[c for c in combined.columns if c != "_membership"]
        )
        return _rank(combined)

    # Key-based union: merge only the key + membership columns, keep data from A
    mA_map = dict(zip(df_A[key], df_A["_membership"]))
    mB_map = dict(zip(df_B[key], df_B["_membership"]))
    all_keys = set(mA_map) | set(mB_map)

    # Start from the union of rows (prefer A's data, fall back to B)
    df_all = pd.concat([df_A, df_B], ignore_index=True).drop_duplicates(subset=[key])
    df_all = df_all.drop(columns=["_membership"])

    mA_arr = np.array([mA_map.get(k, 0.0) for k in df_all[key]])
    mB_arr = np.array([mB_map.get(k, 0.0) for k in df_all[key]])
    df_all["_membership"] = np.maximum(mA_arr, mB_arr)
    return _rank(df_all)


def fuzzy_intersect(df_A: pd.DataFrame,
                    df_B: pd.DataFrame,
                    key: str) -> pd.DataFrame:
    """
    Fuzzy INTERSECT — only rows present in both, membership = min(μA, μB).
    """
    df_A = _ensure_membership(df_A)
    df_B = _ensure_membership(df_B)
    mA_map = dict(zip(df_A[key], df_A["_membership"]))
    mB_map = dict(zip(df_B[key], df_B["_membership"]))
    common_keys = set(mA_map) & set(mB_map)
    result = df_A[df_A[key].isin(common_keys)].copy()
    result["_membership"] = result[key].map(
        {k: min(mA_map[k], mB_map[k]) for k in common_keys}
    )
    return _rank(result)


def fuzzy_except(df_A: pd.DataFrame,
                 df_B: pd.DataFrame,
                 key: str,
                 threshold: float = 0.0) -> pd.DataFrame:
    """
    Fuzzy EXCEPT (MINUS).

    Rows in A that do NOT have a well-matching row in B.
    A row in A is "subtracted" in proportion to its best match in B:
        μ_result = μ_A × (1 - max_match_in_B)
    """
    df_A = _ensure_membership(df_A)
    df_B = _ensure_membership(df_B)

    B_keys = set(df_B[key].tolist())
    results = []
    for _, row in df_A.iterrows():
        if row[key] not in B_keys:
            results.append({**row.to_dict()})
        else:
            # Find best match in B
            matches = df_B[df_B[key] == row[key]]["_membership"]
            best_b = float(matches.max()) if len(matches) > 0 else 0.0
            new_m = float(row["_membership"]) * (1.0 - best_b)
            if new_m > threshold:
                entry = row.to_dict()
                entry["_membership"] = new_m
                results.append(entry)
    if not results:
        return pd.DataFrame(columns=df_A.columns)
    return _rank(pd.DataFrame(results))


# ======================================================================= #
#  9. FUZZY ORDER BY                                                       #
# ======================================================================= #

def fuzzy_order_by(df: pd.DataFrame,
                   by: str = "_membership",
                   ascending: bool = False,
                   top_k: Optional[int] = None) -> pd.DataFrame:
    """
    ORDER BY with optional top-K selection.
    Default sorts by fuzzy membership score descending.
    """
    df = _ensure_membership(df)
    result = df.sort_values(by, ascending=ascending).reset_index(drop=True)
    if top_k is not None:
        result = result.head(top_k)
    return result


# ======================================================================= #
#  10. FUZZY DISTINCT — similarity-based deduplication                    #
# ======================================================================= #

def fuzzy_distinct(df: pd.DataFrame,
                   columns: List[str],
                   similarity_threshold: float = 0.95,
                   keep: str = "highest") -> pd.DataFrame:
    """
    Fuzzy DISTINCT — remove rows that are "too similar" to each other.

    Two rows are considered near-duplicates if their normalised
    Euclidean similarity across `columns` exceeds `similarity_threshold`.

    Parameters
    ----------
    columns               : numeric columns to use for similarity computation
    similarity_threshold  : rows with similarity > this are merged
    keep                  : "highest" → keep row with highest membership
                            "first"   → keep first encountered

    Use case: a query returns two candidates with age=28 and age=29 —
    DISTINCT with threshold=0.95 might consider them too similar and keep only one.
    """
    df = _ensure_membership(df).reset_index(drop=True)
    data = df[columns].values.astype(float)

    # Normalise each column to [0,1] for fair distance computation
    col_range = data.max(axis=0) - data.min(axis=0)
    col_range[col_range == 0] = 1.0
    normalised = (data - data.min(axis=0)) / col_range

    kept = []
    removed = set()

    for i in range(len(df)):
        if i in removed:
            continue
        kept.append(i)
        for j in range(i + 1, len(df)):
            if j in removed:
                continue
            dist = np.linalg.norm(normalised[i] - normalised[j])
            sim = 1.0 / (1.0 + dist)
            if sim >= similarity_threshold:
                if keep == "highest":
                    if df.loc[j, "_membership"] > df.loc[i, "_membership"]:
                        kept[-1] = j
                removed.add(j)

    return _rank(df.loc[kept].reset_index(drop=True))


# ======================================================================= #
#  11. FUZZY EXISTS / IN                                                   #
# ======================================================================= #

def fuzzy_exists(df: pd.DataFrame,
                 sub_df: pd.DataFrame,
                 key: str) -> pd.DataFrame:
    """
    Fuzzy EXISTS subquery.

    For each row in df, compute the degree to which a matching row
    EXISTS in sub_df.

    μ_exists(x) = max membership of matching rows in sub_df
                = 0 if no match at all (classical NOT EXISTS)
    """
    df = _ensure_membership(df)
    sub_map = (sub_df.groupby(key)["_membership"].max()
               if "_membership" in sub_df.columns
               else sub_df.groupby(key).size().clip(upper=1).astype(float))

    df = df.copy()
    df["_exists_membership"] = df[key].map(sub_map).fillna(0.0)
    df["_membership"] = t_norm(
        df["_membership"].values,
        df["_exists_membership"].values
    )
    df = df.drop(columns=["_exists_membership"])
    return _rank(df)


def fuzzy_in(df: pd.DataFrame,
             column: str,
             values: list,
             mf: Optional[MembershipFunction] = None,
             tolerance: float = 0.0) -> pd.DataFrame:
    """
    Fuzzy IN  — membership degree of a column value being "in" a set.

    Classical: WHERE age IN (25, 30, 35)  → binary
    Fuzzy:     each value gets a similarity degree to the nearest
               value in the list, optionally shaped by an MF.

    If `mf` is provided, membership is mf(min_distance_to_values).
    Otherwise linear decay within `tolerance` is used.
    """
    df = _ensure_membership(df)
    col_vals = df[column].values.astype(float)
    in_vals  = np.array(values, dtype=float)

    memberships = np.zeros(len(col_vals))
    for i, v in enumerate(col_vals):
        dists = np.abs(in_vals - v)
        min_dist = float(np.min(dists))
        if mf is not None:
            memberships[i] = float(mf.compute(np.array([min_dist]))[0])
        else:
            memberships[i] = max(0.0, 1.0 - min_dist / (tolerance + 1e-12))

    df = df.copy()
    df["_membership"] = t_norm(df["_membership"].values, memberships)
    return _rank(df)


# ======================================================================= #
#  12. FuzzyQuery — Chainable Query Builder                                #
# ======================================================================= #

@dataclass
class FuzzyQuery:
    """
    Chainable fluent query builder for fuzzy SQL operations.

    Usage
    -----
    result = (
        FuzzyQuery(employees)
        .where("age",        mf_young)
        .where("salary",     mf_high,   logic="AND")
        .where("experience", mf_senior, logic="OR")
        .threshold(0.3)
        .order_by()
        .top(10)
        .execute()
    )

    Each .where() adds a condition.  logic="AND" combines with T-norm,
    logic="OR" combines with S-norm.
    """
    _df           : pd.DataFrame
    _conditions   : list = field(default_factory=list)
    _threshold    : float = 0.0
    _order_col    : str = "_membership"
    _ascending    : bool = False
    _top_k        : Optional[int] = None
    _tnorm        : TNorm = TNorm.MINIMUM
    _snorm        : SNorm = SNorm.MAXIMUM

    def where(self, column: str,
              mf: MembershipFunction,
              logic: str = "AND") -> "FuzzyQuery":
        """Add a fuzzy WHERE condition."""
        self._conditions.append((column, mf, logic))
        return self

    def threshold(self, alpha: float, strong: bool = False) -> "FuzzyQuery":
        """Set minimum membership threshold for output rows."""
        self._threshold = alpha
        self._strong = strong
        return self

    def order_by(self, column: str = "_membership",
                 ascending: bool = False) -> "FuzzyQuery":
        self._order_col = column
        self._ascending = ascending
        return self

    def top(self, k: int) -> "FuzzyQuery":
        self._top_k = k
        return self

    def set_tnorm(self, tnorm: TNorm) -> "FuzzyQuery":
        self._tnorm = tnorm
        return self

    def set_snorm(self, snorm: SNorm) -> "FuzzyQuery":
        self._snorm = snorm
        return self

    def execute(self) -> pd.DataFrame:
        """Execute the accumulated query pipeline."""
        df = _ensure_membership(self._df.copy())

        for column, mf, logic in self._conditions:
            new_m = _compute_membership(df[column], mf)
            old_m = df["_membership"].values
            if logic == "AND":
                df["_membership"] = t_norm(old_m, new_m, self._tnorm)
            elif logic == "OR":
                df["_membership"] = s_norm(old_m, new_m, self._snorm)
            else:
                raise ValueError(f"Unknown logic '{logic}' in condition.")

        df = fuzzy_threshold(df, self._threshold)
        df = fuzzy_order_by(df, self._order_col, self._ascending, self._top_k)
        return df

    def explain(self) -> str:
        """Return a human-readable explanation of the query."""
        lines = ["FuzzyQuery pipeline:"]
        lines.append(f"  Table: {len(self._df)} rows")
        for i, (col, mf, logic) in enumerate(self._conditions):
            prefix = "WHERE" if i == 0 else f"  {logic}"
            lines.append(f"  {prefix} {col} IS '{mf.name}'  [MF: {mf.mf_type.value}]")
        lines.append(f"  THRESHOLD >= {self._threshold}")
        if self._top_k:
            lines.append(f"  TOP {self._top_k}")
        lines.append(f"  ORDER BY {self._order_col} "
                     f"{'ASC' if self._ascending else 'DESC'}")
        return "\n".join(lines)


# ======================================================================= #
#  13. COMPARE — Fuzzy vs Classical side-by-side                          #
# ======================================================================= #

def compare_with_classical(df: pd.DataFrame,
                            fuzzy_result: pd.DataFrame,
                            classical_conditions: Dict[str, tuple],
                            id_col: Optional[str] = None) -> dict:
    """
    Compare fuzzy query results against equivalent crisp SQL.

    Parameters
    ----------
    df                    : original DataFrame
    fuzzy_result          : output of a FuzzyQuery.execute()
    classical_conditions  : {col: (operator, value)}
                            e.g. {"age": ("<", 35), "salary": (">", 50000)}
    id_col                : optional row identifier column

    Returns
    -------
    dict with:
      classical_result    : rows matching classical conditions
      fuzzy_result        : ranked fuzzy result
      only_in_fuzzy       : rows fuzzy found that classical missed
      only_in_classical   : rows classical found that fuzzy ranked < threshold
      overlap             : rows found by both
      fuzzy_advantage     : description of what fuzzy adds
    """
    # Build classical filter
    mask = pd.Series([True] * len(df), index=df.index)
    for col, (op, val) in classical_conditions.items():
        if op == "<":  mask &= df[col] < val
        elif op == "<=": mask &= df[col] <= val
        elif op == ">":  mask &= df[col] > val
        elif op == ">=": mask &= df[col] >= val
        elif op == "==": mask &= df[col] == val
        elif op == "!=": mask &= df[col] != val
    classical_result = df[mask].copy()

    if id_col and id_col in df.columns:
        classical_ids = set(classical_result[id_col].tolist())
        fuzzy_ids     = set(fuzzy_result[id_col].tolist())
        only_fuzzy    = fuzzy_result[fuzzy_result[id_col].isin(fuzzy_ids - classical_ids)]
        only_classical= classical_result[classical_result[id_col].isin(classical_ids - fuzzy_ids)]
        overlap       = fuzzy_result[fuzzy_result[id_col].isin(fuzzy_ids & classical_ids)]
    else:
        only_fuzzy     = pd.DataFrame()
        only_classical = pd.DataFrame()
        overlap        = fuzzy_result

    return {
        "classical_result"  : classical_result,
        "fuzzy_result"      : fuzzy_result,
        "classical_count"   : len(classical_result),
        "fuzzy_count"       : len(fuzzy_result),
        "only_in_fuzzy"     : only_fuzzy,
        "only_in_classical" : only_classical,
        "overlap"           : overlap,
        "fuzzy_advantage"   : (
            f"Fuzzy found {len(only_fuzzy)} additional relevant rows "
            f"that classical SQL missed entirely. "
            f"Classical missed {len(only_fuzzy)} borderline cases."
        )
    }