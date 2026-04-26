"""
main.py
=======
FuzzyRDBEngine — the single public API for the Fuzzy Relational Database.

Wires together every core module:
  StorageEngine      — SQLite persistence + CSV loading
  MFRegistry         — membership function store + compute
  fuzzy_sql          — all fuzzy relational operations
  alpha_cut          — set analysis
  defuzzification    — crisp output from fuzzy sets
  hedges             — linguistic modifiers

The shell (cli/shell.py) calls only this class.
Any Python script can also import and use it directly.

Quick-start
-----------
    from main import FuzzyRDBEngine

    eng = FuzzyRDBEngine()
    eng.load_csv("data/employees.csv", "employees")

    eng.define_term("employees", "age",    "young", "triangular",
                    {"a": 15, "b": 25, "c": 35})
    eng.define_term("employees", "salary", "high",  "trapezoidal",
                    {"a": 70000, "b": 90000, "c": 120000, "d": 150000})

    results = eng.query("employees", [
        {"col": "age",    "hedge": None,   "term": "young", "logic": "AND"},
        {"col": "salary", "hedge": "very", "term": "high",  "logic": "AND"},
    ], threshold=0.3, top_k=10)

    print(results)
"""

from __future__ import annotations
import os, sys, json
import numpy  as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_HERE, "core")
_CLI  = os.path.join(_HERE, "cli")
for _p in (_CORE, _CLI, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.storage              import StorageEngine
from core.fuzzifier            import MFRegistry
from core.hedges               import apply_hedge, HEDGE_REGISTRY, HEDGE_DESCRIPTIONS
from core.defuzzification      import compare_methods as _compare_defuzz, defuzzify
from core.alpha_cut            import analyze as _analyze_set
from core.membership_functions import make_mf, MF_PARAM_GUIDE
from core.fuzzy_sql import (
    FuzzyQuery,
    fuzzy_where, fuzzy_where_multi,
    fuzzy_between, fuzzy_threshold,
    fuzzy_inner_join, fuzzy_left_join,
    fuzzy_right_join, fuzzy_full_join,
    fuzzy_group_by, fuzzy_aggregate,
    fuzzy_having,
    fuzzy_union, fuzzy_intersect, fuzzy_except,
    fuzzy_distinct, fuzzy_in,
    compare_with_classical,
)


# ─────────────────────────────────────────────────────────────────────────────
#  _HedgedMF  — wraps a base MF with a linguistic hedge
# ─────────────────────────────────────────────────────────────────────────────

class _HedgedMF:
    """
    Thin wrapper so a hedged MF can be passed wherever an MF is expected.
    Supports the same .compute() interface.
    """
    def __init__(self, base_mf, hedge: str):
        self._base        = base_mf
        self._hedge       = hedge
        self.name         = f"{hedge} {base_mf.name}"
        self.mf_type      = base_mf.mf_type
        self.params       = base_mf.params
        self.universe_min = base_mf.universe_min
        self.universe_max = base_mf.universe_max

    def compute(self, x) -> np.ndarray:
        return apply_hedge(self._hedge, self._base.compute(x))


# ─────────────────────────────────────────────────────────────────────────────
#  FuzzyRDBEngine
# ─────────────────────────────────────────────────────────────────────────────

class FuzzyRDBEngine:
    """
    The main Fuzzy Relational Database Engine.

    Parameters
    ----------
    db_path : str
        Path to SQLite file. Use ':memory:' (default) for in-session only.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.storage = StorageEngine(db_path)
        self._registries: dict[str, MFRegistry] = {}
        # In-memory result cache for set operations
        # { label: DataFrame }  — populated by query(), join(), etc.
        self._results: dict[str, pd.DataFrame] = {}

    # ═════════════════════════════════════════════════════════════════════════
    #  DATA LOADING
    # ═════════════════════════════════════════════════════════════════════════

    def load_csv(self, filepath: str, table_name: str) -> pd.DataFrame:
        """Load a CSV into the engine. Returns the loaded DataFrame."""
        df = self.storage.load_csv(filepath, table_name)
        self._registries[table_name] = self._rebuild_registry(table_name)
        return df

    def load_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        """Load a pandas DataFrame directly."""
        self.storage.load_dataframe(df, table_name)
        self._registries[table_name] = self._rebuild_registry(table_name)

    # ═════════════════════════════════════════════════════════════════════════
    #  INTROSPECTION
    # ═════════════════════════════════════════════════════════════════════════

    def list_tables(self) -> list[str]:
        return self.storage.list_tables()

    def describe_table(self, table_name: str) -> dict:
        """
        Full description of a table:
          columns, row_count, per-column stats, defined fuzzy terms.
        """
        self._require_table(table_name)
        columns   = self.storage.get_columns(table_name)
        row_count = self.storage.row_count(table_name)
        numeric   = self.storage.get_numeric_columns(table_name)

        stats = {}
        for col in numeric:
            try:
                stats[col] = self.storage.column_stats(table_name, col)
            except Exception:
                pass

        reg        = self._get_registry(table_name)
        fuzzy_cols = reg.list_columns()

        return {
            "table"     : table_name,
            "columns"   : columns,
            "row_count" : row_count,
            "stats"     : stats,
            "fuzzy_cols": fuzzy_cols,
        }

    # ═════════════════════════════════════════════════════════════════════════
    #  MEMBERSHIP FUNCTION MANAGEMENT
    # ═════════════════════════════════════════════════════════════════════════

    def define_term(self, table_name: str, column: str, term: str,
                    mf_type: str, params: dict,
                    umin: float | None = None,
                    umax: float | None = None) -> None:
        """
        Define a fuzzy linguistic term on a column.
        umin/umax are inferred from data if not supplied.
        """
        self._require_table(table_name)

        if umin is None or umax is None:
            data_min, data_max = self.storage.get_column_range(table_name, column)
            umin = umin if umin is not None else data_min
            umax = umax if umax is not None else data_max

        reg = self._get_registry(table_name)
        reg.define(column=column, term=term, mf_type=mf_type,
                   params=params, universe_min=umin, universe_max=umax)

        self.storage.save_mf(
            table_name=table_name, column_name=column,
            term=term, mf_type=mf_type, params=list(params.values()),
        )

    def list_terms(self, table_name: str) -> list[dict]:
        """All MFs defined for a table."""
        self._require_table(table_name)
        return self._get_registry(table_name).summary()

    def get_registry(self, table_name: str) -> MFRegistry:
        self._require_table(table_name)
        return self._get_registry(table_name)

    # ═════════════════════════════════════════════════════════════════════════
    #  QUERY  (fuzzy SELECT)
    # ═════════════════════════════════════════════════════════════════════════

    def query(self, table_name: str,
              conditions: list[dict],
              threshold: float = 0.0,
              top_k: int | None = None,
              result_label: str | None = None) -> pd.DataFrame:
        """
        Fuzzy SELECT with one or more WHERE conditions.

        Each condition dict:  {col, hedge, term, logic}
        Stores result under result_label (or table_name) for later set ops.
        """
        self._require_table(table_name)
        if not conditions:
            raise ValueError("At least one WHERE condition is required.")

        reg = self._get_registry(table_name)
        df  = self.storage.fetch_as_dataframe(table_name)
        if df.empty:
            return df

        qb = FuzzyQuery(df)
        for cond in conditions:
            col, hedge, term, logic = (
                cond["col"], cond.get("hedge"), cond["term"],
                cond.get("logic", "AND"),
            )
            if not reg.has(col, term):
                raise KeyError(
                    f"Term '{term}' not defined for '{col}' in '{table_name}'. "
                    f"Use SHOW TERMS {table_name}."
                )
            base_mf = reg.get(col, term)
            mf = _HedgedMF(base_mf, hedge) if hedge else base_mf
            qb.where(col, mf, logic=logic)

        result = qb.threshold(threshold).execute()
        if top_k:
            result = result.head(top_k)

        label = result_label or table_name
        self._results[label] = result
        return result

    def query_between(self, table_name: str, col: str,
                      low: float, high: float, softness: float = 0.1,
                      threshold: float = 0.0,
                      top_k: int | None = None) -> pd.DataFrame:
        """Soft BETWEEN query."""
        self._require_table(table_name)
        df = self.storage.fetch_as_dataframe(table_name)
        result = fuzzy_between(df, col, low, high, softness)
        result = fuzzy_threshold(result, threshold)
        if top_k:
            result = result.head(top_k)
        self._results[table_name] = result
        return result

    def query_in(self, table_name: str, col: str,
                 values: list, tolerance: float = 0.0,
                 threshold: float = 0.0,
                 top_k: int | None = None) -> pd.DataFrame:
        """Fuzzy IN query."""
        self._require_table(table_name)
        df = self.storage.fetch_as_dataframe(table_name)
        result = fuzzy_in(df, col, values, tolerance=tolerance)
        result = fuzzy_threshold(result, threshold)
        if top_k:
            result = result.head(top_k)
        self._results[table_name] = result
        return result

    def query_distinct(self, table_name: str,
                       columns: list[str] | None = None,
                       similarity: float = 0.95,
                       conditions: list[dict] | None = None,
                       threshold: float = 0.0) -> pd.DataFrame:
        """Fuzzy DISTINCT — remove near-duplicate rows."""
        self._require_table(table_name)
        df = self.storage.fetch_as_dataframe(table_name)

        if conditions:
            reg = self._get_registry(table_name)
            qb  = FuzzyQuery(df)
            for cond in conditions:
                base_mf = reg.get(cond["col"], cond["term"])
                mf = _HedgedMF(base_mf, cond["hedge"]) if cond.get("hedge") else base_mf
                qb.where(cond["col"], mf, logic=cond.get("logic", "AND"))
            df = qb.threshold(threshold).execute()

        numeric_cols = columns or [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c != "_membership"
        ]
        result = fuzzy_distinct(df, numeric_cols, similarity_threshold=similarity)
        self._results[table_name] = result
        return result

    # ═════════════════════════════════════════════════════════════════════════
    #  JOIN
    # ═════════════════════════════════════════════════════════════════════════

    def join(self, left_table: str, right_table: str,
             on: str, join_type: str = "inner",
             tolerance: float = 0.0,
             threshold: float = 0.0,
             top_k: int | None = None,
             result_label: str | None = None) -> pd.DataFrame:
        """
        Fuzzy JOIN between two loaded tables.

        join_type : inner | left | right | full
        tolerance : similarity window (0 = exact key match like classical SQL)
        """
        self._require_table(left_table)
        self._require_table(right_table)

        left_df  = self.storage.fetch_as_dataframe(left_table)
        right_df = self.storage.fetch_as_dataframe(right_table)

        join_fns = {
            "inner": fuzzy_inner_join,
            "left" : fuzzy_left_join,
            "right": fuzzy_right_join,
            "full" : fuzzy_full_join,
        }
        if join_type not in join_fns:
            raise ValueError(
                f"Unknown join type '{join_type}'. "
                f"Valid: {list(join_fns.keys())}"
            )

        result = join_fns[join_type](
            left_df, right_df, on=on, tolerance=tolerance
        )

        if threshold > 0 and "_membership" in result.columns:
            result = fuzzy_threshold(result, threshold)
        if top_k:
            result = result.head(top_k)

        label = result_label or f"{left_table}_x_{right_table}"
        self._results[label] = result
        return result

    # ═════════════════════════════════════════════════════════════════════════
    #  GROUP BY + AGGREGATE
    # ═════════════════════════════════════════════════════════════════════════

    def group_by(self, table_name: str, col: str,
                 terms: list[str],
                 agg_col: str | None = None,
                 agg_funcs: list[str] | None = None,
                 min_membership: float = 0.0) -> dict:
        """
        Fuzzy GROUP BY.

        Rows can belong to multiple groups with different degrees.
        Returns a dict:
          {
            "groups"      : { term: DataFrame },
            "aggregation" : DataFrame or None,
          }
        """
        self._require_table(table_name)
        reg = self._get_registry(table_name)
        df  = self.storage.fetch_as_dataframe(table_name)

        mf_dict = {}
        for term in terms:
            if not reg.has(col, term):
                raise KeyError(
                    f"Term '{term}' not defined for '{col}'. "
                    f"Use DEFINE TERM first."
                )
            mf_dict[term] = reg.get(col, term)

        groups = fuzzy_group_by(df, col, mf_dict, min_membership=min_membership)

        aggregation = None
        if agg_col and groups:
            funcs = agg_funcs or ["fuzzy_count", "weighted_avg", "avg", "min", "max"]
            aggregation = fuzzy_aggregate(groups, agg_col, funcs)

        return {"groups": groups, "aggregation": aggregation}

    # ═════════════════════════════════════════════════════════════════════════
    #  SET OPERATIONS
    # ═════════════════════════════════════════════════════════════════════════

    def set_op(self, op: str,
               table_a: str, table_b: str,
               key: str,
               threshold: float = 0.0) -> pd.DataFrame:
        """
        Fuzzy UNION / INTERSECT / EXCEPT.

        table_a, table_b can be:
          • A loaded table name  (fetches the full table as a fuzzy set)
          • A result label from a previous query stored in self._results

        op : 'union' | 'intersect' | 'except'
        """
        df_a = self._resolve_source(table_a)
        df_b = self._resolve_source(table_b)

        ops = {
            "union"    : fuzzy_union,
            "intersect": fuzzy_intersect,
            "except"   : fuzzy_except,
        }
        if op not in ops:
            raise ValueError(f"Unknown set op '{op}'. Valid: union | intersect | except")

        if op == "except":
            result = fuzzy_except(df_a, df_b, key, threshold=threshold)
        else:
            result = ops[op](df_a, df_b, key=key)
            if threshold > 0 and "_membership" in result.columns:
                result = fuzzy_threshold(result, threshold)

        label = f"{table_a}_{op}_{table_b}"
        self._results[label] = result
        return result

    def _resolve_source(self, name: str) -> pd.DataFrame:
        """Resolve a name to a DataFrame: result cache first, then table."""
        if name in self._results:
            return self._results[name]
        if name in self.storage.list_tables():
            return self.storage.fetch_as_dataframe(name)
        raise KeyError(
            f"'{name}' is neither a loaded table nor a stored query result.\n"
            f"Available tables: {self.storage.list_tables()}\n"
            f"Available results: {list(self._results.keys())}"
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  COMPARE  (viva proof)
    # ═════════════════════════════════════════════════════════════════════════

    def compare(self, table_name: str,
                fuzzy_conditions: list[dict],
                classical_conditions: dict,
                threshold: float = 0.0) -> dict:
        """
        Side-by-side comparison of fuzzy vs classical SQL.

        Returns a dict with:
          classical_result   : DataFrame
          fuzzy_result       : DataFrame (ranked)
          classical_count    : int
          fuzzy_count        : int
          only_in_fuzzy      : rows fuzzy found that classical missed
          only_in_classical  : rows classical found below fuzzy threshold
          overlap            : rows both found
          fuzzy_advantage    : human-readable summary string
        """
        self._require_table(table_name)
        df = self.storage.fetch_as_dataframe(table_name)

        # Infer id column (first column, or first unique-ish column)
        id_col = df.columns[0]

        fuzzy_result = self.query(
            table_name, fuzzy_conditions, threshold=threshold
        )

        return compare_with_classical(
            df=df,
            fuzzy_result=fuzzy_result,
            classical_conditions=classical_conditions,
            id_col=id_col,
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════

    def analyze(self, table_name: str, column: str, term: str) -> dict:
        """Full alpha-cut analysis of a fuzzy set."""
        self._require_table(table_name)
        mf = self._get_registry(table_name).get(column, term)
        universe = np.linspace(mf.universe_min, mf.universe_max, 500)
        mu = mf.compute(universe)
        return _analyze_set(universe, mu, name=f"{column} IS {term}")

    def defuzz(self, table_name: str, column: str,
               term: str, method: str = "all") -> dict | float:
        """Defuzzify a term's MF. method='all' compares all methods."""
        self._require_table(table_name)
        mf = self._get_registry(table_name).get(column, term)
        universe = np.linspace(mf.universe_min, mf.universe_max, 500)
        mu = mf.compute(universe)
        if method == "all":
            return _compare_defuzz(universe, mu)
        return defuzzify(universe, mu, method=method)

    def suggest_terms(self, table_name: str, column: str,
                      n_terms: int = 3) -> list[dict]:
        """Data-driven MF parameter suggestions for a column."""
        self._require_table(table_name)
        df  = self.storage.fetch_as_dataframe(table_name)
        reg = self._get_registry(table_name)
        return reg.suggest_from_data(column, df[column], n_terms=n_terms)

    def ascii_plot(self, table_name: str, column: str) -> str:
        """ASCII plot of all MFs defined on a column."""
        self._require_table(table_name)
        return self._get_registry(table_name).ascii_plot(column)

    # ═════════════════════════════════════════════════════════════════════════
    #  STATIC INFO
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def list_hedges() -> list[dict]:
        return [
            {"hedge": k, "description": v}
            for k, v in HEDGE_DESCRIPTIONS.items()
        ]

    @staticmethod
    def list_mf_types() -> list[dict]:
        return [
            {"mf_type": k, "parameters": v}
            for k, v in MF_PARAM_GUIDE.items()
        ]

    # ═════════════════════════════════════════════════════════════════════════
    #  INTERNALS
    # ═════════════════════════════════════════════════════════════════════════

    def _require_table(self, name: str) -> None:
        if name not in self.storage.list_tables():
            raise KeyError(
                f"Table '{name}' not found. "
                f"Loaded tables: {self.storage.list_tables()}. "
                "Use  LOAD <path> AS <table>  to load data."
            )

    def _get_registry(self, table_name: str) -> MFRegistry:
        if table_name not in self._registries:
            self._registries[table_name] = self._rebuild_registry(table_name)
        return self._registries[table_name]

    def _rebuild_registry(self, table_name: str) -> MFRegistry:
        """Restore MFRegistry from SQLite _mf_registry metadata."""
        from cli.parser import MF_PARAM_KEYS
        reg     = MFRegistry()
        all_mfs = self.storage.get_all_mfs(table_name)

        for col, mf_list in all_mfs.items():
            for entry in mf_list:
                mf_type = entry["mf_type"]
                params  = entry["params"]
                keys    = MF_PARAM_KEYS.get(mf_type, [])
                params_dict = (
                    dict(zip(keys, params))
                    if len(keys) == len(params)
                    else {str(i): v for i, v in enumerate(params)}
                )
                try:
                    umin, umax = self.storage.get_column_range(table_name, col)
                except Exception:
                    umin, umax = 0.0, 1.0

                reg.define(
                    column=col, term=entry["term"],
                    mf_type=mf_type, params=params_dict,
                    universe_min=umin, universe_max=umax,
                )
        return reg

    def __repr__(self):
        tables = self.storage.list_tables()
        mf_counts = {t: len(self._get_registry(t).summary()) for t in tables}
        return (
            f"<FuzzyRDBEngine  db='{self.storage.db_path}'  "
            f"tables={tables}  mf_counts={mf_counts}>"
        )