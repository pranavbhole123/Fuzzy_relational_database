"""
main.py
=======
FuzzyRDBEngine — the single public API for the Fuzzy Relational Database.

This is the orchestration layer that wires together:
  StorageEngine  (storage.py)   — SQLite persistence + CSV loading
  MFRegistry     (fuzzifier.py) — membership function store + compute
  fuzzy_sql      (fuzzy_sql.py) — all fuzzy relational operations
  alpha_cut      (alpha_cut.py) — set analysis
  defuzzification (defuzzification.py) — crisp output

The shell (cli/shell.py) calls only this class.
Any Python script can also import and use it directly.

──────────────────────────────────────────────────────────────────
Quick-start example
──────────────────────────────────────────────────────────────────

    from main import FuzzyRDBEngine

    eng = FuzzyRDBEngine()
    eng.load_csv("data/employees.csv", "employees")

    eng.define_term("employees", "age", "young",
                    "triangular", {"a": 15, "b": 25, "c": 35})
    eng.define_term("employees", "salary", "high",
                    "trapezoidal", {"a": 70000, "b": 90000,
                                    "c": 120000, "d": 150000})

    results = eng.query(
        "employees",
        conditions=[
            {"col": "age",    "hedge": None,   "term": "young", "logic": "AND"},
            {"col": "salary", "hedge": "very", "term": "high",  "logic": "AND"},
        ],
        threshold=0.3,
        top_k=10,
    )
    print(results)
──────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import os
import sys
import json
import numpy as np
import pandas as pd

# Make core/ importable regardless of how this file is run
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_HERE, "core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from core.storage             import StorageEngine
from core.fuzzifier           import MFRegistry
from core.fuzzy_sql           import FuzzyQuery, fuzzy_where, fuzzy_where_multi
from core.alpha_cut           import analyze as _analyze_set
from core.defuzzification     import compare_methods as _compare_defuzz, defuzzify
from core.membership_functions import make_mf


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class FuzzyRDBEngine:
    """
    The main Fuzzy Relational Database Engine.

    One engine instance manages:
      • one SQLite database  (in-memory or file)
      • one MFRegistry per loaded table

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
        Use ':memory:' (default) for an ephemeral in-session database.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.storage: StorageEngine = StorageEngine(db_path)
        # { table_name: MFRegistry }
        self._registries: dict[str, MFRegistry] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────────────

    def load_csv(self, filepath: str, table_name: str) -> pd.DataFrame:
        """
        Load a CSV into the engine and create a fresh MFRegistry for it.

        Returns the loaded DataFrame for preview.
        """
        df = self.storage.load_csv(filepath, table_name)
        # Restore MFs from storage metadata if any exist
        self._registries[table_name] = self._rebuild_registry(table_name)
        return df

    def load_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        """Load a pandas DataFrame directly."""
        self.storage.load_dataframe(df, table_name)
        self._registries[table_name] = self._rebuild_registry(table_name)

    # ─────────────────────────────────────────────────────────────────────────
    # Table introspection
    # ─────────────────────────────────────────────────────────────────────────

    def list_tables(self) -> list[str]:
        return self.storage.list_tables()

    def describe_table(self, table_name: str) -> dict:
        """
        Return a description dict:
          columns   : {col: sqlite_type}
          row_count : int
          stats     : {numeric_col: {min, max, mean, median, std}}
          fuzzy_cols: list of columns with MFs defined
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

        reg        = self._get_or_create_registry(table_name)
        fuzzy_cols = reg.list_columns()

        return {
            "table"     : table_name,
            "columns"   : columns,
            "row_count" : row_count,
            "stats"     : stats,
            "fuzzy_cols": fuzzy_cols,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Membership function management
    # ─────────────────────────────────────────────────────────────────────────

    def define_term(
        self,
        table_name  : str,
        column      : str,
        term        : str,
        mf_type     : str,
        params      : dict,
        umin        : float | None = None,
        umax        : float | None = None,
    ) -> None:
        """
        Define a fuzzy linguistic term on a column.

        If umin/umax are not provided, they are inferred from the column's
        actual min/max values in the loaded data.

        Parameters
        ----------
        table_name : target table
        column     : numeric column to fuzzify
        term       : linguistic label, e.g. 'young', 'high'
        mf_type    : one of triangular | trapezoidal | gaussian |
                     generalized_bell | sigmoid | s_shaped |
                     z_shaped | pi_shaped | singleton
        params     : named-param dict, e.g. {"a":15, "b":25, "c":35}
        umin, umax : universe bounds (inferred from data if omitted)
        """
        self._require_table(table_name)

        # Infer universe bounds from data if not supplied
        if umin is None or umax is None:
            data_min, data_max = self.storage.get_column_range(table_name, column)
            umin = umin if umin is not None else data_min
            umax = umax if umax is not None else data_max

        reg = self._get_or_create_registry(table_name)
        reg.define(
            column       = column,
            term         = term,
            mf_type      = mf_type,
            params       = params,
            universe_min = umin,
            universe_max = umax,
        )

        # Persist to SQLite metadata
        self.storage.save_mf(
            table_name  = table_name,
            column_name = column,
            term        = term,
            mf_type     = mf_type,
            params      = list(params.values()),
        )

    def list_terms(self, table_name: str) -> list[dict]:
        """Return summary of all MFs defined for a table."""
        self._require_table(table_name)
        reg = self._get_or_create_registry(table_name)
        return reg.summary()

    def get_registry(self, table_name: str) -> MFRegistry:
        """Return the MFRegistry for a table (advanced use)."""
        self._require_table(table_name)
        return self._get_or_create_registry(table_name)

    # ─────────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────────

    def query(
        self,
        table_name : str,
        conditions : list[dict],
        threshold  : float = 0.0,
        top_k      : int | None = None,
    ) -> pd.DataFrame:
        """
        Execute a fuzzy SELECT query.

        Parameters
        ----------
        table_name : table to query
        conditions : list of condition dicts, each with keys:
                       col   — column name
                       hedge — hedge name (or None)
                       term  — linguistic term name
                       logic — 'AND' | 'OR'
        threshold  : minimum membership degree to include a row
        top_k      : return only the top-k rows (None = all)

        Returns
        -------
        DataFrame ranked by '_membership' descending, rows ≥ threshold.
        """
        self._require_table(table_name)
        if not conditions:
            raise ValueError("At least one WHERE condition is required.")

        reg = self._get_or_create_registry(table_name)
        df  = self.storage.fetch_as_dataframe(table_name)

        if df.empty:
            return df

        # Resolve each condition into an MF (with optional hedge applied)
        qb = FuzzyQuery(df)
        for cond in conditions:
            col, hedge, term, logic = (
                cond["col"], cond.get("hedge"), cond["term"], cond.get("logic", "AND")
            )

            # Validate term exists
            if not reg.has(col, term):
                raise KeyError(
                    f"Term '{term}' not defined for column '{col}' in table "
                    f"'{table_name}'. Use SHOW TERMS {table_name} to list defined terms."
                )

            # Get the MF; if a hedge is requested we wrap it in a custom MF
            base_mf = reg.get(col, term)
            if hedge:
                mf = _HedgedMF(base_mf, hedge)
            else:
                mf = base_mf

            qb.where(col, mf, logic=logic)

        result = qb.threshold(threshold).execute()

        if top_k:
            result = result.head(top_k)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(self, table_name: str, column: str, term: str) -> dict:
        """
        Run the full alpha-cut analysis on a fuzzy set.

        Returns the dict produced by alpha_cut.analyze().
        """
        self._require_table(table_name)
        reg = self._get_or_create_registry(table_name)
        mf  = reg.get(column, term)

        col_min, col_max = self.storage.get_column_range(table_name, column)
        universe = np.linspace(mf.universe_min, mf.universe_max, 500)
        mu       = mf.compute(universe)

        return _analyze_set(universe, mu, name=f"{column} IS {term}")

    def defuzz(self, table_name: str, column: str,
               term: str, method: str = "all") -> dict | float:
        """
        Defuzzify a term's MF curve.

        method = 'all'  → returns dict of all methods (compare_methods)
        method = 'centroid' | 'bisector' | 'mom' | 'som' | 'lom'
                            → returns single float
        """
        self._require_table(table_name)
        reg = self._get_or_create_registry(table_name)
        mf  = reg.get(column, term)

        universe = np.linspace(mf.universe_min, mf.universe_max, 500)
        mu       = mf.compute(universe)

        if method == "all":
            return _compare_defuzz(universe, mu)
        return defuzzify(universe, mu, method=method)

    # ─────────────────────────────────────────────────────────────────────────
    # Data-driven MF suggestions
    # ─────────────────────────────────────────────────────────────────────────

    def suggest_terms(self, table_name: str, column: str,
                      n_terms: int = 3) -> list[dict]:
        """
        Suggest MF parameters for a column based on its data distribution.
        Returns a list of suggestion dicts ready to pass to define_term().
        """
        self._require_table(table_name)
        df  = self.storage.fetch_as_dataframe(table_name)
        reg = self._get_or_create_registry(table_name)
        return reg.suggest_from_data(column, df[column], n_terms=n_terms)

    # ─────────────────────────────────────────────────────────────────────────
    # Visualisation
    # ─────────────────────────────────────────────────────────────────────────

    def ascii_plot(self, table_name: str, column: str) -> str:
        """Return an ASCII plot of all MFs defined on a column."""
        self._require_table(table_name)
        reg = self._get_or_create_registry(table_name)
        return reg.ascii_plot(column)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _require_table(self, table_name: str) -> None:
        if table_name not in self.storage.list_tables():
            raise KeyError(
                f"Table '{table_name}' not found. "
                f"Available: {self.storage.list_tables()}. "
                "Use  LOAD <path> AS <table>  to load data."
            )

    def _get_or_create_registry(self, table_name: str) -> MFRegistry:
        if table_name not in self._registries:
            self._registries[table_name] = self._rebuild_registry(table_name)
        return self._registries[table_name]

    def _rebuild_registry(self, table_name: str) -> MFRegistry:
        """
        Restore an MFRegistry from the SQLite _mf_registry metadata.
        Called after loading a table, so existing MF definitions survive
        a session restart when db_path points to a real file.
        """
        reg = MFRegistry()
        all_mfs = self.storage.get_all_mfs(table_name)
        for col, mf_list in all_mfs.items():
            for entry in mf_list:
                mf_type = entry["mf_type"]
                params  = entry["params"]  # list of floats from storage

                # Reconstruct the named-param dict
                from cli.parser import MF_PARAM_KEYS
                keys = MF_PARAM_KEYS.get(mf_type, [])
                if len(keys) == len(params):
                    params_dict = dict(zip(keys, params))
                else:
                    # Fallback: store as positional if keys don't match
                    params_dict = {str(i): v for i, v in enumerate(params)}

                # Infer universe from column stats if available
                try:
                    umin, umax = self.storage.get_column_range(table_name, col)
                except Exception:
                    umin, umax = 0.0, 1.0

                reg.define(
                    column       = col,
                    term         = entry["term"],
                    mf_type      = mf_type,
                    params       = params_dict,
                    universe_min = umin,
                    universe_max = umax,
                )
        return reg

    def __repr__(self):
        tables = self.storage.list_tables()
        mf_counts = {t: len(self._get_or_create_registry(t).summary())
                     for t in tables}
        return (
            f"<FuzzyRDBEngine  db='{self.storage.db_path}'  "
            f"tables={tables}  mf_counts={mf_counts}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Hedged MF wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _HedgedMF:
    """
    Thin wrapper that applies a linguistic hedge on top of any MF.
    Presents the same .compute() / .name / .mf_type interface as
    MembershipFunction so it can be passed to FuzzyQuery.where().
    """

    def __init__(self, base_mf, hedge: str):
        from core.hedges import apply_hedge
        self._base     = base_mf
        self._hedge    = hedge
        self._apply    = apply_hedge
        self.name      = f"{hedge} {base_mf.name}"
        self.mf_type   = base_mf.mf_type
        self.params    = base_mf.params
        self.universe_min = base_mf.universe_min
        self.universe_max = base_mf.universe_max

    def compute(self, x) -> np.ndarray:
        mu = self._base.compute(x)
        return self._apply(self._hedge, mu)