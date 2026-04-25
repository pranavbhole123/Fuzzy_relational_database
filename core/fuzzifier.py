"""
fuzzifier.py
============
The MF Registry — the central brain of the engine.

Responsibilities
----------------
1. Store and retrieve MembershipFunction objects keyed by (column, term)
2. Resolve a query condition (column, hedge, term) → membership array
3. Suggest MF parameters from data statistics
4. Serialize / deserialize the registry to JSON for session persistence
5. ASCII-plot MFs in the terminal
"""

import json
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from membership_functions import MembershipFunction, make_mf, MFType, MF_PARAM_GUIDE
from hedges import apply_hedge, apply_hedge_chain, HEDGE_REGISTRY
from defuzzification import defuzzify, compare_methods


# ─────────────────────────────────────────────────────────────
#  MF Registry
# ─────────────────────────────────────────────────────────────

class MFRegistry:
    """
    Central store of all membership functions in the current session.

    Structure: { column_name: { term_name: MembershipFunction } }

    Example
    -------
    reg = MFRegistry()
    reg.define("age", "young", "triangular", {"a":15,"b":25,"c":35}, 0, 100)
    mu = reg.compute("age", "young", ages_array)
    mu_very = reg.compute("age", "young", ages_array, hedge="very")
    """

    def __init__(self):
        # { col: { term: MembershipFunction } }
        self._registry: dict = {}

    # ─────────────────────────────────────────────────────
    #  Registration
    # ─────────────────────────────────────────────────────

    def define(self, column: str, term: str, mf_type: str,
               params: dict, universe_min: float, universe_max: float):
        """Add or overwrite an MF for (column, term)."""
        mf = make_mf(
            name=term,
            mf_type=mf_type,
            params=params,
            universe_min=universe_min,
            universe_max=universe_max,
        )
        self._registry.setdefault(column, {})[term] = mf
        return mf

    def delete(self, column: str, term: str):
        """Remove an MF. Raises KeyError if not found."""
        if column not in self._registry or term not in self._registry[column]:
            raise KeyError(f"No MF defined for {column}.{term}")
        del self._registry[column][term]
        if not self._registry[column]:
            del self._registry[column]

    # ─────────────────────────────────────────────────────
    #  Lookup
    # ─────────────────────────────────────────────────────

    def get(self, column: str, term: str) -> MembershipFunction:
        try:
            return self._registry[column][term]
        except KeyError:
            available = self.list_terms(column)
            raise KeyError(
                f"No MF for '{column}.{term}'. "
                f"Defined terms for '{column}': {available or 'none'}"
            )

    def list_columns(self) -> list:
        return list(self._registry.keys())

    def list_terms(self, column: str) -> list:
        return list(self._registry.get(column, {}).keys())

    def has(self, column: str, term: str) -> bool:
        return column in self._registry and term in self._registry[column]

    # ─────────────────────────────────────────────────────
    #  Compute membership (with optional hedge)
    # ─────────────────────────────────────────────────────

    def compute(self, column: str, term: str,
                values, hedge: str = None,
                hedge_chain: list = None) -> np.ndarray:
        """
        Compute membership degrees for `values` under (column, term),
        optionally modified by a hedge or a chain of hedges.

        Parameters
        ----------
        column      : e.g. "age"
        term        : e.g. "young"
        values      : scalar or array of crisp values
        hedge       : single hedge name, e.g. "very"
        hedge_chain : list of hedge names applied right-to-left
                      e.g. ["not", "very"]  →  not(very(μ))

        Returns
        -------
        np.ndarray of membership degrees in [0, 1]
        """
        mf = self.get(column, term)
        mu = mf.compute(np.asarray(values, dtype=float))

        if hedge_chain:
            mu = apply_hedge_chain(hedge_chain, mu)
        elif hedge:
            from hedges import apply_hedge
            mu = apply_hedge(hedge, mu)

        return mu

    # ─────────────────────────────────────────────────────
    #  Data-driven MF suggestion
    # ─────────────────────────────────────────────────────

    def suggest_from_data(self, column: str, series,
                          n_terms: int = 3,
                          term_names: list = None) -> list:
        """
        Suggest MF parameters by dividing the column's range into
        n_terms overlapping triangular MFs based on percentiles.

        Returns list of suggestion dicts (does NOT auto-register).

        Typical use: show suggestions to the user, let them confirm.
        """
        vals = np.asarray(series.dropna(), dtype=float)
        col_min, col_max = float(vals.min()), float(vals.max())

        if term_names is None:
            defaults = {
                1: ["main"],
                2: ["low", "high"],
                3: ["low", "medium", "high"],
                4: ["very_low", "low", "high", "very_high"],
                5: ["very_low", "low", "medium", "high", "very_high"],
            }
            term_names = defaults.get(n_terms,
                         [f"term_{i+1}" for i in range(n_terms)])

        percentiles = np.linspace(0, 100, n_terms + 2)[1:-1]
        peaks = np.percentile(vals, percentiles)
        suggestions = []

        for i, (term, peak) in enumerate(zip(term_names, peaks)):
            left  = peaks[i - 1] if i > 0           else col_min
            right = peaks[i + 1] if i < n_terms - 1 else col_max
            suggestions.append({
                "column"       : column,
                "term"         : term,
                "mf_type"      : "triangular",
                "params"       : {"a": round(left, 2),
                                  "b": round(peak, 2),
                                  "c": round(right, 2)},
                "universe_min" : col_min,
                "universe_max" : col_max,
            })
        return suggestions

    # ─────────────────────────────────────────────────────
    #  ASCII plot
    # ─────────────────────────────────────────────────────

    def ascii_plot(self, column: str, width: int = 60,
                   height: int = 10) -> str:
        """
        Draw an ASCII chart of all MFs defined for `column`.
        Returns a multi-line string ready for print().
        """
        terms = self._registry.get(column, {})
        if not terms:
            return f"No MFs defined for column '{column}'."

        # Union of all universes
        u_min = min(mf.universe_min for mf in terms.values())
        u_max = max(mf.universe_max for mf in terms.values())
        universe = np.linspace(u_min, u_max, width)

        # Characters for multiple MFs
        chars = ['#', '@', '*', '+', 'x', 'o', '=', '~']
        lines = []

        # Build a 2-D grid: row 0 = top (μ=1), row height-1 = bottom (μ=0)
        grid = [[' '] * width for _ in range(height)]

        for idx, (term, mf) in enumerate(terms.items()):
            mu = mf.compute(universe)
            ch = chars[idx % len(chars)]
            for col_i, mu_val in enumerate(mu):
                row_i = int((1.0 - mu_val) * (height - 1))
                row_i = max(0, min(height - 1, row_i))
                if grid[row_i][col_i] == ' ':
                    grid[row_i][col_i] = ch

        # Y-axis
        lines.append(f"\n  MFs for column: {column!r}")
        lines.append(f"  Universe: [{u_min:.1f}, {u_max:.1f}]")
        lines.append("  " + "─" * (width + 4))
        for row_i, row in enumerate(grid):
            mu_label = f"{1.0 - row_i/(height-1):.1f}"
            lines.append(f"  {mu_label} │{''.join(row)}│")
        lines.append("  " + "─" * (width + 4))
        # X-axis labels
        x_label_left  = f"{u_min:.0f}"
        x_label_right = f"{u_max:.0f}"
        mid_space = width - len(x_label_left) - len(x_label_right)
        lines.append("       " + x_label_left + " " * max(0, mid_space) + x_label_right)

        # Legend
        lines.append("")
        for idx, (term, mf) in enumerate(terms.items()):
            ch = chars[idx % len(chars)]
            lines.append(f"  {ch}  {term}  [{mf.mf_type.value}  {mf.params}]")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────
    #  Serialization
    # ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the entire registry to a JSON-safe dict."""
        result = {}
        for col, terms in self._registry.items():
            result[col] = {}
            for term, mf in terms.items():
                result[col][term] = {
                    "mf_type"     : mf.mf_type.value,
                    "params"      : mf.params,
                    "universe_min": mf.universe_min,
                    "universe_max": mf.universe_max,
                }
        return result

    def from_dict(self, data: dict):
        """Restore registry from a serialized dict (e.g. loaded from JSON)."""
        self._registry = {}
        for col, terms in data.items():
            for term, cfg in terms.items():
                self.define(
                    column=col, term=term,
                    mf_type=cfg["mf_type"],
                    params=cfg["params"],
                    universe_min=cfg["universe_min"],
                    universe_max=cfg["universe_max"],
                )

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            self.from_dict(json.load(f))

    # ─────────────────────────────────────────────────────
    #  Summary
    # ─────────────────────────────────────────────────────

    def summary(self) -> list:
        rows = []
        for col, terms in self._registry.items():
            for term, mf in terms.items():
                rows.append({
                    "column"  : col,
                    "term"    : term,
                    "type"    : mf.mf_type.value,
                    "params"  : str(mf.params),
                    "universe": f"[{mf.universe_min}, {mf.universe_max}]",
                })
        return rows

    def __repr__(self):
        n = sum(len(v) for v in self._registry.values())
        return f"MFRegistry({n} MFs across {len(self._registry)} columns)"