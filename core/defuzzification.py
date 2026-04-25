"""
defuzzification.py
==================
Convert a fuzzy output set back to a single crisp number.
Used when the engine produces a fuzzy output that needs a real value.

Methods
-------
centroid       Centre of gravity (most common)
bisector       Divides area into two equal halves
mom            Mean of Maxima
som            Smallest of Maxima
lom            Largest of Maxima
weighted_avg   Weighted average (Sugeno-style)
"""

import numpy as np
from typing import Tuple


def centroid(universe: np.ndarray, membership: np.ndarray) -> float:
    """
    Centre of Gravity (COG) — most widely used.
    x* = Σ(x · μ(x)) / Σμ(x)
    """
    u = np.asarray(universe, dtype=float)
    m = np.asarray(membership, dtype=float)
    denom = np.sum(m)
    if denom < 1e-12:
        return float(np.mean(u))
    return float(np.sum(u * m) / denom)


def bisector(universe: np.ndarray, membership: np.ndarray) -> float:
    """
    Bisector — the point that divides the area under the MF curve equally.
    More robust than centroid for asymmetric sets.
    """
    u = np.asarray(universe, dtype=float)
    m = np.asarray(membership, dtype=float)
    cumulative = np.cumsum(m)
    half = cumulative[-1] / 2.0
    idx = np.searchsorted(cumulative, half)
    idx = min(idx, len(u) - 1)
    return float(u[idx])


def mean_of_maxima(universe: np.ndarray, membership: np.ndarray) -> float:
    """
    Mean of Maxima (MOM) — average of all x values achieving maximum μ.
    """
    u = np.asarray(universe, dtype=float)
    m = np.asarray(membership, dtype=float)
    max_val = np.max(m)
    if max_val < 1e-12:
        return float(np.mean(u))
    maxima_indices = np.where(np.isclose(m, max_val, atol=1e-9))[0]
    return float(np.mean(u[maxima_indices]))


def smallest_of_maxima(universe: np.ndarray, membership: np.ndarray) -> float:
    """Smallest x value achieving the maximum membership."""
    u = np.asarray(universe, dtype=float)
    m = np.asarray(membership, dtype=float)
    max_val = np.max(m)
    if max_val < 1e-12:
        return float(u[0])
    idx = np.where(np.isclose(m, max_val, atol=1e-9))[0]
    return float(u[idx[0]])


def largest_of_maxima(universe: np.ndarray, membership: np.ndarray) -> float:
    """Largest x value achieving the maximum membership."""
    u = np.asarray(universe, dtype=float)
    m = np.asarray(membership, dtype=float)
    max_val = np.max(m)
    if max_val < 1e-12:
        return float(u[-1])
    idx = np.where(np.isclose(m, max_val, atol=1e-9))[0]
    return float(u[idx[-1]])


def weighted_average(universe: np.ndarray, membership: np.ndarray) -> float:
    """
    Weighted Average — standard for Sugeno (singleton consequents).
    Same formula as centroid but semantically distinct in Sugeno systems.
    """
    return centroid(universe, membership)


# ─────────────────────────────────────────────────────────────
#  Unified dispatch
# ─────────────────────────────────────────────────────────────

DEFUZZ_METHODS = {
    "centroid"       : centroid,
    "bisector"       : bisector,
    "mom"            : mean_of_maxima,
    "som"            : smallest_of_maxima,
    "lom"            : largest_of_maxima,
    "weighted_avg"   : weighted_average,
}

DEFUZZ_DESCRIPTIONS = {
    "centroid"     : "Centre of gravity — most common, balances the whole shape",
    "bisector"     : "Splits area equally — robust for asymmetric sets",
    "mom"          : "Mean of maxima — average of peak region",
    "som"          : "Smallest of maxima — conservative estimate",
    "lom"          : "Largest of maxima — optimistic estimate",
    "weighted_avg" : "Weighted average — used in Sugeno inference",
}


def defuzzify(universe: np.ndarray, membership: np.ndarray,
              method: str = "centroid") -> float:
    """
    Defuzzify a fuzzy set to a crisp value.

    Parameters
    ----------
    universe   : array of x values
    membership : array of μ(x) values
    method     : one of centroid | bisector | mom | som | lom | weighted_avg

    Returns
    -------
    crisp float

    Example
    -------
    u = np.linspace(0, 100, 500)
    m = triangular(u, 20, 30, 50)
    x_crisp = defuzzify(u, m, method="centroid")
    """
    key = method.lower()
    if key not in DEFUZZ_METHODS:
        raise ValueError(
            f"Unknown defuzz method '{method}'. "
            f"Valid: {list(DEFUZZ_METHODS.keys())}"
        )
    return DEFUZZ_METHODS[key](universe, membership)


def compare_methods(universe: np.ndarray,
                    membership: np.ndarray) -> dict:
    """
    Run all defuzzification methods and return a comparison dict.
    Useful for the ANALYZE command to show all results side by side.
    """
    return {
        method: round(func(universe, membership), 4)
        for method, func in DEFUZZ_METHODS.items()
    }