"""
hedges.py
=========
Linguistic hedges — modifiers that concentrate or dilate fuzzy sets.

Hedge         Effect on μ(x)         Example
─────────────────────────────────────────────────────────────
very          μ²           concentrate  "very young" is younger
extremely     μ³           concentrate  even more restrictive
indeed        μ^1.25       slight conc. mild reinforcement
plus          μ^1.25       alias for indeed
somewhat      μ^0.5 (√μ)  dilate       "somewhat young" is broader
slightly      μ^0.3        dilate       even broader
minus         μ^0.75       alias for slightly  
not           1 - μ        complement   "not young"
more_or_less  μ^0.667      dilate       "more or less young"
"""

import numpy as np
from typing import Callable, Dict

# ─────────────────────────────────────────────────────────────
#  Core hedge functions
#  Each takes a membership array and returns a modified array.
# ─────────────────────────────────────────────────────────────

def _concentrate(mu: np.ndarray, power: float) -> np.ndarray:
    return np.clip(np.power(mu, power), 0.0, 1.0)

def _dilate(mu: np.ndarray, power: float) -> np.ndarray:
    return np.clip(np.power(mu, power), 0.0, 1.0)

HEDGE_REGISTRY: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "very"        : lambda mu: _concentrate(mu, 2.0),
    "extremely"   : lambda mu: _concentrate(mu, 3.0),
    "indeed"      : lambda mu: _concentrate(mu, 1.25),
    "plus"        : lambda mu: _concentrate(mu, 1.25),
    "somewhat"    : lambda mu: _dilate(mu, 0.5),
    "quite"       : lambda mu: _dilate(mu, 0.5),
    "slightly"    : lambda mu: _dilate(mu, 0.3),
    "minus"       : lambda mu: _dilate(mu, 0.75),
    "more_or_less": lambda mu: _dilate(mu, 0.667),
    "not"         : lambda mu: np.clip(1.0 - mu, 0.0, 1.0),
    "not_very"    : lambda mu: np.clip(1.0 - mu**2, 0.0, 1.0),
}

HEDGE_DESCRIPTIONS = {
    "very"        : "concentrates (μ²) — more restrictive",
    "extremely"   : "strongly concentrates (μ³)",
    "indeed"      : "slightly concentrates (μ^1.25)",
    "plus"        : "alias for 'indeed'",
    "somewhat"    : "dilates (√μ) — more inclusive",
    "quite"       : "alias for 'somewhat'",
    "slightly"    : "strongly dilates (μ^0.3)",
    "minus"       : "gently dilates (μ^0.75)",
    "more_or_less": "dilates moderately (μ^0.67)",
    "not"         : "complement (1 − μ)",
    "not_very"    : "complement of very (1 − μ²)",
}


def apply_hedge(hedge_name: str, membership: np.ndarray) -> np.ndarray:
    """
    Apply a named hedge to a membership array.

    Parameters
    ----------
    hedge_name  : string key from HEDGE_REGISTRY (case-insensitive)
    membership  : numpy array of membership degrees in [0, 1]

    Returns
    -------
    Modified membership array.

    Example
    -------
    mu = mf_young.compute(ages)          # base degrees
    mu_very_young = apply_hedge("very", mu)   # concentrated
    """
    key = hedge_name.lower().replace(" ", "_")
    if key not in HEDGE_REGISTRY:
        raise ValueError(
            f"Unknown hedge '{hedge_name}'. "
            f"Available: {list(HEDGE_REGISTRY.keys())}"
        )
    return HEDGE_REGISTRY[key](np.asarray(membership, dtype=float))


def apply_hedge_chain(hedge_names: list, membership: np.ndarray) -> np.ndarray:
    """
    Apply multiple hedges in sequence: "not very young" → not(very(μ)).

    Example
    -------
    apply_hedge_chain(["not", "very"], mu_young)
    """
    result = np.asarray(membership, dtype=float)
    for hedge in reversed(hedge_names):   # right-to-left like function composition
        result = apply_hedge(hedge, result)
    return result


def list_hedges() -> list:
    return [
        {"hedge": k, "effect": v}
        for k, v in HEDGE_DESCRIPTIONS.items()
    ]