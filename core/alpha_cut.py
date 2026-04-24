"""
alpha_cut.py
============
Everything related to α-cuts and the analysis of a single fuzzy set:

  1. α-cut & strong α-cut
  2. Support, Core, Height
  3. Normality check & normalisation
  4. Crossover points & Bandwidth
  5. Cardinality (scalar count & relative)
  6. α-cut Decomposition Theorem
  7. Level sets
  8. Convexity check
  9. Scalar measures: entropy (De Luca-Termini), Yager, specificity
"""

import numpy as np
from typing import Optional


# ======================================================================= #
#  1. α-cut                                                                #
# ======================================================================= #

def alpha_cut(universe, membership_values, alpha: float,
              strong: bool = False):
    """
    Compute the α-cut (or strong α-cut) of a fuzzy set.

    Regular α-cut  : A_α = {x ∈ U | μ_A(x) ≥ α}
    Strong α-cut   : A_α+ = {x ∈ U | μ_A(x) > α}

    Returns
    -------
    (elements, degrees) : arrays of universe points and their memberships
                          that satisfy the cut condition.
    """
    universe = np.asarray(universe, dtype=float)
    membership_values = np.asarray(membership_values, dtype=float)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    mask = membership_values > alpha if strong else membership_values >= alpha
    return universe[mask], membership_values[mask]


def alpha_cut_interval(universe, membership_values, alpha: float,
                       strong: bool = False):
    """
    Return the closed interval [x_min, x_max] of an α-cut,
    or None if the cut is empty.

    For a convex fuzzy set this is a single interval; for non-convex sets
    it is the bounding box of potentially disjoint regions.
    """
    elements, _ = alpha_cut(universe, membership_values, alpha, strong)
    if len(elements) == 0:
        return None
    return (float(elements.min()), float(elements.max()))


# ======================================================================= #
#  2. Support, Core, Height                                                #
# ======================================================================= #

def get_support(universe, membership_values):
    """
    Support: supp(A) = {x | μ_A(x) > 0}
    The set of all points with non-zero membership.
    """
    return alpha_cut(universe, membership_values, alpha=0.0, strong=True)


def get_core(universe, membership_values):
    """
    Core: core(A) = {x | μ_A(x) = 1}
    The set of points with full membership.
    """
    u = np.asarray(universe, dtype=float)
    m = np.asarray(membership_values, dtype=float)
    mask = np.isclose(m, 1.0)
    return u[mask], m[mask]


def get_height(membership_values) -> float:
    """Height: hgt(A) = max μ_A(x)"""
    return float(np.max(membership_values))


# ======================================================================= #
#  3. Normality                                                            #
# ======================================================================= #

def is_normal(membership_values, tol: float = 1e-9) -> bool:
    """True if max(μ) == 1 (the set has at least one fully-member element)."""
    return bool(np.max(membership_values) >= 1.0 - tol)


def normalize(membership_values) -> np.ndarray:
    """
    Scale membership values so max = 1.
    Leaves zero vectors unchanged (returns zeros).
    """
    m = np.asarray(membership_values, dtype=float)
    h = np.max(m)
    if h == 0:
        return m.copy()
    return m / h


# ======================================================================= #
#  4. Crossover points & Bandwidth                                         #
# ======================================================================= #

def get_crossover_points(universe, membership_values,
                         level: float = 0.5) -> list:
    """
    Find x values where μ_A(x) == `level` (default 0.5).
    Uses linear interpolation between adjacent samples.
    """
    u = np.asarray(universe, dtype=float)
    m = np.asarray(membership_values, dtype=float)
    diff = m - level
    crossovers = []
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:
            # Linear interpolation for sub-sample precision
            t = -diff[i] / (diff[i + 1] - diff[i])
            crossovers.append(float(u[i] + t * (u[i + 1] - u[i])))
        elif diff[i] == 0.0:
            crossovers.append(float(u[i]))
    return crossovers


def get_bandwidth(universe, membership_values, level: float = 0.5) -> Optional[float]:
    """
    Bandwidth at membership level `level` (default 0.5).
    Returns the width between the outermost crossover points,
    or None if fewer than two crossover points exist.
    """
    cp = get_crossover_points(universe, membership_values, level)
    if len(cp) >= 2:
        return float(cp[-1] - cp[0])
    return None


# ======================================================================= #
#  5. Cardinality                                                          #
# ======================================================================= #

def get_cardinality(membership_values) -> float:
    """
    Sigma-count (scalar cardinality):
    |A| = Σ μ_A(x) for all x in U
    """
    return float(np.sum(membership_values))


def get_relative_cardinality(membership_values) -> float:
    """
    Relative cardinality: |A| / |U|
    Proportion of the universe effectively covered by the fuzzy set.
    """
    m = np.asarray(membership_values, dtype=float)
    n = len(m)
    if n == 0:
        return 0.0
    return get_cardinality(m) / n


# ======================================================================= #
#  6. α-cut Decomposition Theorem                                          #
# ======================================================================= #

def decompose(universe, membership_values, resolution: int = 50):
    """
    α-cut Decomposition Theorem (Zadeh-Goguen):

        A = ⋃_{α ∈ (0,1]}  α · A_α

    where A_α is the characteristic function (0/1) of the α-cut,
    and α · A_α scales it to α.

    This theorem states that any fuzzy set can be uniquely reconstructed
    as the union of all its scaled α-cuts.

    Returns
    -------
    list of (alpha, scaled_alpha_cut_mf) tuples
    """
    alphas = np.linspace(0.0, 1.0, resolution)
    decomposition = []
    m = np.asarray(membership_values, dtype=float)
    for alpha in alphas:
        # A_α: 1 where μ ≥ α, else 0 — then scale by α
        cut_mf = np.where(m >= alpha, alpha, 0.0)
        decomposition.append((float(alpha), cut_mf))
    return decomposition


def reconstruct_from_decomposition(decomposition) -> np.ndarray:
    """
    Reconstruct the original fuzzy set from its decomposition.

    By the Decomposition Theorem:
        μ_A(x) = sup_α { α · 𝟙[μ_A(x) ≥ α] }  =  max over all α-cuts.

    If reconstruction matches original, the theorem is verified.
    """
    result = np.zeros_like(decomposition[0][1], dtype=float)
    for _, scaled_cut in decomposition:
        result = np.maximum(result, scaled_cut)
    return result


def verify_decomposition(universe, membership_values,
                         resolution: int = 200, tol: float = 1e-6) -> dict:
    """
    Decompose and reconstruct; report max reconstruction error.
    A well-sampled decomposition should have error ≈ 0.
    """
    m = np.asarray(membership_values, dtype=float)
    decomp = decompose(universe, m, resolution)
    reconstructed = reconstruct_from_decomposition(decomp)
    error = np.abs(m - reconstructed)
    return {
        "max_error"       : float(np.max(error)),
        "mean_error"      : float(np.mean(error)),
        "theorem_holds"   : bool(np.max(error) < tol),
        "original"        : m,
        "reconstructed"   : reconstructed,
    }


# ======================================================================= #
#  7. Level sets                                                            #
# ======================================================================= #

def get_level_sets(universe, membership_values,
                   levels=None) -> dict:
    """
    Compute all level sets of a fuzzy set.

    A level set at α is the crisp set { x | μ(x) ≥ α }.

    Parameters
    ----------
    levels : array-like of α values, or None (auto: unique membership values)

    Returns
    -------
    dict  {alpha: list of universe elements}
    """
    u = np.asarray(universe, dtype=float)
    m = np.asarray(membership_values, dtype=float)
    if levels is None:
        levels = np.unique(m)
    result = {}
    for alpha in levels:
        mask = m >= float(alpha)
        result[round(float(alpha), 6)] = u[mask].tolist()
    return result


# ======================================================================= #
#  8. Convexity                                                            #
# ======================================================================= #

def is_convex(universe, membership_values, tol: float = 1e-9) -> bool:
    """
    A fuzzy set A is convex if, for all x1 ≤ x ≤ x2:
        μ_A(x) ≥ min(μ_A(x1), μ_A(x2))

    For a sampled set this is checked by verifying that the MF
    never dips below the min of its two neighbours.

    Equivalently: every α-cut is a convex (interval) set.
    """
    m = np.asarray(membership_values, dtype=float)
    for i in range(1, len(m) - 1):
        if m[i] < min(m[i - 1], m[i + 1]) - tol:
            return False
    return True


# ======================================================================= #
#  9. Scalar measures                                                      #
# ======================================================================= #

def de_luca_termini_entropy(membership_values) -> float:
    """
    De Luca & Termini (1972) fuzzy entropy:

        E(A) = -1/(n·ln2) · Σ [μ·ln(μ) + (1-μ)·ln(1-μ)]

    Maximum entropy (E=1) for μ=0.5 everywhere (maximally fuzzy).
    Zero entropy (E=0) for crisp sets (μ ∈ {0,1} everywhere).
    """
    m = np.asarray(membership_values, dtype=float)
    eps = 1e-12
    m_c = np.clip(m, eps, 1.0 - eps)
    h = -(m_c * np.log(m_c) + (1.0 - m_c) * np.log(1.0 - m_c))
    return float(np.sum(h) / (len(m) * np.log(2)))


def yager_entropy(membership_values, p: float = 2.0) -> float:
    """
    Yager's fuzzy entropy:
        E_p(A) = 1 - (1/n) · ||2A - 1||_p

    p=1 gives linear entropy, p=2 gives quadratic (default).
    """
    m = np.asarray(membership_values, dtype=float)
    n = len(m)
    return float(1.0 - (1.0 / n) * np.sum(np.abs(2.0 * m - 1.0) ** p) ** (1.0 / p))


def specificity(membership_values) -> float:
    """
    Specificity of a fuzzy set:  1 for a singleton, 0 for uniform.
    S(A) = 1 - (1/n) · Σ μ(x)
    """
    m = np.asarray(membership_values, dtype=float)
    return float(1.0 - np.sum(m) / len(m))


# ======================================================================= #
#  Convenience: full analysis report                                       #
# ======================================================================= #

def analyze(universe, membership_values, name: str = "A") -> dict:
    """
    Run the complete analysis on a single fuzzy set and return a report dict.
    """
    u = np.asarray(universe, dtype=float)
    m = np.asarray(membership_values, dtype=float)

    support_u, _ = get_support(u, m)
    core_u, _    = get_core(u, m)

    return {
        "name"                 : name,
        "height"               : get_height(m),
        "is_normal"            : is_normal(m),
        "is_convex"            : is_convex(u, m),
        "support_interval"     : (float(support_u.min()), float(support_u.max()))
                                   if len(support_u) else None,
        "core_interval"        : (float(core_u.min()), float(core_u.max()))
                                   if len(core_u) else None,
        "cardinality"          : get_cardinality(m),
        "relative_cardinality" : get_relative_cardinality(m),
        "bandwidth_0.5"        : get_bandwidth(u, m, 0.5),
        "crossover_pts_0.5"    : get_crossover_points(u, m, 0.5),
        "entropy_dlt"          : de_luca_termini_entropy(m),
        "entropy_yager"        : yager_entropy(m),
        "specificity"          : specificity(m),
        "decomp_verified"      : verify_decomposition(u, m)["theorem_holds"],
    }