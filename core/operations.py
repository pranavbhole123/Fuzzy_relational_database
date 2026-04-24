"""
operations.py
=============
Fuzzy set operations:
  - T-norms  (fuzzy AND)
  - S-norms / T-conorms  (fuzzy OR)
  - Complement families
  - Set-level: union, intersection, difference, symmetric difference
  - Aggregation operators for combining multiple membership degrees
"""

import numpy as np
from enum import Enum


# ======================================================================= #
#  T-norm (fuzzy AND) families                                             #
# ======================================================================= #

class TNorm(Enum):
    MINIMUM          = "minimum"           # Zadeh (classical)
    ALGEBRAIC_PRODUCT = "algebraic_product" # Probabilistic AND
    BOUNDED_PRODUCT  = "bounded_product"   # Lukasiewicz
    DRASTIC_PRODUCT  = "drastic_product"   # Strongest negation
    HAMACHER         = "hamacher"          # Parameterized family
    EINSTEIN         = "einstein"          # Einstein product


def t_norm(a, b, method: TNorm = TNorm.MINIMUM) -> np.ndarray:
    """
    Apply a T-norm (fuzzy AND) to membership degrees a and b.

    T-norms must satisfy: commutativity, associativity,
    monotonicity, and T(x,1)=x (boundary condition).

    Ordering (weakest → strongest):
      drastic ≤ bounded ≤ Hamacher ≤ Einstein ≤ algebraic ≤ minimum
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if method == TNorm.MINIMUM:
        return np.minimum(a, b)

    elif method == TNorm.ALGEBRAIC_PRODUCT:
        return a * b

    elif method == TNorm.BOUNDED_PRODUCT:
        return np.maximum(0.0, a + b - 1.0)

    elif method == TNorm.DRASTIC_PRODUCT:
        result = np.zeros_like(a)
        mask_b1 = b == 1.0
        mask_a1 = a == 1.0
        result[mask_b1] = a[mask_b1]
        result[mask_a1] = b[mask_a1]
        return result

    elif method == TNorm.HAMACHER:
        denom = a + b - a * b
        return np.where(denom == 0, 0.0, (a * b) / denom)

    elif method == TNorm.EINSTEIN:
        return (a * b) / (2.0 - (a + b - a * b))

    raise ValueError(f"Unknown T-norm: {method}")


# ======================================================================= #
#  S-norm / T-conorm (fuzzy OR) families                                  #
# ======================================================================= #

class SNorm(Enum):
    MAXIMUM          = "maximum"           # Zadeh (classical)
    ALGEBRAIC_SUM    = "algebraic_sum"     # Probabilistic OR
    BOUNDED_SUM      = "bounded_sum"       # Lukasiewicz
    DRASTIC_SUM      = "drastic_sum"       # Strongest
    HAMACHER         = "hamacher"
    EINSTEIN         = "einstein"


def s_norm(a, b, method: SNorm = SNorm.MAXIMUM) -> np.ndarray:
    """
    Apply an S-norm (fuzzy OR / T-conorm) to membership degrees a and b.

    Every T-norm T has a dual S-norm: S(a,b) = 1 - T(1-a, 1-b).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if method == SNorm.MAXIMUM:
        return np.maximum(a, b)

    elif method == SNorm.ALGEBRAIC_SUM:
        return a + b - a * b

    elif method == SNorm.BOUNDED_SUM:
        return np.minimum(1.0, a + b)

    elif method == SNorm.DRASTIC_SUM:
        result = np.ones_like(a)
        mask_b0 = b == 0.0
        mask_a0 = a == 0.0
        result[mask_b0] = a[mask_b0]
        result[mask_a0] = b[mask_a0]
        return result

    elif method == SNorm.HAMACHER:
        denom = 1.0 - a * b
        return np.where(denom == 0, 1.0, (a + b - 2.0 * a * b) / denom)

    elif method == SNorm.EINSTEIN:
        return (a + b) / (1.0 + a * b)

    raise ValueError(f"Unknown S-norm: {method}")


# ======================================================================= #
#  Complement (fuzzy NOT) families                                         #
# ======================================================================= #

def complement(a, method: str = "standard", **kwargs) -> np.ndarray:
    """
    Fuzzy complement of membership array a.

    Methods
    -------
    standard  : c(a) = 1 - a                (Zadeh)
    sugeno    : c(a) = (1 - a) / (1 + λa)   requires kwarg lambda_ (default -0.5)
    yager     : c(a) = (1 - a^w)^(1/w)      requires kwarg w (default 2)
    """
    a = np.asarray(a, dtype=float)
    if method == "standard":
        return 1.0 - a
    elif method == "sugeno":
        lam = kwargs.get("lambda_", -0.5)
        return (1.0 - a) / (1.0 + lam * a)
    elif method == "yager":
        w = kwargs.get("w", 2.0)
        return (1.0 - a ** w) ** (1.0 / w)
    raise ValueError(f"Unknown complement method: {method}")


# ======================================================================= #
#  High-level fuzzy set operations                                         #
#  All operate element-wise on membership arrays of the same length.      #
# ======================================================================= #

def fuzzy_union(A, B, method: SNorm = SNorm.MAXIMUM) -> np.ndarray:
    """A ∪ B using the chosen S-norm."""
    return s_norm(A, B, method)


def fuzzy_intersection(A, B, method: TNorm = TNorm.MINIMUM) -> np.ndarray:
    """A ∩ B using the chosen T-norm."""
    return t_norm(A, B, method)


def fuzzy_complement(A, method: str = "standard", **kwargs) -> np.ndarray:
    """Ā (complement of A)."""
    return complement(A, method, **kwargs)


def fuzzy_difference(A, B,
                     tnorm: TNorm = TNorm.MINIMUM,
                     comp_method: str = "standard") -> np.ndarray:
    """A \\ B  =  A ∩ (NOT B)"""
    return fuzzy_intersection(A, fuzzy_complement(B, comp_method), tnorm)


def fuzzy_symmetric_difference(A, B,
                                tnorm: TNorm = TNorm.MINIMUM,
                                snorm: SNorm = SNorm.MAXIMUM,
                                comp_method: str = "standard") -> np.ndarray:
    """(A \\ B) ∪ (B \\ A)"""
    return fuzzy_union(
        fuzzy_difference(A, B, tnorm, comp_method),
        fuzzy_difference(B, A, tnorm, comp_method),
        snorm
    )


def cartesian_product(A, B, tnorm: TNorm = TNorm.MINIMUM) -> np.ndarray:
    """
    Fuzzy Cartesian product: R(x, y) = T(μ_A(x), μ_B(y)).
    Returns a 2-D matrix of shape (len(A), len(B)).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # Broadcast: A as column, B as row
    return t_norm(A[:, None], B[None, :], tnorm)


# ======================================================================= #
#  Aggregation operators                                                   #
#  Combine a vector of membership degrees into a single scalar.           #
# ======================================================================= #

class AggMethod(Enum):
    MIN            = "min"
    MAX            = "max"
    MEAN           = "mean"
    PRODUCT        = "product"
    BOUNDED_SUM    = "bounded_sum"
    GEOMETRIC_MEAN = "geometric_mean"
    HARMONIC_MEAN  = "harmonic_mean"
    OWA            = "owa"       # Ordered Weighted Averaging (requires weights)
    WEIGHTED_MEAN  = "weighted_mean"  # requires weights


def aggregate(memberships, method: str = "min", weights=None) -> float:
    """
    Aggregate a list/array of membership degrees into a single value.

    Parameters
    ----------
    memberships : array-like of floats in [0, 1]
    method      : one of AggMethod values (as string)
    weights     : required for 'owa' and 'weighted_mean'

    Returns
    -------
    float in [0, 1]
    """
    m = np.asarray(memberships, dtype=float)
    n = len(m)

    if method == "min":
        return float(np.min(m))

    elif method == "max":
        return float(np.max(m))

    elif method == "mean":
        return float(np.mean(m))

    elif method == "product":
        return float(np.prod(m))

    elif method == "bounded_sum":
        return float(min(1.0, np.sum(m)))

    elif method == "geometric_mean":
        return float(np.prod(m) ** (1.0 / n)) if n > 0 else 0.0

    elif method == "harmonic_mean":
        safe = np.maximum(m, 1e-10)
        return float(n / np.sum(1.0 / safe))

    elif method == "owa":
        if weights is None:
            raise ValueError("OWA aggregation requires 'weights'.")
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()  # normalise
        sorted_m = np.sort(m)[::-1]  # descending
        return float(np.dot(w, sorted_m[:len(w)]))

    elif method == "weighted_mean":
        if weights is None:
            raise ValueError("weighted_mean requires 'weights'.")
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        return float(np.dot(w, m))

    raise ValueError(
        f"Unknown aggregation method: '{method}'. "
        f"Valid: {[e.value for e in AggMethod]}"
    )


# ======================================================================= #
#  Implication operators (used in fuzzy rule evaluation)                  #
# ======================================================================= #

def implication(a: float, b: float, method: str = "mamdani") -> float:
    """
    Fuzzy implication: A → B.

    Methods
    -------
    mamdani   : min(a, b)          — truncation
    larsen    : a * b              — scaling
    lukasiewicz : min(1, 1-a+b)
    zadeh     : max(min(a,b), 1-a)
    """
    if method == "mamdani":
        return float(min(a, b))
    elif method == "larsen":
        return float(a * b)
    elif method == "lukasiewicz":
        return float(min(1.0, 1.0 - a + b))
    elif method == "zadeh":
        return float(max(min(a, b), 1.0 - a))
    raise ValueError(f"Unknown implication: {method}")