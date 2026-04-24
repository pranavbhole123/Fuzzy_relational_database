"""
membership_functions.py
=======================
All supported fuzzy membership function types.
Each MF takes crisp values and returns membership degrees in [0, 1].
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum


class MFType(Enum):
    TRIANGULAR       = "triangular"
    TRAPEZOIDAL      = "trapezoidal"
    GAUSSIAN         = "gaussian"
    GENERALIZED_BELL = "generalized_bell"
    SIGMOID          = "sigmoid"
    S_SHAPED         = "s_shaped"
    Z_SHAPED         = "z_shaped"
    PI_SHAPED        = "pi_shaped"
    SINGLETON        = "singleton"
    CUSTOM           = "custom"


@dataclass
class MembershipFunction:
    """
    A fuzzy membership function bound to a universe of discourse.

    Parameters
    ----------
    name          : Human-readable label (e.g. "young", "high_salary")
    mf_type       : One of MFType enum values
    params        : Dict of keyword args forwarded to the raw MF (e.g. {'a':0,'b':25,'c':35})
    universe_min  : Left boundary of the universe
    universe_max  : Right boundary of the universe
    _func         : Required only when mf_type == MFType.CUSTOM
    """
    name          : str
    mf_type       : MFType
    params        : dict
    universe_min  : float
    universe_max  : float
    _func         : Optional[Callable] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    #  Core interface                                                       #
    # ------------------------------------------------------------------ #

    def compute(self, x) -> np.ndarray:
        """Return membership degree(s) for scalar or array input x."""
        x = np.asarray(x, dtype=float)
        dispatch = {
            MFType.TRIANGULAR       : lambda: triangular(x, **self.params),
            MFType.TRAPEZOIDAL      : lambda: trapezoidal(x, **self.params),
            MFType.GAUSSIAN         : lambda: gaussian(x, **self.params),
            MFType.GENERALIZED_BELL : lambda: generalized_bell(x, **self.params),
            MFType.SIGMOID          : lambda: sigmoid(x, **self.params),
            MFType.S_SHAPED         : lambda: s_shaped(x, **self.params),
            MFType.Z_SHAPED         : lambda: z_shaped(x, **self.params),
            MFType.PI_SHAPED        : lambda: pi_shaped(x, **self.params),
            MFType.SINGLETON        : lambda: singleton(x, **self.params),
            MFType.CUSTOM           : lambda: self._func(x),
        }
        if self.mf_type not in dispatch:
            raise ValueError(f"Unknown MF type: {self.mf_type}")
        return dispatch[self.mf_type]()

    def __call__(self, x) -> np.ndarray:
        return self.compute(x)

    # ------------------------------------------------------------------ #
    #  Universe helpers                                                     #
    # ------------------------------------------------------------------ #

    def get_universe(self, resolution: int = 1000) -> np.ndarray:
        return np.linspace(self.universe_min, self.universe_max, resolution)

    def get_curve(self, resolution: int = 1000):
        """Return (universe, membership_values) for plotting."""
        u = self.get_universe(resolution)
        return u, self.compute(u)

    # ------------------------------------------------------------------ #
    #  Introspection                                                        #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        u, m = self.get_curve()
        return {
            "name"        : self.name,
            "type"        : self.mf_type.value,
            "params"      : self.params,
            "universe"    : (self.universe_min, self.universe_max),
            "height"      : float(np.max(m)),
            "support_min" : float(u[m > 0][0])  if np.any(m > 0) else None,
            "support_max" : float(u[m > 0][-1]) if np.any(m > 0) else None,
        }

    def __repr__(self):
        return (f"MF(name='{self.name}', type={self.mf_type.value}, "
                f"universe=[{self.universe_min}, {self.universe_max}])")


# ======================================================================= #
#  Raw membership function implementations                                 #
#  All accept numpy arrays and return values clipped to [0, 1]            #
# ======================================================================= #

def triangular(x, a: float, b: float, c: float) -> np.ndarray:
    """
    Triangular MF — peaks at b, zero at a (left foot) and c (right foot).

        a     b     c
        |    /|\\    |
        |   / | \\   |
        |  /  |  \\  |
        | /   |   \\ |
        |/    |    \\|
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    if b != a:
        m1 = (x > a) & (x <= b)
        result[m1] = (x[m1] - a) / (b - a)
    if c != b:
        m2 = (x > b) & (x < c)
        result[m2] = (c - x[m2]) / (c - b)
    result[x == b] = 1.0
    return np.clip(result, 0.0, 1.0)


def trapezoidal(x, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Trapezoidal MF — flat top between b and c.

        a   b       c   d
        |  /|‾‾‾‾‾‾‾|\\  |
        | / |       | \\ |
        |/  |       |  \\|
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    if b != a:
        m1 = (x >= a) & (x < b)
        result[m1] = (x[m1] - a) / (b - a)
    m2 = (x >= b) & (x <= c)
    result[m2] = 1.0
    if d != c:
        m3 = (x > c) & (x <= d)
        result[m3] = (d - x[m3]) / (d - c)
    return np.clip(result, 0.0, 1.0)


def gaussian(x, mean: float, sigma: float) -> np.ndarray:
    """
    Gaussian (normal bell-curve) MF centred at mean with spread sigma.
    μ(x) = exp(-0.5 * ((x - mean)/sigma)^2)
    """
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def generalized_bell(x, a: float, b: float, c: float) -> np.ndarray:
    """
    Generalized Bell MF — smooth, shoulder-like shape.
    μ(x) = 1 / (1 + |((x - c) / a)|^(2b))
      a : half-width at the crossover points
      b : slope (larger → steeper sides)
      c : centre
    """
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.abs((x - c) / a) ** (2 * b))


def sigmoid(x, c: float, a: float) -> np.ndarray:
    """
    Sigmoid MF.
      c : crossover point (where μ = 0.5)
      a : slope  (positive → increasing S,  negative → decreasing Z)
    """
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-a * (x - c)))


def s_shaped(x, a: float, b: float) -> np.ndarray:
    """
    S-shaped MF: 0 at a, rises smoothly, 1 at b.
    Good for modelling "large", "old", "high".
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    mid = (a + b) / 2.0

    m1 = (x >= a) & (x <= mid)
    result[m1] = 2.0 * ((x[m1] - a) / (b - a)) ** 2

    m2 = (x > mid) & (x <= b)
    result[m2] = 1.0 - 2.0 * ((x[m2] - b) / (b - a)) ** 2

    result[x > b] = 1.0
    return np.clip(result, 0.0, 1.0)


def z_shaped(x, a: float, b: float) -> np.ndarray:
    """
    Z-shaped MF: mirror of s_shaped. 1 at a, falls to 0 at b.
    Good for modelling "small", "young", "low".
    """
    return 1.0 - s_shaped(x, a, b)


def pi_shaped(x, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Pi-shaped MF: combination of S (rising) and Z (falling).
    Good for symmetric, bounded linguistic concepts.
    """
    return s_shaped(x, a, b) * z_shaped(x, c, d)


def singleton(x, center: float, tolerance: float = 1e-9) -> np.ndarray:
    """
    Singleton MF: exactly 1 at center, 0 everywhere else.
    Used in Sugeno-type inference systems.
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    result[np.abs(x - center) < tolerance] = 1.0
    return result


# ======================================================================= #
#  Factory / registry helpers                                               #
# ======================================================================= #

def make_mf(name: str, mf_type: str, params: dict,
            universe_min: float, universe_max: float,
            custom_func: Optional[Callable] = None) -> MembershipFunction:
    """
    Convenience factory — accepts mf_type as a string.

    Example
    -------
    mf = make_mf("young", "triangular", {"a": 0, "b": 20, "c": 35}, 0, 100)
    print(mf.compute(25))   # → 0.666...
    """
    try:
        t = MFType(mf_type.lower())
    except ValueError:
        raise ValueError(
            f"Unknown mf_type '{mf_type}'. "
            f"Valid options: {[e.value for e in MFType]}"
        )
    return MembershipFunction(
        name=name,
        mf_type=t,
        params=params,
        universe_min=universe_min,
        universe_max=universe_max,
        _func=custom_func,
    )


MF_PARAM_GUIDE = {
    "triangular"       : "a (left foot), b (peak), c (right foot)",
    "trapezoidal"      : "a (left foot), b (left shoulder), c (right shoulder), d (right foot)",
    "gaussian"         : "mean (centre), sigma (spread)",
    "generalized_bell" : "a (half-width), b (slope), c (centre)",
    "sigmoid"          : "c (crossover), a (slope; +ve=rising, -ve=falling)",
    "s_shaped"         : "a (start), b (end)",
    "z_shaped"         : "a (start), b (end)",
    "pi_shaped"        : "a (left start), b (left end), c (right start), d (right end)",
    "singleton"        : "center (exact point), tolerance (default 1e-9)",
    "custom"           : "pass custom_func to make_mf()",
}