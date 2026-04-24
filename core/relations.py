"""
relations.py
============
Fuzzy Relations — the core of a Fuzzy Relational Database.

A fuzzy relation R on X × Y is a fuzzy set on the Cartesian product,
represented as a matrix R[i, j] ∈ [0, 1].

Contents
--------
1. FuzzyRelation class
     - Storage and labelling
     - Inverse (transpose)
     - Complement
     - Union and Intersection of relations
2. Compositions
     - Max-Min  (Zadeh / standard)
     - Max-Product
     - Max-Average
3. Relation Properties
     - Reflexivity, Symmetry, Transitivity, Equivalence
4. Transitive Closure
5. Relational operations on database tables
     - fuzzy_select   : filter rows by membership threshold
     - fuzzy_project  : keep columns
     - fuzzy_join     : join two relations via composition
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


# ======================================================================= #
#  FuzzyRelation                                                           #
# ======================================================================= #

@dataclass
class FuzzyRelation:
    """
    A fuzzy relation stored as an n×m membership matrix.

    Parameters
    ----------
    matrix      : 2-D array-like of floats in [0, 1]
    row_labels  : labels for rows  (X domain)
    col_labels  : labels for columns (Y domain)
    name        : identifier string
    """
    matrix     : np.ndarray
    row_labels : List[str] = field(default_factory=list)
    col_labels : List[str] = field(default_factory=list)
    name       : str = "R"

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=float)
        assert self.matrix.ndim == 2, "Relation matrix must be 2-D"
        self.matrix = np.clip(self.matrix, 0.0, 1.0)
        rows, cols = self.matrix.shape
        if not self.row_labels:
            self.row_labels = [f"x{i}" for i in range(rows)]
        if not self.col_labels:
            self.col_labels = [f"y{j}" for j in range(cols)]

    @property
    def shape(self):
        return self.matrix.shape

    def __getitem__(self, idx):
        return self.matrix[idx]

    # ------------------------------------------------------------------ #
    #  Basic transformations                                               #
    # ------------------------------------------------------------------ #

    def inverse(self) -> "FuzzyRelation":
        """R⁻¹ : transpose (swap row/col domains)."""
        return FuzzyRelation(
            self.matrix.T.copy(),
            self.col_labels[:],
            self.row_labels[:],
            f"{self.name}⁻¹"
        )

    def complement(self) -> "FuzzyRelation":
        """R̄ : element-wise 1 - R."""
        return FuzzyRelation(
            1.0 - self.matrix,
            self.row_labels[:],
            self.col_labels[:],
            f"¬{self.name}"
        )

    def union(self, other: "FuzzyRelation") -> "FuzzyRelation":
        """R ∪ S = max(R, S)  (element-wise)."""
        assert self.shape == other.shape, "Shapes must match for union"
        return FuzzyRelation(
            np.maximum(self.matrix, other.matrix),
            self.row_labels[:],
            self.col_labels[:],
            f"({self.name} ∪ {other.name})"
        )

    def intersection(self, other: "FuzzyRelation") -> "FuzzyRelation":
        """R ∩ S = min(R, S)  (element-wise)."""
        assert self.shape == other.shape, "Shapes must match for intersection"
        return FuzzyRelation(
            np.minimum(self.matrix, other.matrix),
            self.row_labels[:],
            self.col_labels[:],
            f"({self.name} ∩ {other.name})"
        )

    # ------------------------------------------------------------------ #
    #  Compositions                                                        #
    # ------------------------------------------------------------------ #

    def max_min_compose(self, other: "FuzzyRelation") -> "FuzzyRelation":
        """
        Max-Min (Zadeh) composition: (R ∘ S)[i, j] = max_k min(R[i,k], S[k,j])

        This is the most widely used composition in fuzzy databases and
        approximate reasoning.
        """
        R, S = self.matrix, other.matrix
        assert R.shape[1] == S.shape[0], (
            f"Incompatible dimensions: {R.shape} ∘ {S.shape}"
        )
        # Vectorised: R[:, :, None] has shape (m, k, 1), S[None] (1, k, n)
        result = np.max(np.minimum(R[:, :, None], S[None, :, :]), axis=1)
        return FuzzyRelation(result, self.row_labels[:], other.col_labels[:],
                             f"({self.name} ∘ {other.name})[max-min]")

    def max_product_compose(self, other: "FuzzyRelation") -> "FuzzyRelation":
        """
        Max-Product composition: (R ∘ S)[i, j] = max_k (R[i,k] · S[k,j])
        Useful when probabilistic AND is more appropriate than Zadeh min.
        """
        R, S = self.matrix, other.matrix
        assert R.shape[1] == S.shape[0]
        result = np.max(R[:, :, None] * S[None, :, :], axis=1)
        return FuzzyRelation(result, self.row_labels[:], other.col_labels[:],
                             f"({self.name} ∘ {other.name})[max-product]")

    def max_avg_compose(self, other: "FuzzyRelation") -> "FuzzyRelation":
        """
        Max-Average composition: (R ∘ S)[i, j] = max_k 0.5·(R[i,k] + S[k,j])
        A compromise between max-min and max-product.
        """
        R, S = self.matrix, other.matrix
        assert R.shape[1] == S.shape[0]
        result = np.max(0.5 * (R[:, :, None] + S[None, :, :]), axis=1)
        return FuzzyRelation(result, self.row_labels[:], other.col_labels[:],
                             f"({self.name} ∘ {other.name})[max-avg]")

    def compose(self, other: "FuzzyRelation",
                method: str = "max_min") -> "FuzzyRelation":
        """Dispatch composition by string."""
        if method == "max_min":
            return self.max_min_compose(other)
        elif method == "max_product":
            return self.max_product_compose(other)
        elif method == "max_avg":
            return self.max_avg_compose(other)
        raise ValueError(f"Unknown composition method: '{method}'")

    # ------------------------------------------------------------------ #
    #  Relation properties  (only meaningful for square relations)        #
    # ------------------------------------------------------------------ #

    def _check_square(self, op_name: str):
        if self.shape[0] != self.shape[1]:
            raise ValueError(f"{op_name} requires a square relation matrix.")

    def is_reflexive(self, tol: float = 1e-9) -> bool:
        """R is reflexive if R[i,i] = 1 for all i (identity is fully related)."""
        self._check_square("is_reflexive")
        return bool(np.all(np.diag(self.matrix) >= 1.0 - tol))

    def is_irreflexive(self, tol: float = 1e-9) -> bool:
        """R is irreflexive if R[i,i] = 0 for all i."""
        self._check_square("is_irreflexive")
        return bool(np.all(np.diag(self.matrix) <= tol))

    def is_symmetric(self, tol: float = 1e-9) -> bool:
        """R is symmetric if R = Rᵀ."""
        self._check_square("is_symmetric")
        return bool(np.allclose(self.matrix, self.matrix.T, atol=tol))

    def is_antisymmetric(self, tol: float = 1e-9) -> bool:
        """R is antisymmetric if R[i,j] > 0 and R[j,i] > 0 → i == j."""
        self._check_square("is_antisymmetric")
        n = self.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if self.matrix[i, j] > tol and self.matrix[j, i] > tol:
                    return False
        return True

    def is_transitive(self, tol: float = 1e-9) -> bool:
        """
        R is max-min transitive if R ∘ R ⊆ R,
        i.e., (R ∘ R)[i,j] ≤ R[i,j] + tol for all i, j.
        """
        self._check_square("is_transitive")
        composed = self.max_min_compose(self)
        return bool(np.all(composed.matrix <= self.matrix + tol))

    def is_equivalence(self) -> bool:
        """Fuzzy equivalence = reflexive + symmetric + transitive."""
        return self.is_reflexive() and self.is_symmetric() and self.is_transitive()

    def is_compatibility(self) -> bool:
        """Compatibility relation = reflexive + symmetric (not necessarily transitive)."""
        return self.is_reflexive() and self.is_symmetric()

    def properties_report(self) -> dict:
        """Return a dict of all boolean relation properties."""
        try:
            return {
                "reflexive"     : self.is_reflexive(),
                "irreflexive"   : self.is_irreflexive(),
                "symmetric"     : self.is_symmetric(),
                "antisymmetric" : self.is_antisymmetric(),
                "transitive"    : self.is_transitive(),
                "equivalence"   : self.is_equivalence(),
                "compatibility" : self.is_compatibility(),
            }
        except ValueError as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------ #
    #  Transitive closure                                                  #
    # ------------------------------------------------------------------ #

    def transitive_closure(self, max_iter: int = 100,
                           tol: float = 1e-9) -> "FuzzyRelation":
        """
        Compute the max-min transitive closure R* of this relation.

        R* is the smallest transitive fuzzy relation that contains R.
        Algorithm: iteratively compose R with itself and take the max-union
        until convergence.
        """
        self._check_square("transitive_closure")
        R_star = FuzzyRelation(self.matrix.copy(),
                               self.row_labels[:],
                               self.col_labels[:],
                               f"{self.name}*")
        for _ in range(max_iter):
            R_new = np.maximum(R_star.matrix,
                               R_star.max_min_compose(R_star).matrix)
            if np.allclose(R_new, R_star.matrix, atol=tol):
                break
            R_star.matrix = R_new
        return R_star

    # ------------------------------------------------------------------ #
    #  Output helpers                                                      #
    # ------------------------------------------------------------------ #

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.matrix,
                            index=self.row_labels,
                            columns=self.col_labels)

    def __repr__(self):
        return (f"FuzzyRelation(name='{self.name}', "
                f"shape={self.shape})")


# ======================================================================= #
#  Construction helpers                                                    #
# ======================================================================= #

def from_function(X, Y, func, name: str = "R") -> FuzzyRelation:
    """
    Build a FuzzyRelation from a Python function f(x, y) → [0,1].

    Example
    -------
    R = from_function([1,2,3], [1,2,3],
                      lambda x, y: 1 / (1 + abs(x - y)))
    """
    matrix = np.array([[func(x, y) for y in Y] for x in X], dtype=float)
    xl = [str(x) for x in X]
    yl = [str(y) for y in Y]
    return FuzzyRelation(matrix, xl, yl, name)


def from_mf_pair(universe_A, mf_A, universe_B, mf_B,
                 tnorm_func=None, name: str = "R") -> FuzzyRelation:
    """
    Cartesian-product relation using a T-norm:
        R(x, y) = T(μ_A(x), μ_B(y))

    Defaults to min T-norm.
    """
    if tnorm_func is None:
        tnorm_func = np.minimum

    mA = np.asarray(mf_A, dtype=float)
    mB = np.asarray(mf_B, dtype=float)
    matrix = tnorm_func(mA[:, None], mB[None, :])
    return FuzzyRelation(matrix,
                         [str(x) for x in universe_A],
                         [str(y) for y in universe_B],
                         name)


# ======================================================================= #
#  Relational DB operations on pandas DataFrames with fuzzy degrees       #
# ======================================================================= #

def fuzzy_select(df: pd.DataFrame,
                 membership_col: str,
                 threshold: float = 0.0,
                 strong: bool = False) -> pd.DataFrame:
    """
    Select rows whose fuzzy membership degree satisfies the threshold.

    Parameters
    ----------
    df             : DataFrame that contains a membership_col column
    membership_col : column name holding per-row membership degrees
    threshold      : α value
    strong         : if True use strict inequality (> instead of ≥)
    """
    m = df[membership_col].values
    mask = m > threshold if strong else m >= threshold
    return df[mask].copy()


def fuzzy_project(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Keep only the specified columns (standard relational project)."""
    return df[columns].copy()


def fuzzy_join(df_left: pd.DataFrame,
               df_right: pd.DataFrame,
               on: str,
               membership_col_left: str,
               membership_col_right: str,
               composition: str = "max_min") -> pd.DataFrame:
    """
    Join two fuzzy tables on a key column and combine their membership
    degrees using the specified composition.

    The resulting membership is computed row-wise after merging:
      max_min : min(μ_left, μ_right)
      max_prod: μ_left × μ_right
      max_avg : 0.5 × (μ_left + μ_right)
    """
    merged = pd.merge(df_left, df_right, on=on,
                      suffixes=("_L", "_R"))
    mL = merged[membership_col_left + "_L"].values
    mR = merged[membership_col_right + "_R"].values

    if composition == "max_min":
        merged["membership"] = np.minimum(mL, mR)
    elif composition == "max_product":
        merged["membership"] = mL * mR
    elif composition == "max_avg":
        merged["membership"] = 0.5 * (mL + mR)
    else:
        raise ValueError(f"Unknown composition: '{composition}'")

    return merged.sort_values("membership", ascending=False)