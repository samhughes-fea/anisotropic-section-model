# pre_processing/section_integration/thin_walled/wall_stiffness.py
"""Condensed wall stiffness for beam and cross-section-mode theories.

A beam theory does not carry every wall resultant. Resultants it does not
model are *released*: set to zero, with their strains left free. Static
condensation of the wall law then gives the stiffness of the strains that
are kept::

    K_c = K_kk - K_kr @ inv(K_rr) @ K_rk

which equals ``inv(C_kk)``, the inverse of the kept block of the wall
compliance ``C = inv(K)`` (see docs/proofs/section/wall_condensation.md).

Standard choices
----------------
- ``beam_wall_stiffness``: releases the hoop force ``N_ss`` and hoop moment
  ``M_ss``. Kept strains ``[eps_xx, gamma_xs, kappa_xx, kappa_xs]`` (4, 4).
  This is the wall law of classical thin-walled beam theory, where the
  section does not deform in its own plane.
- ``gbt_wall_stiffness``: releases only ``N_ss``. Kept strains
  ``[eps_xx, gamma_xs, kappa_xx, kappa_ss, kappa_xs]`` (5, 5). Used by
  cross-section distortion modes, which bend the walls in ``s``.

Public API
----------
WALL_STRAINS, WALL_RESULTANTS, WallStiffness, condense, condensed_wall_stiffness,
beam_wall_stiffness, gbt_wall_stiffness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import numpy.typing as npt

from pre_processing.section_integration.thin_walled.laminate import WallLaminate

WALL_STRAINS: Tuple[str, ...] = (
    "eps_xx", "eps_ss", "gamma_xs", "kappa_xx", "kappa_ss", "kappa_xs",
)
WALL_RESULTANTS: Tuple[str, ...] = ("N_xx", "N_ss", "N_xs", "M_xx", "M_ss", "M_xs")

BEAM_RELEASED: Tuple[str, ...] = ("N_ss", "M_ss")
GBT_RELEASED: Tuple[str, ...] = ("N_ss",)


@dataclass(frozen=True)
class WallStiffness:
    """A condensed wall stiffness and the names of its kept strains.

    Attributes
    ----------
    matrix : ndarray (m, m)
        Stiffness of the kept strains, symmetric positive definite.
    strains : tuple of str
        Kept strain names, in matrix order (a subset of ``WALL_STRAINS``).
    """

    matrix: npt.NDArray[np.float64]
    strains: Tuple[str, ...]

    def index(self, strain: str) -> int:
        return self.strains.index(strain)

    def __getitem__(self, pair: Tuple[str, str]) -> float:
        return float(self.matrix[self.index(pair[0]), self.index(pair[1])])


def condense(K: npt.NDArray[np.float64], released: Iterable[int]) -> Tuple[npt.NDArray[np.float64], Tuple[int, ...]]:
    """Statically condense out the ``released`` rows/columns of a symmetric stiffness.

    Parameters
    ----------
    K : ndarray (n, n)
        Symmetric positive definite stiffness.
    released : iterable of int
        Indices whose conjugate force is zero.

    Returns
    -------
    K_c : ndarray (m, m)
        ``K_kk - K_kr @ solve(K_rr, K_rk)``, symmetrised.
    kept : tuple of int
        Kept indices in ascending order.
    """
    K = np.asarray(K, dtype=np.float64)
    n = K.shape[0]
    if K.shape != (n, n):
        raise ValueError(f"K must be square, got shape {K.shape}")
    rel = sorted(set(int(i) for i in released))
    if any(i < 0 or i >= n for i in rel):
        raise ValueError(f"released indices {rel} out of range for size {n}")
    kept = tuple(i for i in range(n) if i not in rel)
    if not kept:
        raise ValueError("cannot release every component")
    Kkk = K[np.ix_(kept, kept)]
    if not rel:
        return 0.5 * (Kkk + Kkk.T), kept
    Krr = K[np.ix_(rel, rel)]
    Kkr = K[np.ix_(kept, rel)]
    Kc = Kkk - Kkr @ np.linalg.solve(Krr, Kkr.T)
    return 0.5 * (Kc + Kc.T), kept


def condensed_wall_stiffness(laminate: WallLaminate, released: Iterable[str] = BEAM_RELEASED) -> WallStiffness:
    """Condensed in-plane wall stiffness with the named resultants released.

    Parameters
    ----------
    laminate : WallLaminate
    released : iterable of str
        Names from ``WALL_RESULTANTS`` whose value is zero.
    """
    names = tuple(released)
    unknown = [r for r in names if r not in WALL_RESULTANTS]
    if unknown:
        raise ValueError(f"unknown wall resultants {unknown}; expected names from {WALL_RESULTANTS}")
    Kc, kept = condense(laminate.ABD(), [WALL_RESULTANTS.index(r) for r in names])
    return WallStiffness(matrix=Kc, strains=tuple(WALL_STRAINS[i] for i in kept))


def beam_wall_stiffness(laminate: WallLaminate) -> WallStiffness:
    """Wall law for thin-walled beam theory: ``N_ss = M_ss = 0``, (4, 4)."""
    return condensed_wall_stiffness(laminate, BEAM_RELEASED)


def gbt_wall_stiffness(laminate: WallLaminate) -> WallStiffness:
    """Wall law for cross-section distortion modes: ``N_ss = 0``, (5, 5)."""
    return condensed_wall_stiffness(laminate, GBT_RELEASED)
