# pre_processing/section_integration/solid/material.py
"""Three-dimensional orthotropic materials for 2D section meshes, and their rotation to section axes.

Axes
----
Section axes are the beam's: ``x`` along the beam, ``(y, z)`` in the section. A material
point has its own axes ``1`` (fibre), ``2`` and ``3``. In a laminated wall, ``s`` is the
wall direction in the section plane at angle ``alpha`` from ``y`` toward ``z``, and
``n = x cross s`` is the wall normal (the thin-walled convention, see
``thin_walled/laminate.py``). A ply at angle ``theta`` has::

    e1 = cos(theta) x + sin(theta) s,   e2 = -sin(theta) x + cos(theta) s,   e3 = n

so ``theta`` is measured from ``x`` toward ``s``, as in ``laminate.txt``.

Voigt order
-----------
Stresses and engineering strains in the order ``VOIGT``::

    [xx, yy, zz, yz, xz, xy]

The 6 x 6 stiffness in section axes is ``C = T C_123 T^T`` with ``T`` the stress
transformation (Bond) matrix of the rotation from material to section axes.

Public API
----------
VOIGT, SolidMaterial, material_axes, bond_matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

VOIGT: Tuple[str, ...] = ("xx", "yy", "zz", "yz", "xz", "xy")
_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


@dataclass(frozen=True)
class SolidMaterial:
    """Orthotropic material in its own axes ``1, 2, 3`` (engineering constants)."""

    E1: float
    E2: float
    E3: float
    G12: float
    G13: float
    G23: float
    nu12: float
    nu13: float
    nu23: float
    rho: float = 0.0

    def __post_init__(self) -> None:
        for name in ("E1", "E2", "E3", "G12", "G13", "G23"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"SolidMaterial.{name} must be positive, got {getattr(self, name)}")
        if np.linalg.eigvalsh(self.compliance())[0] <= 0.0:
            raise ValueError("SolidMaterial constants are not positive definite")

    @classmethod
    def isotropic(cls, E: float, nu: float, rho: float = 0.0) -> "SolidMaterial":
        G = E / (2.0 * (1.0 + nu))
        return cls(E, E, E, G, G, G, nu, nu, nu, rho)

    @classmethod
    def transversely_isotropic(cls, E1: float, E2: float, G12: float, nu12: float, G23: float,
                               rho: float = 0.0, nu23: Optional[float] = None) -> "SolidMaterial":
        """Isotropic in the 2-3 plane: ``E3 = E2``, ``G13 = G12``, ``nu13 = nu12`` and
        ``nu23 = E2 / (2 G23) - 1`` unless given."""
        if nu23 is None:
            nu23 = E2 / (2.0 * G23) - 1.0
        return cls(E1, E2, E2, G12, G12, G23, nu12, nu12, nu23, rho)

    @classmethod
    def from_ply(cls, ply_material, nu23: Optional[float] = None) -> "SolidMaterial":
        """From a thin-walled ``PlyMaterial`` (``E1, E2, G12, nu12, G13, G23``), with ``E3 = E2``
        and ``nu13 = nu12``. ``nu23`` defaults to ``E2 / (2 G23) - 1`` (transverse isotropy)."""
        m = ply_material
        if nu23 is None:
            nu23 = m.E2 / (2.0 * m.G23) - 1.0
        return cls(m.E1, m.E2, m.E2, m.G12, m.G13, m.G23, m.nu12, m.nu12, nu23, getattr(m, "rho", 0.0))

    def compliance(self) -> npt.NDArray[np.float64]:
        """``S`` (6, 6) in material axes, Voigt order ``[11, 22, 33, 23, 13, 12]``."""
        S = np.zeros((6, 6))
        S[0, 0], S[1, 1], S[2, 2] = 1 / self.E1, 1 / self.E2, 1 / self.E3
        S[0, 1] = S[1, 0] = -self.nu12 / self.E1
        S[0, 2] = S[2, 0] = -self.nu13 / self.E1
        S[1, 2] = S[2, 1] = -self.nu23 / self.E2
        S[3, 3], S[4, 4], S[5, 5] = 1 / self.G23, 1 / self.G13, 1 / self.G12
        return S

    def stiffness(self) -> npt.NDArray[np.float64]:
        """``C`` (6, 6) in material axes."""
        C = np.linalg.inv(self.compliance())
        return 0.5 * (C + C.T)

    def stiffness_in_section(self, theta_deg: float = 0.0, alpha_deg: float = 0.0) -> npt.NDArray[np.float64]:
        """``C`` (6, 6) in section axes for ply angle ``theta`` in a wall at angle ``alpha``."""
        T = bond_matrix(material_axes(theta_deg, alpha_deg))
        C = T @ self.stiffness() @ T.T
        return 0.5 * (C + C.T)


def material_axes(theta_deg: float, alpha_deg: float) -> npt.NDArray[np.float64]:
    """Columns ``e1, e2, e3`` in section axes ``(x, y, z)`` (see the module notes)."""
    th, al = np.deg2rad(theta_deg), np.deg2rad(alpha_deg)
    x = np.array([1.0, 0.0, 0.0])
    s = np.array([0.0, np.cos(al), np.sin(al)])
    n = np.cross(x, s)
    e1 = np.cos(th) * x + np.sin(th) * s
    e2 = -np.sin(th) * x + np.cos(th) * s
    return np.column_stack([e1, e2, n])


def bond_matrix(R: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Stress transformation ``sigma_section = T sigma_material`` for the rotation ``R``
    (columns: material axes in section axes), Voigt order ``VOIGT``.

    Engineering strains transform with ``T^-T``, so ``C_section = T C_material T^T``.
    """
    R = np.asarray(R, dtype=np.float64)
    T = np.zeros((6, 6))
    for I, (i, j) in enumerate(_PAIRS):
        for J, (k, m) in enumerate(_PAIRS):
            if k == m:
                T[I, J] = R[i, k] * R[j, m]
            else:
                T[I, J] = R[i, k] * R[j, m] + R[i, m] * R[j, k]
    return T
