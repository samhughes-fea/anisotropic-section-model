# pre_processing/section_integration/thin_walled/laminate.py
"""Classical laminate theory for the walls of a thin-walled beam section.

Each wall of a section is a laminate. This module builds its membrane,
coupling and bending stiffness (the ABD matrix) and its transverse shear
stiffness from the plies.

Wall axes
---------
- ``x``: the beam axis.
- ``s``: along the wall mid-line, from its start point to its end point.
- ``n``: through the thickness, ``n = x cross s`` (right handed).

Ply angle ``theta`` is measured from ``x`` toward ``s``. Plies are listed
from the bottom (``n = -h/2``) to the top (``n = +h/2``); the reference
surface is the wall mid-surface.

Voigt order
-----------
- Membrane strains ``[eps_xx, eps_ss, gamma_xs]``, forces ``[N_xx, N_ss, N_xs]``.
- Curvatures ``[kappa_xx, kappa_ss, kappa_xs]``, moments ``[M_xx, M_ss, M_xs]``.
- Transverse shear strains ``[gamma_xn, gamma_sn]``, forces ``[Q_x, Q_s]``.

The full wall law is ``[N; M] = ABD @ [eps; kappa]`` with
``ABD = [[A, B], [B, D]]`` and ``[Q_x, Q_s] = H @ [gamma_xn, gamma_sn]``.

Public API
----------
PlyMaterial, Ply, WallLaminate.

See Also
--------
docs/element_library/thin_walled_section_walls.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np
import numpy.typing as npt

DEFAULT_SHEAR_CORRECTION = 5.0 / 6.0


@dataclass(frozen=True)
class PlyMaterial:
    """Orthotropic ply material in its own axes (1 = fibre, 2 = transverse, 3 = normal).

    Parameters
    ----------
    E1, E2 : float
        Young's moduli along and across the fibre.
    G12 : float
        In-plane shear modulus.
    nu12 : float
        Major Poisson's ratio.
    G13, G23 : float
        Transverse shear moduli.
    rho : float, optional
        Density (only used for mass properties).
    """

    E1: float
    E2: float
    G12: float
    nu12: float
    G13: float
    G23: float
    rho: float = 0.0

    def __post_init__(self) -> None:
        for name in ("E1", "E2", "G12", "G13", "G23"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"PlyMaterial.{name} must be positive, got {getattr(self, name)}")
        if self.rho < 0.0:
            raise ValueError(f"PlyMaterial.rho must be non-negative, got {self.rho}")
        # Positive definite plane-stress compliance needs nu12 * nu21 < 1.
        if not self.nu12**2 * self.E2 / self.E1 < 1.0:
            raise ValueError(
                f"PlyMaterial with nu12={self.nu12}, E1={self.E1}, E2={self.E2} "
                "is not positive definite (needs nu12**2 * E2 / E1 < 1)"
            )

    @classmethod
    def isotropic(cls, E: float, nu: float, rho: float = 0.0) -> "PlyMaterial":
        """Isotropic material with ``G = E / (2 (1 + nu))`` in every plane."""
        G = E / (2.0 * (1.0 + nu))
        return cls(E1=E, E2=E, G12=G, nu12=nu, G13=G, G23=G, rho=rho)

    @property
    def nu21(self) -> float:
        return self.nu12 * self.E2 / self.E1

    def reduced_stiffness(self) -> npt.NDArray[np.float64]:
        """Plane-stress reduced stiffness ``Q`` (3, 3) in ply axes, order ``[eps_11, eps_22, gamma_12]``."""
        den = 1.0 - self.nu12 * self.nu21
        Q11 = self.E1 / den
        Q22 = self.E2 / den
        Q12 = self.nu12 * self.E2 / den
        return np.array(
            [[Q11, Q12, 0.0], [Q12, Q22, 0.0], [0.0, 0.0, self.G12]], dtype=np.float64
        )

    def transverse_shear_stiffness(self) -> npt.NDArray[np.float64]:
        """Transverse shear stiffness (2, 2) in ply axes, order ``[gamma_13, gamma_23]``."""
        return np.diag([self.G13, self.G23]).astype(np.float64)


def strain_rotation(theta_rad: float) -> npt.NDArray[np.float64]:
    """Engineering in-plane strain transform ``T_eps`` with ``eps_ply = T_eps @ eps_wall``.

    ``eps_wall = [eps_xx, eps_ss, gamma_xs]`` and ``eps_ply = [eps_11, eps_22, gamma_12]``,
    with fibre direction 1 at angle ``theta`` from ``x`` toward ``s``.
    """
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array(
        [
            [c * c, s * s, c * s],
            [s * s, c * c, -c * s],
            [-2.0 * c * s, 2.0 * c * s, c * c - s * s],
        ],
        dtype=np.float64,
    )


def shear_rotation(theta_rad: float) -> npt.NDArray[np.float64]:
    """Transverse shear strain transform with ``[gamma_13, gamma_23] = R @ [gamma_xn, gamma_sn]``."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


@dataclass(frozen=True)
class Ply:
    """One ply of a wall laminate.

    Parameters
    ----------
    material : PlyMaterial
    angle_deg : float
        Fibre angle in degrees, from the beam axis ``x`` toward the wall direction ``s``.
    thickness : float
        Ply thickness (positive).
    """

    material: PlyMaterial
    angle_deg: float
    thickness: float

    def __post_init__(self) -> None:
        if not self.thickness > 0.0:
            raise ValueError(f"Ply thickness must be positive, got {self.thickness}")

    @property
    def angle_rad(self) -> float:
        return float(np.deg2rad(self.angle_deg))

    def rotated_stiffness(self) -> npt.NDArray[np.float64]:
        """``Q_bar = T_eps.T @ Q @ T_eps`` (3, 3) in wall axes ``[eps_xx, eps_ss, gamma_xs]``.

        Built from strain-energy equivalence, so ``eps_wall.T @ Q_bar @ eps_wall`` equals
        ``eps_ply.T @ Q @ eps_ply`` for every strain state.
        """
        T = strain_rotation(self.angle_rad)
        return T.T @ self.material.reduced_stiffness() @ T

    def rotated_transverse_shear_stiffness(self) -> npt.NDArray[np.float64]:
        """Transverse shear stiffness (2, 2) in wall axes ``[gamma_xn, gamma_sn]``.

        Entries: ``[[G13 c^2 + G23 s^2, (G13 - G23) c s], [(G13 - G23) c s, G23 c^2 + G13 s^2]]``.
        """
        R = shear_rotation(self.angle_rad)
        return R.T @ self.material.transverse_shear_stiffness() @ R


@dataclass(frozen=True)
class WallLaminate:
    """A wall laminate: plies stacked from bottom to top about the mid-surface.

    Parameters
    ----------
    plies : sequence of Ply
        Bottom ply first.
    shear_correction : float, optional
        Factor applied to the transverse shear stiffness ``H`` (default 5/6).

    Notes
    -----
    With ply ``k`` between ``n_{k-1}`` and ``n_k`` (mid-surface at ``n = 0``)::

        A = sum Q_bar_k (n_k - n_{k-1})
        B = sum Q_bar_k (n_k^2 - n_{k-1}^2) / 2
        D = sum Q_bar_k (n_k^3 - n_{k-1}^3) / 3
        H = shear_correction * sum Qs_bar_k (n_k - n_{k-1})
    """

    plies: Tuple[Ply, ...]
    shear_correction: float = DEFAULT_SHEAR_CORRECTION
    _interfaces: npt.NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __init__(self, plies: Sequence[Ply], shear_correction: float = DEFAULT_SHEAR_CORRECTION):
        plies_t = tuple(plies)
        if not plies_t:
            raise ValueError("WallLaminate needs at least one ply")
        if not shear_correction > 0.0:
            raise ValueError(f"shear_correction must be positive, got {shear_correction}")
        object.__setattr__(self, "plies", plies_t)
        object.__setattr__(self, "shear_correction", float(shear_correction))
        h = sum(p.thickness for p in plies_t)
        z = np.concatenate([[0.0], np.cumsum([p.thickness for p in plies_t])]) - 0.5 * h
        object.__setattr__(self, "_interfaces", z)

    @classmethod
    def isotropic(cls, E: float, nu: float, thickness: float, rho: float = 0.0,
                  shear_correction: float = DEFAULT_SHEAR_CORRECTION) -> "WallLaminate":
        """Single isotropic ply of the given thickness."""
        return cls([Ply(PlyMaterial.isotropic(E, nu, rho), 0.0, thickness)], shear_correction)

    @property
    def thickness(self) -> float:
        return float(self._interfaces[-1] - self._interfaces[0])

    @property
    def interfaces(self) -> npt.NDArray[np.float64]:
        """Ply interface coordinates ``n_0 .. n_N`` (``n_0 = -h/2``, ``n_N = +h/2``)."""
        return self._interfaces.copy()

    @property
    def mass_per_area(self) -> float:
        return float(sum(p.material.rho * p.thickness for p in self.plies))

    def _moment(self, power: int) -> npt.NDArray[np.float64]:
        # the laminate is immutable, so its moments are computed once (stress recovery asks often)
        cache = self.__dict__.setdefault("_moment_cache", {})
        if power not in cache:
            z = self._interfaces
            out = np.zeros((3, 3), dtype=np.float64)
            for k, ply in enumerate(self.plies):
                out += ply.rotated_stiffness() * (z[k + 1] ** power - z[k] ** power) / power
            cache[power] = out
        return cache[power].copy()

    def A(self) -> npt.NDArray[np.float64]:
        """Membrane stiffness (3, 3)."""
        return self._moment(1)

    def B(self) -> npt.NDArray[np.float64]:
        """Membrane-bending coupling stiffness (3, 3); zero for a symmetric layup."""
        return self._moment(2)

    def D(self) -> npt.NDArray[np.float64]:
        """Bending stiffness (3, 3)."""
        return self._moment(3)

    def ABD(self) -> npt.NDArray[np.float64]:
        """Full wall stiffness (6, 6), order ``[N_xx, N_ss, N_xs, M_xx, M_ss, M_xs]``."""
        A, B, D = self.A(), self.B(), self.D()
        return np.block([[A, B], [B, D]])

    def H(self) -> npt.NDArray[np.float64]:
        """Transverse shear stiffness (2, 2), order ``[Q_x, Q_s]`` against ``[gamma_xn, gamma_sn]``."""
        z = self._interfaces
        out = np.zeros((2, 2), dtype=np.float64)
        for k, ply in enumerate(self.plies):
            out += ply.rotated_transverse_shear_stiffness() * (z[k + 1] - z[k])
        return self.shear_correction * out
