# pre_processing/section_integration/section_analysis.py
"""Section constitutive data shared by every beam element: ``SectionAnalysis``.

A ``SectionAnalysis`` holds the section stiffness that elements turn into
their ``D`` matrices. It is built either from the classical isotropic inputs
(``section.txt`` and ``material.txt``) or from a thin-walled anisotropic
section analysis (``laminate.txt``, ``section_geometry.txt`` and
``element_section.txt``).

Element Voigt order and signs
-----------------------------
``stiffness`` is in the element Voigt order
``[eps_x, kappa_y, kappa_z, gamma_xy, gamma_xz, phi_x]`` with the signs the
element ``B`` matrices use. Probing the linear Euler-Bernoulli and Timoshenko
``B`` operators with right-hand-rule nodal rotations shows::

    eps_x    = u_x'
    kappa_y  = -theta_y'        (= u_z'' for Euler-Bernoulli)
    kappa_z  = theta_z'         (= u_y'')
    gamma_xy = u_y' - theta_z
    gamma_xz = u_z' + theta_y
    phi_x    = theta_x'

so the element strains are ``S @ psi`` with ``psi`` the right-handed
strains of ``thin_walled.section_analysis`` and ``S = diag(1, -1, 1, 1, 1, 1)``.
The element stiffness is ``S @ K_rh @ S`` (``S`` is its own inverse). For an
isotropic section with no couplings the two are identical.

Public API
----------
ELEMENT_SIGN, SectionAnalysis, to_element_voigt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

ELEMENT_SIGN = np.diag([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
CLASSICAL_INDEX = (0, 1, 2, 5)  # [eps_x, kappa_y, kappa_z, phi_x]


def to_element_voigt(K_rh: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Map a right-handed 6 x 6 section stiffness to the element sign convention."""
    K = np.asarray(K_rh, dtype=np.float64)
    if K.shape != (6, 6):
        raise ValueError(f"expected a (6, 6) stiffness, got {K.shape}")
    return ELEMENT_SIGN @ K @ ELEMENT_SIGN


@dataclass(frozen=True)
class SectionAnalysis:
    """Section stiffness and properties in the element convention.

    Attributes
    ----------
    stiffness : ndarray (6, 6)
        Timoshenko section stiffness, element Voigt order and signs.
    classical_stiffness : ndarray (4, 4)
        Shear-rigid stiffness for ``[eps_x, kappa_y, kappa_z, phi_x]``
        (inverse of the force-flexibility block, not a sub-block of ``stiffness``).
    warping_stiffness : float
        Vlasov ``E Gamma`` about the shear centre.
    shear_centre, tension_centre : (float, float)
        In section coordinates ``(y, z)`` about the beam reference axis.
    mass : dict
        ``m``, ``y_m``, ``z_m``, ``I_yy``, ``I_zz``, ``I_yz``, ``I_p`` per length.
    source : str
        ``"isotropic"`` or ``"thin_walled"``.
    section_id : int or None
        Thin-walled section id, when ``source == "thin_walled"``.
    """

    stiffness: npt.NDArray[np.float64]
    classical_stiffness: npt.NDArray[np.float64]
    warping_stiffness: float = 0.0
    shear_centre: Tuple[float, float] = (0.0, 0.0)
    tension_centre: Tuple[float, float] = (0.0, 0.0)
    mass: Dict[str, float] = field(default_factory=dict)
    source: str = "isotropic"
    section_id: Optional[int] = None

    def __post_init__(self) -> None:
        K = np.asarray(self.stiffness, dtype=np.float64)
        C = np.asarray(self.classical_stiffness, dtype=np.float64)
        if K.shape != (6, 6) or C.shape != (4, 4):
            raise ValueError("stiffness must be (6, 6) and classical_stiffness (4, 4)")
        if not (np.allclose(K, K.T, rtol=1e-10, atol=0.0) and np.allclose(C, C.T, rtol=1e-10, atol=0.0)):
            raise ValueError("section stiffness must be symmetric")
        object.__setattr__(self, "stiffness", K)
        object.__setattr__(self, "classical_stiffness", C)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_isotropic(
        cls,
        *,
        E: float,
        G: float,
        A: float,
        I_y: float,
        I_z: float,
        J_t: float,
        kappa: float = 5.0 / 6.0,
        Gamma: float = 0.0,
        rho: float = 0.0,
        I_x: Optional[float] = None,
        y_sc: float = 0.0,
        z_sc: float = 0.0,
    ) -> "SectionAnalysis":
        """Uncoupled section from classical inputs: ``diag(EA, EI_y, EI_z, kGA, kGA, GJ)``.

        ``y_sc``, ``z_sc`` are recorded as the shear centre only; the stiffness
        stays diagonal (each element keeps its own legacy shear-centre handling).
        """
        D = np.diag([E * A, E * I_y, E * I_z, kappa * G * A, kappa * G * A, G * J_t])
        C = np.diag([E * A, E * I_y, E * I_z, G * J_t])
        I_p = I_x if I_x is not None else I_y + I_z
        mass = {"m": rho * A, "y_m": 0.0, "z_m": 0.0, "I_yy": rho * I_y,
                "I_zz": rho * I_z, "I_yz": 0.0, "I_p": rho * I_p}
        return cls(D, C, E * Gamma, (float(y_sc), float(z_sc)), (0.0, 0.0), mass, "isotropic", None)

    @classmethod
    def from_thin_walled(cls, analysis, section_id: Optional[int] = None) -> "SectionAnalysis":
        """From a ``ThinWalledSectionAnalysis`` (right-handed results mapped to element signs)."""
        K = to_element_voigt(analysis.stiffness)
        C_rh = analysis.classical_stiffness
        S4 = np.diag([1.0, -1.0, 1.0, 1.0])
        C = S4 @ C_rh @ S4
        return cls(
            K,
            C,
            float(analysis.warping_constant),
            tuple(analysis.shear_centre),
            tuple(analysis.tension_centre),
            dict(analysis.mass_properties),
            "thin_walled",
            None if section_id is None else int(section_id),
        )

    # ------------------------------------------------------------------ #
    @property
    def is_coupled(self) -> bool:
        """True when any off-diagonal stiffness term is non-zero (relative to its diagonal)."""
        K = self.stiffness
        d = np.sqrt(np.abs(np.outer(np.diag(K), np.diag(K))))
        off = np.abs(K - np.diag(np.diag(K)))
        return bool(np.any(off > 1e-12 * np.where(d > 0, d, 1.0)))

    def euler_bernoulli_D(self) -> npt.NDArray[np.float64]:
        """6 x 6 ``D`` for shear-rigid elements: ``classical_stiffness`` on rows ``[0, 1, 2, 5]``, zero shear rows."""
        D = np.zeros((6, 6))
        D[np.ix_(CLASSICAL_INDEX, CLASSICAL_INDEX)] = self.classical_stiffness
        return D

    def mass_matrix(self) -> npt.NDArray[np.float64]:
        """Rigid-section mass per length (6, 6) for nodal fields ``[u_x, u_y, u_z, theta_x, theta_y, theta_z]``.

        Rotations are right-handed (as the nodal DOFs). A point at ``(y, z)`` moves by
        ``(u_x - y theta_z + z theta_y, u_y - z theta_x, u_z + y theta_x)``, so with
        ``S_y = m z_m`` and ``S_z = m y_m``::

            M[0,0] = M[1,1] = M[2,2] = m
            M[0,4] = S_y,  M[0,5] = -S_z,  M[1,3] = -S_y,  M[2,3] = S_z
            M[3,3] = I_p,  M[4,4] = I_yy,  M[5,5] = I_zz,  M[4,5] = -I_yz
        """
        mp = self.mass
        if not mp:
            raise ValueError("this SectionAnalysis has no mass properties")
        m = mp["m"]
        Sy, Sz = m * mp["z_m"], m * mp["y_m"]
        M = np.zeros((6, 6))
        M[0, 0] = M[1, 1] = M[2, 2] = m
        M[0, 4] = M[4, 0] = Sy
        M[0, 5] = M[5, 0] = -Sz
        M[1, 3] = M[3, 1] = -Sy
        M[2, 3] = M[3, 2] = Sz
        M[3, 3] = mp["I_p"]
        M[4, 4] = mp["I_yy"]
        M[5, 5] = mp["I_zz"]
        M[4, 5] = M[5, 4] = -mp["I_yz"]
        return M

    def stress_resultants(self, strain: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """``S = stiffness @ strain`` in element Voigt order."""
        return self.stiffness @ np.asarray(strain, dtype=np.float64)
