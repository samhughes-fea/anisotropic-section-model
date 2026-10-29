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


def _rectangle_points(E: float, A: float, I_y: float, I_z: float) -> Optional[npt.NDArray[np.float64]]:
    """3 x 3 Gauss points ``[y, z, E dA]`` of the rectangle with the given ``A``, ``I_y``, ``I_z``."""
    if A <= 0 or I_y <= 0 or I_z <= 0:
        return None
    h, b = np.sqrt(12.0 * I_y / A), np.sqrt(12.0 * I_z / A)
    x, w = np.polynomial.legendre.leggauss(3)
    return np.array([[0.5 * b * yi, 0.5 * h * zi, E * 0.25 * b * h * wy * wz]
                     for yi, wy in zip(x, w) for zi, wz in zip(x, w)])


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
    benscoter_stiffness : ndarray (8, 8) or None
        Element-signed Benscoter stiffness for ``[eps_x, kappa_y, kappa_z, gamma_xy,
        gamma_xz, phi_x, d, psi']`` with ``d = psi - phi_x`` (thin-walled sections only).
    warping_inertia : float
        ``int rho_A omega^2 ds`` for the warping amplitude DOF.
    gbt_modes : int
        Cross-section distortional modes carried, ``M`` (0 when none were asked for).
    gbt_shear_modes : int
        Extra shear-lag warping modes carried, ``P``.
    gbt_sv_modes : int
        Warping modes spanning the Saint-Venant warping, ``n_sv`` (always carried by a GBT
        section; 0 without GBT matrices).
    third_order_stiffness : ndarray (8, 8) or None
        Element-signed third-order shear stiffness (Reddy, Levinson) for
        ``[eps_x, kappa_y, kappa_z, gamma_xy, gamma_xz, phi_x, chi_y, chi_z]``, where
        ``chi = gamma' - third_order_rate @ psi6`` stretches the walls through the slaved
        shear warping. Its leading 6 x 6 block is ``stiffness``.
    third_order_rate : ndarray (2, 6) or None
        Element-signed ``R``: the shear-strain rate ``gamma'`` of the Saint-Venant solution
        is ``R @ psi6`` (zero for an uncoupled section).
    geometric_terms : ndarray (7, 6) or None
        Second-order section terms ``[N, S_y, S_z, W, V_y, V_z, Q_r]`` (``GEOMETRIC_TERMS``)
        per unit element-signed section force: ``geometric_terms @ F``. The terms are
        integrals over the section in its own right-handed ``(y, z)`` and feed the
        geometric stiffness (``docs/element_library/second_order.md``).
    benscoter_geometric_terms : ndarray (7, 2) or None
        The same terms per unit Benscoter strain ``[d, psi']`` (the bimoment's part).
    third_order_geometric_terms : ndarray (7, 2) or None
        The same terms per unit third-order strain ``[chi_y, chi_z]``.
    axial_points : ndarray (m, 3) or None
        Section quadrature ``[y, z, k]`` with ``sum k f = int E f dA`` (fourth-order energy).
    gbt_stiffness : ndarray or None
        Element-signed GBT stiffness, square of size ``6 + 2(n_sv + P) + 3M``, for
        ``[psi6, b'(n_sv + P), b(n_sv + P), a''(M), a'(M), a(M)]``.
    gbt_mass : ndarray or None
        Section mass, square of size ``6 + (n_sv + P) + 2M``, for
        ``[u, v, w, theta_x, theta_y, theta_z, b(n_sv + P), a(M), a'(M)]``.
    """

    stiffness: npt.NDArray[np.float64]
    classical_stiffness: npt.NDArray[np.float64]
    warping_stiffness: float = 0.0
    shear_centre: Tuple[float, float] = (0.0, 0.0)
    tension_centre: Tuple[float, float] = (0.0, 0.0)
    mass: Dict[str, float] = field(default_factory=dict)
    source: str = "isotropic"
    section_id: Optional[int] = None
    benscoter_stiffness: Optional[npt.NDArray[np.float64]] = None
    warping_inertia: float = 0.0
    gbt_modes: int = 0
    gbt_stiffness: Optional[npt.NDArray[np.float64]] = None
    gbt_mass: Optional[npt.NDArray[np.float64]] = None
    gbt_shear_modes: int = 0
    gbt_sv_modes: int = 0
    third_order_stiffness: Optional[npt.NDArray[np.float64]] = None
    third_order_rate: Optional[npt.NDArray[np.float64]] = None
    geometric_terms: Optional[npt.NDArray[np.float64]] = None
    benscoter_geometric_terms: Optional[npt.NDArray[np.float64]] = None
    third_order_geometric_terms: Optional[npt.NDArray[np.float64]] = None
    axial_points: Optional[npt.NDArray[np.float64]] = None

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
        The third-order shear terms assume a solid rectangle (Reddy's cubic shear
        warping): ``EI_z / 84`` for ``chi_y`` and ``EI_y / 84`` for ``chi_z``. The
        second-order terms take the centroid at the origin and a doubly symmetric
        section (Wagner term ``W = N I_p / A``); the cubic shear warping adds none.
        """
        D = np.diag([E * A, E * I_y, E * I_z, kappa * G * A, kappa * G * A, G * J_t])
        C = np.diag([E * A, E * I_y, E * I_z, G * J_t])
        I_p = I_x if I_x is not None else I_y + I_z
        mass = {"m": rho * A, "y_m": 0.0, "z_m": 0.0, "I_yy": rho * I_y,
                "I_zz": rho * I_z, "I_yz": 0.0, "I_p": rho * I_p}
        D8 = np.zeros((8, 8))
        D8[:6, :6] = D
        D8[6, 6], D8[7, 7] = E * I_z / 84.0, E * I_y / 84.0
        H = np.zeros((7, 6))
        H[0, 0] = 1.0                  # N
        H[1, 2] = -1.0                 # S_y = int sigma y = -M_z
        H[2, 1] = 1.0                  # S_z = int sigma z = M_y
        H[3, 0] = I_p / A if A > 0 else 0.0
        H[4, 3] = H[5, 4] = 1.0        # V_y, V_z
        return cls(D, C, E * Gamma, (float(y_sc), float(z_sc)), (0.0, 0.0), mass, "isotropic", None,
                   third_order_stiffness=D8, third_order_rate=np.zeros((2, 6)),
                   geometric_terms=H @ ELEMENT_SIGN, third_order_geometric_terms=np.zeros((7, 2)),
                   axial_points=_rectangle_points(E, A, I_y, I_z))

    @classmethod
    def from_thin_walled(
        cls,
        analysis,
        section_id: Optional[int] = None,
        gbt_modes: int = 0,
        gbt_shear_modes: int = 0,
        gbt: bool = False,
    ) -> "SectionAnalysis":
        """From a ``ThinWalledSectionAnalysis`` (right-handed results mapped to element signs).

        ``gbt=True`` (implied by ``gbt_modes`` or ``gbt_shear_modes`` > 0) also builds the
        GBT matrices with that many distortional and extra shear-lag modes
        (``CrossSectionModes``; the analysis must use ``hoop="restrained"``).
        """
        K = to_element_voigt(analysis.stiffness)
        C_rh = analysis.classical_stiffness
        S4 = np.diag([1.0, -1.0, 1.0, 1.0])
        C = S4 @ C_rh @ S4
        try:
            S8 = np.diag([1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            KB = S8 @ analysis.benscoter_stiffness @ S8
            w_inertia = float(analysis.warping_inertia)
            K3 = S8 @ analysis.third_order_stiffness @ S8
            R3 = analysis.third_order_rate @ ELEMENT_SIGN
        except ValueError:
            KB, w_inertia, K3, R3 = None, 0.0, None, None
        H = analysis.geometric_terms @ ELEMENT_SIGN
        try:
            XB, X3 = analysis.benscoter_geometric_terms, analysis.third_order_geometric_terms
        except ValueError:
            XB = X3 = None
        K_gbt = M_gbt = None
        n_sv = 0
        if gbt or gbt_modes or gbt_shear_modes:
            from pre_processing.section_integration.thin_walled.cross_section_modes import CrossSectionModes

            modes = CrossSectionModes(analysis)
            n_sv = modes.n_sv_warping
            K_r = modes.stiffness(gbt_modes, gbt_shear_modes)
            Sg = np.ones(K_r.shape[0])
            Sg[1] = -1.0  # element sign of kappa_y
            K_gbt = Sg[:, None] * K_r * Sg[None, :]
            M_gbt = modes.mass(gbt_modes, gbt_shear_modes)
        out = cls(
            K,
            C,
            float(analysis.warping_constant),
            tuple(analysis.shear_centre),
            tuple(analysis.tension_centre),
            dict(analysis.mass_properties),
            "thin_walled",
            None if section_id is None else int(section_id),
            KB,
            w_inertia,
            int(gbt_modes),
            K_gbt,
            M_gbt,
            int(gbt_shear_modes),
            int(n_sv),
            K3,
            R3,
            H,
            XB,
            X3,
            analysis.axial_points,
        )
        if M_gbt is not None:
            # rigid block with the walls' own rotary inertia, as mass_matrix()
            M_gbt[:6, :6] = out.mass_matrix()
        return out

    # ------------------------------------------------------------------ #
    def gbt_indices(self, n_modes: int, n_shear: int = 0) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Rows of the stored GBT stiffness and mass (or wall load) for the first modes.

        The element carries the ``gbt_sv_modes`` Saint-Venant warping modes, the first
        ``n_shear`` shear-lag modes and the first ``n_modes`` distortional modes.
        """
        M, P = self.gbt_modes, self.gbt_shear_modes
        if self.gbt_stiffness is None or not (0 <= n_modes <= M and 0 <= n_shear <= P):
            raise ValueError(
                f"section carries {M} cross-section and {P} shear-lag modes"
                f"{'' if self.gbt_stiffness is not None else ' (no GBT matrices)'}; "
                f"asked for {n_modes} and {n_shear}"
            )
        Pt = self.gbt_sv_modes + P
        j = np.arange(self.gbt_sv_modes + n_shear)
        k = np.arange(n_modes)
        oa = 6 + 2 * Pt
        ik = np.r_[np.arange(6), 6 + j, 6 + Pt + j, oa + k, oa + M + k, oa + 2 * M + k]
        im = np.r_[np.arange(6), 6 + j, 6 + Pt + k, 6 + Pt + M + k]
        return ik, im

    def gbt_matrices(self, n_modes: int, n_shear: int = 0) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """GBT stiffness and mass of the first modes (see ``gbt_indices``)."""
        ik, im = self.gbt_indices(n_modes, n_shear)
        return self.gbt_stiffness[np.ix_(ik, ik)].copy(), self.gbt_mass[np.ix_(im, im)].copy()

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
