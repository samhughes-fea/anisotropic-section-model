# anisotropic_section_model/ply_failure.py
"""Ply failure criteria: maximum stress, Tsai-Wu, Hashin and Puck, as exposures.

Every criterion gives an **exposure** ``f`` for a ply stress state: the factor by which
the stresses are past (``f > 1``) or short of (``f < 1``) the failure surface along a
proportional load path. The reserve factor is ``RF = 1 / f`` and the margin of safety
``MS = RF - 1``. For a criterion that is homogeneous of degree one in the stresses
(maximum stress, Hashin fibre compression, Puck) ``f`` is the criterion itself; for a
quadratic one (Tsai-Wu, the other Hashin modes) it is the root of ``a RF^2 + b RF = 1``,
so ``f`` always scales linearly with the load.

Stresses are in ply axes (1 fibre, 2 transverse in the ply plane, 3 normal), Voigt order
``[sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12]``, shape ``(..., 6)``.

- **Maximum stress:** each stress against its strength.
- **Tsai-Wu** (3D) with the interaction ``F_ij = F_star sqrt(F_ii F_jj)`` (default
  ``F_star = -1/2``).
- **Hashin** (1980, 3D): fibre tension / compression and matrix tension / compression.
- **Puck** (Puck and Schuermann 2002; VDI 2014): fibre failure ``sigma_11 / X``, and
  inter-fibre failure on the most stressed action plane, found by a search over the
  plane angle ``theta``. In plane stress it reduces to Puck's modes A, B and C.

The criteria themselves are laminate-constitutive-model's array exposures
(``ply_exposures`` and friends). ``PlyStrength`` keeps this package's defaults and maps
onto them: ``S13 = S12``, ``S23`` defaults to Puck's ``R_TT``, and ``F_star`` is the
Tsai-Wu interaction (``-1/2`` is LCM's ``"tsai-wu"``).

Public API
----------
PlyStrength, CRITERIA, max_stress, tsai_wu, hashin, puck, exposures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

__all__ = ["PlyStrength", "CRITERIA", "max_stress", "tsai_wu", "hashin", "puck", "exposures"]

#: criteria in output order
CRITERIA = ("max_stress", "tsai_wu", "hashin", "puck")


@dataclass(frozen=True)
class PlyStrength:
    """Ply strengths (all positive, compressive ones as magnitudes) and criterion parameters.

    Parameters
    ----------
    Xt, Xc : float
        Fibre-direction tensile and compressive strengths.
    Yt, Yc : float
        Transverse tensile and compressive strengths.
    S12 : float
        In-plane (and 1-3) shear strength.
    S23 : float, optional
        Transverse (2-3) shear strength. Default Puck's ``R_TT^A = Yc / (2 (1 + p_TT))``.
    Zt, Zc : float, optional
        Through-thickness strengths. Default ``Yt``, ``Yc`` (transversely isotropic ply).
    F_star : float
        Normalised Tsai-Wu interaction, ``-1/2`` by default.
    p_TL_t, p_TL_c, p_TT : float
        Puck inclination parameters ``p_perp_par^+``, ``p_perp_par^-`` and
        ``p_perp_perp^{+,-}``. Defaults 0.30, 0.25 and 0.25 (glass-epoxy values; carbon-epoxy
        is about 0.35, 0.30 and 0.25-0.30).
    """

    Xt: float
    Xc: float
    Yt: float
    Yc: float
    S12: float
    S23: Optional[float] = None
    Zt: Optional[float] = None
    Zc: Optional[float] = None
    F_star: float = -0.5
    p_TL_t: float = 0.30
    p_TL_c: float = 0.25
    p_TT: float = 0.25

    def __post_init__(self) -> None:
        for name in ("Xt", "Xc", "Yt", "Yc", "S12"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"PlyStrength.{name} must be positive, got {getattr(self, name)}")
        for name in ("S23", "Zt", "Zc"):
            v = getattr(self, name)
            if v is not None and not v > 0.0:
                raise ValueError(f"PlyStrength.{name} must be positive, got {v}")
        if not -1.0 < self.F_star < 1.0:
            raise ValueError(f"PlyStrength.F_star must lie in (-1, 1), got {self.F_star}")

    @classmethod
    def from_strength_properties(cls, strength, **parameters) -> "PlyStrength":
        """From laminate-constitutive-model's ``StrengthProperties3D``; ``parameters`` sets
        ``F_star`` and the Puck inclinations. Its ``S13`` is not used (this package takes
        ``S12`` for ``tau_13``)."""
        return cls(Xt=strength.Xt, Xc=strength.Xc, Yt=strength.Yt, Yc=strength.Yc, S12=strength.S12,
                   S23=strength.S23, Zt=strength.Zt, Zc=strength.Zc, **parameters)

    @property
    def R_TT(self) -> float:
        """Puck's fracture resistance of the action plane against transverse shear, ``R_perp_perp^A``."""
        return self.Yc / (2.0 * (1.0 + self.p_TT))

    @property
    def s23(self) -> float:
        return self.S23 if self.S23 is not None else self.R_TT

    @property
    def zt(self) -> float:
        return self.Zt if self.Zt is not None else self.Yt

    @property
    def zc(self) -> float:
        return self.Zc if self.Zc is not None else self.Yc


def _lcm(st: PlyStrength):
    """The same strengths and parameters in laminate-constitutive-model's terms."""
    from laminate_constitutive_model.laminate_materials_3d import StrengthProperties3D
    from laminate_constitutive_model.laminate_ply_exposure import PuckParameters

    strength = StrengthProperties3D(Xt=st.Xt, Xc=st.Xc, Yt=st.Yt, Yc=st.Yc, S12=st.S12,
                                    Zt=st.zt, Zc=st.zc, S13=st.S12, S23=st.s23)
    interaction = "tsai-wu" if st.F_star == -0.5 else st.F_star
    return strength, interaction, PuckParameters(st.p_TL_t, st.p_TL_c, st.p_TT)


def max_stress(s, st: PlyStrength) -> Tuple[np.ndarray, np.ndarray]:
    """Maximum-stress exposure and its governing component (``"s11t"``, ``"s22c"``, ``"t12"`` ...)."""
    from laminate_constitutive_model.laminate_ply_exposure import max_stress_exposure

    return max_stress_exposure(s, _lcm(st)[0])


def tsai_wu(s, st: PlyStrength) -> np.ndarray:
    """Tsai-Wu exposure (3D, ``F_ij = F_star sqrt(F_ii F_jj)`` for the normal-stress pairs)."""
    from laminate_constitutive_model.laminate_ply_exposure import tsai_wu_exposure

    strength, interaction, _ = _lcm(st)
    return tsai_wu_exposure(s, strength, interaction)


def hashin(s, st: PlyStrength) -> Tuple[np.ndarray, np.ndarray]:
    """Hashin (1980) exposure and mode: ``"FT"``, ``"FC"``, ``"MT"`` or ``"MC"``."""
    from laminate_constitutive_model.laminate_ply_exposure import hashin_exposure

    return hashin_exposure(s, _lcm(st)[0])


def puck(s, st: PlyStrength) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Puck exposure, mode (``"FFT"``, ``"FFC"``, ``"IFF-A"``, ``"IFF-B"``, ``"IFF-C"``) and
    the fracture-plane angle in degrees (for inter-fibre failure).

    The action plane is searched in 1-degree steps over [-90, 90) and refined by a
    parabola through the best step and its neighbours. Modes A, B and C are named as in
    plane stress: A for a tensile normal stress on the fracture plane, B for a compressive
    one on the plane at 0 degrees, C on an inclined plane.
    """
    from laminate_constitutive_model.laminate_ply_exposure import puck_exposure

    strength, _, params = _lcm(st)
    return puck_exposure(s, strength, params)


def exposures(s, st: PlyStrength) -> Dict[str, np.ndarray]:
    """Every criterion's exposure and mode: ``{"max_stress": f, "max_stress_mode": ..., ...,
    "puck_angle": degrees, "exposure": max over criteria, "margin": 1 / exposure - 1}``."""
    from laminate_constitutive_model.laminate_ply_exposure import ply_exposures

    strength, interaction, params = _lcm(st)
    return ply_exposures(s, strength, interaction, params)


def ply_stress_from_wall(sigma_11, sigma_22, tau_12, tau_xn, tau_sn, sigma_nn, angle_deg) -> npt.NDArray[np.float64]:
    """Ply-axis 3D stresses ``(..., 6)`` from a thin-walled recovery: in-plane ply stresses,
    and the transverse shears turned from wall to ply axes (``tau_13, tau_23``)."""
    from anisotropic_section_model.thin_walled.laminate import shear_rotation

    R = shear_rotation(np.radians(angle_deg))
    t13 = R[0, 0] * np.asarray(tau_xn) + R[0, 1] * np.asarray(tau_sn)
    t23 = R[1, 0] * np.asarray(tau_xn) + R[1, 1] * np.asarray(tau_sn)
    return np.stack(np.broadcast_arrays(sigma_11, sigma_22, sigma_nn, t23, t13, tau_12), axis=-1).astype(np.float64)
