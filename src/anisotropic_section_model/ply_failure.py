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
        if not -1.0 <= self.F_star <= 1.0:
            raise ValueError(f"PlyStrength.F_star must lie in [-1, 1], got {self.F_star}")

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


def _stress(s) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    if s.shape[-1] != 6:
        raise ValueError(f"ply stresses must have 6 components (..., 6), got shape {s.shape}")
    return s


def _quadratic_exposure(a, b) -> np.ndarray:
    """``f = 1 / RF`` with ``a RF^2 + b RF = 1`` (``a >= 0``); ``f = 0`` for a zero state."""
    a = np.maximum(np.asarray(a, dtype=np.float64), 0.0)
    b = np.asarray(b, dtype=np.float64)
    # 1 / RF = (b + sqrt(b^2 + 4 a)) / 2, the positive root of f^2 - b f - a = 0
    return 0.5 * (b + np.sqrt(b * b + 4.0 * a))


def max_stress(s, st: PlyStrength) -> Tuple[np.ndarray, np.ndarray]:
    """Maximum-stress exposure and its governing component (``"s11t"``, ``"s22c"``, ``"t12"`` ...)."""
    s = _stress(s)
    s11, s22, s33, t23, t13, t12 = np.moveaxis(s, -1, 0)
    terms = np.stack([
        np.where(s11 >= 0, s11 / st.Xt, -s11 / st.Xc),
        np.where(s22 >= 0, s22 / st.Yt, -s22 / st.Yc),
        np.where(s33 >= 0, s33 / st.zt, -s33 / st.zc),
        np.abs(t23) / st.s23,
        np.abs(t13) / st.S12,
        np.abs(t12) / st.S12,
    ])
    k = np.argmax(terms, axis=0)
    names = np.array([["s11t", "s11c"], ["s22t", "s22c"], ["s33t", "s33c"], ["t23", "t23"], ["t13", "t13"],
                      ["t12", "t12"]])
    comp = np.take_along_axis(np.moveaxis(s, -1, 0), k[None], 0)[0]
    mode = names[k, (comp < 0).astype(int)]
    return np.take_along_axis(terms, k[None], 0)[0], mode


def tsai_wu(s, st: PlyStrength) -> np.ndarray:
    """Tsai-Wu exposure (3D, ``F_ij = F_star sqrt(F_ii F_jj)`` for the normal-stress pairs)."""
    s = _stress(s)
    s11, s22, s33, t23, t13, t12 = np.moveaxis(s, -1, 0)
    F1, F2, F3 = 1 / st.Xt - 1 / st.Xc, 1 / st.Yt - 1 / st.Yc, 1 / st.zt - 1 / st.zc
    F11, F22, F33 = 1 / (st.Xt * st.Xc), 1 / (st.Yt * st.Yc), 1 / (st.zt * st.zc)
    F12 = st.F_star * np.sqrt(F11 * F22)
    F13 = st.F_star * np.sqrt(F11 * F33)
    F23 = st.F_star * np.sqrt(F22 * F33)
    a = (F11 * s11**2 + F22 * s22**2 + F33 * s33**2 + 2 * F12 * s11 * s22 + 2 * F13 * s11 * s33
         + 2 * F23 * s22 * s33 + (t23 / st.s23) ** 2 + (t13 / st.S12) ** 2 + (t12 / st.S12) ** 2)
    b = F1 * s11 + F2 * s22 + F3 * s33
    return _quadratic_exposure(a, b)


def hashin(s, st: PlyStrength) -> Tuple[np.ndarray, np.ndarray]:
    """Hashin (1980) exposure and mode: ``"FT"``, ``"FC"``, ``"MT"`` or ``"MC"``."""
    s = _stress(s)
    s11, s22, s33, t23, t13, t12 = np.moveaxis(s, -1, 0)
    S, S23 = st.S12, st.s23
    shear_l = (t12**2 + t13**2) / S**2
    fibre = np.where(s11 >= 0, np.sqrt((s11 / st.Xt) ** 2 + shear_l), -s11 / st.Xc)
    sn = s22 + s33
    shear_t = (t23**2 - s22 * s33) / S23**2
    mt = np.sqrt(np.maximum((sn / st.Yt) ** 2 + shear_t + shear_l, 0.0))
    a_mc = (sn / (2 * S23)) ** 2 + shear_t + shear_l
    b_mc = ((st.Yc / (2 * S23)) ** 2 - 1.0) * sn / st.Yc
    mc = _quadratic_exposure(a_mc, b_mc)
    matrix = np.where(sn >= 0, mt, mc)
    f = np.maximum(fibre, matrix)
    mode = np.where(fibre >= matrix, np.where(s11 >= 0, "FT", "FC"), np.where(sn >= 0, "MT", "MC"))
    return f, mode


_THETA = np.radians(np.arange(-90.0, 90.0, 1.0))


def _puck_iff(s22, s33, t23, t13, t12, st: PlyStrength, theta) -> np.ndarray:
    """Puck's inter-fibre exposure on the action plane at ``theta`` (broadcast)."""
    c, sn_ = np.cos(theta), np.sin(theta)
    sig_n = s22 * c * c + s33 * sn_ * sn_ + 2 * t23 * sn_ * c
    tau_nt = (s33 - s22) * sn_ * c + t23 * (c * c - sn_ * sn_)
    tau_n1 = t13 * sn_ + t12 * c
    RTp, RTT, RTL = st.Yt, st.R_TT, st.S12
    shear2 = tau_nt**2 + tau_n1**2
    with np.errstate(invalid="ignore", divide="ignore"):
        cos2 = np.where(shear2 > 0, tau_nt**2 / shear2, 0.0)
    sin2 = np.where(shear2 > 0, 1.0 - cos2, 1.0)
    pt = st.p_TT / RTT * cos2 + st.p_TL_t / RTL * sin2     # p_perp_psi^+ / R_perp_psi^A
    pc = st.p_TT / RTT * cos2 + st.p_TL_c / RTL * sin2     # p_perp_psi^- / R_perp_psi^A
    tens = np.sqrt(((1 / RTp - pt) * sig_n) ** 2 + (tau_nt / RTT) ** 2 + (tau_n1 / RTL) ** 2) + pt * sig_n
    comp = np.sqrt((tau_nt / RTT) ** 2 + (tau_n1 / RTL) ** 2 + (pc * sig_n) ** 2) + pc * sig_n
    return np.where(sig_n >= 0, tens, comp)


def puck(s, st: PlyStrength) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Puck exposure, mode (``"FFT"``, ``"FFC"``, ``"IFF-A"``, ``"IFF-B"``, ``"IFF-C"``) and
    the fracture-plane angle in degrees (for inter-fibre failure).

    The action plane is searched in 1-degree steps over [-90, 90) and refined by a
    parabola through the best step and its neighbours. Modes A, B and C are named as in
    plane stress: A for a tensile normal stress on the fracture plane, B for a compressive
    one on the plane at 0 degrees, C on an inclined plane.
    """
    s = _stress(s)
    s11, s22, s33, t23, t13, t12 = (x[..., None] for x in np.moveaxis(s, -1, 0))
    th = _THETA
    f_all = _puck_iff(s22, s33, t23, t13, t12, st, th)
    k = np.argmax(f_all, axis=-1)
    # parabolic refinement of the maximum
    n = th.size
    km, kp = (k - 1) % n, (k + 1) % n
    f0 = np.take_along_axis(f_all, k[..., None], -1)[..., 0]
    fm = np.take_along_axis(f_all, km[..., None], -1)[..., 0]
    fp = np.take_along_axis(f_all, kp[..., None], -1)[..., 0]
    den = fm - 2 * f0 + fp
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(den < 0, 0.5 * (fm - fp) / den, 0.0)
    d = np.clip(d, -0.5, 0.5)
    step = th[1] - th[0]
    th_fp = th[k] + d * step
    iff = _puck_iff(s22[..., 0], s33[..., 0], t23[..., 0], t13[..., 0], t12[..., 0], st, th_fp)
    iff = np.maximum(iff, f0)
    s11 = s11[..., 0]
    ff = np.where(s11 >= 0, s11 / st.Xt, -s11 / st.Xc)
    c, sn_ = np.cos(th_fp), np.sin(th_fp)
    sig_n = s22[..., 0] * c * c + s33[..., 0] * sn_ * sn_ + 2 * t23[..., 0] * sn_ * c
    iff_mode = np.where(sig_n >= 0, "IFF-A", np.where(np.abs(np.degrees(th_fp)) < 0.5, "IFF-B", "IFF-C"))
    mode = np.where(ff >= iff, np.where(s11 >= 0, "FFT", "FFC"), iff_mode)
    return np.maximum(ff, iff), mode, np.degrees(th_fp)


def exposures(s, st: PlyStrength) -> Dict[str, np.ndarray]:
    """Every criterion's exposure and mode: ``{"max_stress": f, "max_stress_mode": ..., ...,
    "puck_angle": degrees, "exposure": max over criteria, "margin": 1 / exposure - 1}``."""
    out: Dict[str, np.ndarray] = {}
    out["max_stress"], out["max_stress_mode"] = max_stress(s, st)
    out["tsai_wu"] = tsai_wu(s, st)
    out["hashin"], out["hashin_mode"] = hashin(s, st)
    out["puck"], out["puck_mode"], out["puck_angle"] = puck(s, st)
    f = np.max(np.stack([out[c] for c in CRITERIA]), axis=0)
    out["exposure"] = f
    with np.errstate(divide="ignore"):
        out["margin"] = np.where(f > 0, 1.0 / np.where(f > 0, f, 1.0) - 1.0, np.inf)
    return out


def ply_stress_from_wall(sigma_11, sigma_22, tau_12, tau_xn, tau_sn, sigma_nn, angle_deg) -> npt.NDArray[np.float64]:
    """Ply-axis 3D stresses ``(..., 6)`` from a thin-walled recovery: in-plane ply stresses,
    and the transverse shears turned from wall to ply axes (``tau_13, tau_23``)."""
    from anisotropic_section_model.thin_walled.laminate import shear_rotation

    R = shear_rotation(np.radians(angle_deg))
    t13 = R[0, 0] * np.asarray(tau_xn) + R[0, 1] * np.asarray(tau_sn)
    t23 = R[1, 0] * np.asarray(tau_xn) + R[1, 1] * np.asarray(tau_sn)
    return np.stack(np.broadcast_arrays(sigma_11, sigma_22, sigma_nn, t23, t13, tau_12), axis=-1).astype(np.float64)
