# pre_processing/section_integration/thin_walled/stress_recovery.py
"""Wall and ply stresses of a thin-walled section from its beam solution.

Steps at a point ``s`` of a wall:

1. **Kept wall strains.** A beam theory gives the wall strains it carries:
   ``[eps_xx, gamma_xs, kappa_xx, kappa_xs]`` (Saint-Venant, Benscoter) or also
   ``kappa_ss`` (GBT), and their rates along the beam ``d/dx``.
2. **Mid-surface strains.** The released resultants are zero: ``N_ss = 0``, and
   ``kappa_ss = 0`` (``hoop="restrained"``) or ``M_ss = 0`` (``hoop="free"``).
   Solving the released rows of the wall law ``[N; M] = ABD [eps; kappa]`` gives
   all six ``[eps_xx, eps_ss, gamma_xs, kappa_xx, kappa_ss, kappa_xs]``.
3. **Wall resultants** ``[N_xx, N_ss, N_xs, M_xx, M_ss, M_xs] = ABD e``.
4. **Ply stresses.** Through the thickness ``eps(n) = eps0 + n kappa``. Each ply
   has ``sigma = Q_bar eps`` in wall axes ``[sigma_xx, sigma_ss, tau_xs]`` and
   ``Q eps_ply`` in ply axes ``[sigma_11, sigma_22, tau_12]``.
5. **Interlaminar stresses** from 3D equilibrium integrated up from the bottom
   face (``n = -h/2``, traction free)::

       tau_xn(n) = -int (d sigma_xx/dx + d tau_xs/ds) dn
       tau_sn(n) = -int (d tau_xs/dx + d sigma_ss/ds) dn
       sigma_nn(n) = -int (d tau_xn/dx + d tau_sn/ds) dn

   The ``x`` rates come from the beam solution and the ``s`` rates from the wall
   strain field along the wall. On a straight wall these are exact for the
   in-plane stresses of each ply, which are linear in ``n``. With no surface
   load the values at the top face are the equilibrium residuals
   ``-(dN_xx/dx + dN_xs/ds)``, ``-(dN_xs/dx + dN_ss/ds)`` and
   ``-(dQ_x/dx + dQ_s/ds)``.

Public API
----------
FULL_WALL_STRAINS, mid_surface_strains, wall_resultants, through_thickness,
PlyPoint, WallPointStress, wall_point_stress, SaintVenantRecovery.

See Also
--------
docs/element_library/thin_walled_stress_recovery.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from numpy.polynomial import polynomial as P

from pre_processing.section_integration.thin_walled.laminate import WallLaminate, strain_rotation

#: mid-surface strains of a wall, order of ``WallLaminate.ABD``
FULL_WALL_STRAINS: Tuple[str, ...] = ("eps_xx", "eps_ss", "gamma_xs", "kappa_xx", "kappa_ss", "kappa_xs")
#: the five strains a beam theory can give, in their positions in ``FULL_WALL_STRAINS``
_KEPT = (0, 2, 3, 4, 5)

Array = npt.NDArray[np.float64]


def mid_surface_strains(
    laminate: WallLaminate, kept: npt.ArrayLike, hoop: str = "free"
) -> Array:
    """All six mid-surface strains from the strains a beam theory carries.

    Parameters
    ----------
    laminate : WallLaminate
    kept : array (4,) or (5,)
        ``[eps_xx, gamma_xs, kappa_xx, kappa_xs]`` (beam theories) or
        ``[eps_xx, gamma_xs, kappa_xx, kappa_ss, kappa_xs]`` (GBT, which carries
        the hoop curvature itself).
    hoop : {"restrained", "free"}
        With four kept strains: ``kappa_ss = 0`` or ``M_ss = 0``. Ignored with five.

    Returns
    -------
    ndarray (6,) in order ``FULL_WALL_STRAINS``; ``N_ss = 0`` always holds.
    """
    k = np.asarray(kept, dtype=np.float64)
    e = np.zeros(6)
    if k.shape == (5,):
        e[list(_KEPT)] = k
        unknown = [1]
    elif k.shape == (4,):
        e[[0, 2, 3, 5]] = k
        if hoop == "restrained":
            unknown = [1]
        elif hoop == "free":
            unknown = [1, 4]
        else:
            raise ValueError(f"hoop must be 'restrained' or 'free', got {hoop!r}")
    else:
        raise ValueError(f"kept strains must have 4 or 5 entries, got shape {k.shape}")
    K = laminate.ABD()
    known = [i for i in range(6) if i not in unknown]
    e[unknown] = -np.linalg.solve(K[np.ix_(unknown, unknown)], K[np.ix_(unknown, known)] @ e[known])
    return e


def wall_resultants(laminate: WallLaminate, strains6: npt.ArrayLike) -> Array:
    """``[N_xx, N_ss, N_xs, M_xx, M_ss, M_xs]`` for mid-surface strains (6,)."""
    return laminate.ABD() @ np.asarray(strains6, dtype=np.float64)


@dataclass(frozen=True)
class PlyPoint:
    """Stresses at one height ``n`` through a wall (``n = 0`` the mid-surface).

    Attributes
    ----------
    ply : int
        Ply index, from 0 at the bottom.
    n : float
    strain_wall, stress_wall : ndarray (3,)
        ``[eps_xx, eps_ss, gamma_xs]`` and ``[sigma_xx, sigma_ss, tau_xs]``.
    strain_ply, stress_ply : ndarray (3,)
        ``[eps_11, eps_22, gamma_12]`` and ``[sigma_11, sigma_22, tau_12]``.
    tau_xn, tau_sn, sigma_nn : float
        Interlaminar stresses in wall axes (zero when no rates were given).
    tau_13, tau_23 : float
        Transverse shear stresses in ply axes.
    """

    ply: int
    n: float
    strain_wall: Array
    stress_wall: Array
    strain_ply: Array
    stress_ply: Array
    tau_xn: float = 0.0
    tau_sn: float = 0.0
    sigma_nn: float = 0.0
    tau_13: float = 0.0
    tau_23: float = 0.0


def _stress_polys(laminate: WallLaminate, e6: Array) -> List[Array]:
    """Per ply: coefficients (3, 2) of ``sigma_wall(n) = c0 + c1 n``."""
    out = []
    for ply in laminate.plies:
        Qb = ply.rotated_stiffness()
        out.append(np.column_stack([Qb @ e6[:3], Qb @ e6[3:]]))
    return out


def _cumulative(z: Array, polys: List[Array]) -> List[Array]:
    """Integrals ``F(n) = int_{z_0}^{n} f`` of a piecewise polynomial, per ply (coefficients)."""
    out = []
    acc = 0.0
    for k, c in enumerate(polys):
        I = P.polyint(c, lbnd=z[k])
        I[0] += acc
        out.append(I)
        acc = P.polyval(z[k + 1], I)
    return out


def through_thickness(
    laminate: WallLaminate,
    strains6: npt.ArrayLike,
    rate_x: Optional[npt.ArrayLike] = None,
    rate_s: Optional[npt.ArrayLike] = None,
    rate_xs: Optional[npt.ArrayLike] = None,
    rate_ss: Optional[npt.ArrayLike] = None,
    points_per_ply: int = 3,
) -> List[PlyPoint]:
    """Ply stresses at ``points_per_ply`` heights per ply (bottom, ..., top of each ply).

    ``rate_x``, ``rate_s`` are ``d/dx`` and ``d/ds`` of the six mid-surface strains;
    ``rate_xs`` and ``rate_ss`` the mixed and second ``s`` rates (for ``sigma_nn``;
    ``d2/dx2`` is taken as zero). Without rates the interlaminar stresses are zero.
    """
    e = np.asarray(strains6, dtype=np.float64)
    z = laminate.interfaces
    zero = np.zeros(6)
    ex = zero if rate_x is None else np.asarray(rate_x, dtype=np.float64)
    es = zero if rate_s is None else np.asarray(rate_s, dtype=np.float64)
    exs = zero if rate_xs is None else np.asarray(rate_xs, dtype=np.float64)
    ess = zero if rate_ss is None else np.asarray(rate_ss, dtype=np.float64)

    sig = _stress_polys(laminate, e)
    sx, ss_, sxs, sss = (_stress_polys(laminate, r) for r in (ex, es, exs, ess))
    # f_x = d sigma_xx/dx + d tau_xs/ds ; f_s = d tau_xs/dx + d sigma_ss/ds
    fx = [sx[k][0] + ss_[k][2] for k in range(len(z) - 1)]
    fs = [sx[k][2] + ss_[k][1] for k in range(len(z) - 1)]
    txn = [-c for c in _cumulative(z, fx)]
    tsn = [-c for c in _cumulative(z, fs)]
    # d tau_xn/dx + d tau_sn/ds = -int (d2 tau_xs/dx ds) - int (d2 tau_xs/dx ds + d2 sigma_ss/ds2)
    g = [-(2.0 * sxs[k][2] + sss[k][1]) for k in range(len(z) - 1)]
    snn = [-c for c in _cumulative(z, _cumulative(z, g))]

    out: List[PlyPoint] = []
    for k, ply in enumerate(laminate.plies):
        T = strain_rotation(ply.angle_rad)
        Q = ply.material.reduced_stiffness()
        c, s = np.cos(ply.angle_rad), np.sin(ply.angle_rad)
        for n in np.linspace(z[k], z[k + 1], max(points_per_ply, 2)):
            eps = e[:3] + n * e[3:]
            eps_p = T @ eps
            t_xn = float(P.polyval(n, txn[k]))
            t_sn = float(P.polyval(n, tsn[k]))
            out.append(PlyPoint(
                ply=k, n=float(n), strain_wall=eps, stress_wall=P.polyval(n, sig[k].T),
                strain_ply=eps_p, stress_ply=Q @ eps_p,
                tau_xn=t_xn, tau_sn=t_sn, sigma_nn=float(P.polyval(n, snn[k])),
                tau_13=c * t_xn + s * t_sn, tau_23=-s * t_xn + c * t_sn,
            ))
    return out


@dataclass(frozen=True)
class WallPointStress:
    """Recovered state at one point of a wall.

    Attributes
    ----------
    wall_id : int
    s : float
        Distance along the wall from its start point.
    position : ndarray (2,)
        ``(y, z)`` of the point.
    strains : ndarray (6,)
        Mid-surface strains, order ``FULL_WALL_STRAINS``.
    resultants : ndarray (6,)
        ``[N_xx, N_ss, N_xs, M_xx, M_ss, M_xs]``.
    transverse_shear : ndarray (2,)
        Kirchhoff wall shear forces ``Q_x = dM_xx/dx + dM_xs/ds``, ``Q_s = dM_xs/dx + dM_ss/ds``.
    plies : list of PlyPoint
    """

    wall_id: int
    s: float
    position: Array
    strains: Array
    resultants: Array
    transverse_shear: Array
    plies: List[PlyPoint]


#: callable ``s -> (kept strains, their x rate)`` along one wall
KeptField = Callable[[float], Tuple[Array, Array]]


def _s_stencil(s: float, lo: float, hi: float) -> Array:
    """Three points for ``d/ds`` and ``d2/ds2`` at ``s`` inside ``[lo, hi]``."""
    h = 1e-3 * (hi - lo)
    if s - h < lo:
        return np.array([s, s + h, s + 2 * h])
    if s + h > hi:
        return np.array([s - 2 * h, s - h, s])
    return np.array([s - h, s, s + h])


def wall_point_stress(
    laminate: WallLaminate,
    field: KeptField,
    wall_id: int,
    s: float,
    position: npt.ArrayLike,
    element_span: Tuple[float, float],
    hoop: str = "free",
    points_per_ply: int = 3,
) -> WallPointStress:
    """Recover the state at ``s`` from a kept-strain field along the wall.

    ``element_span`` is the interval of the wall element holding ``s`` (the field
    is smooth inside it); the ``s`` rates are taken on a three-point stencil there.
    """
    pts = _s_stencil(s, *element_span)
    vals = [field(si) for si in pts]
    e = [mid_surface_strains(laminate, v[0], hoop) for v in vals]
    ex = [mid_surface_strains(laminate, v[1], hoop) for v in vals]
    # quadratic through the stencil: value, d/ds, d2/ds2 at s
    V = np.vander(pts - s, 3, increasing=True)
    Ci = np.linalg.inv(V)
    E, EX = np.array(e), np.array(ex)
    c_e, c_x = Ci @ E, Ci @ EX
    e0, es, ess = c_e[0], c_e[1], 2.0 * c_e[2]
    ex0, exs = c_x[0], c_x[1]
    K = laminate.ABD()
    R, Rx, Rs = K @ e0, K @ ex0, K @ es
    Q = np.array([Rx[3] + Rs[5], Rx[5] + Rs[4]])
    plies = through_thickness(laminate, e0, ex0, es, exs, ess, points_per_ply)
    return WallPointStress(int(wall_id), float(s), np.asarray(position, dtype=np.float64), e0, R, Q, plies)


class SaintVenantRecovery:
    """Wall and ply stresses of a ``ThinWalledSectionAnalysis`` under section forces.

    The section forces vary along the beam as with no distributed load
    (``M_y' = V_z``, ``M_z' = -V_y``), which sets the ``x`` rates.
    """

    def __init__(self, analysis) -> None:
        self.analysis = analysis

    def kept_field(self, wall_id: int, forces: npt.ArrayLike) -> KeptField:
        from pre_processing.section_integration.thin_walled.section_analysis import T_RATE

        f = np.asarray(forces, dtype=np.float64)
        fx = T_RATE @ f
        an = self.analysis
        return lambda s: (an.wall_strains_at(wall_id, s, f), an.wall_strains_at(wall_id, s, fx))

    def element_span(self, wall_id: int, s: float) -> Tuple[float, float]:
        sec = self.analysis.section
        L = sec.length(sec.wall(wall_id))
        le = L / self.analysis.n_sub
        k = min(max(int(s / le), 0), self.analysis.n_sub - 1)
        return k * le, (k + 1) * le

    def at(self, wall_id: int, s: float, forces: npt.ArrayLike, points_per_ply: int = 3) -> WallPointStress:
        """Recovered state at distance ``s`` along wall ``wall_id`` for section forces (6,)."""
        an = self.analysis
        w = an.section.wall(wall_id)
        lam = an.laminates[w.laminate_id]
        pos = an.point_at(wall_id, s).position
        return wall_point_stress(lam, self.kept_field(wall_id, forces), wall_id, s, pos,
                                 self.element_span(wall_id, s), an.hoop, points_per_ply)

    def along(self, wall_id: int, forces: npt.ArrayLike, n_points: int = 11,
              points_per_ply: int = 3) -> List[WallPointStress]:
        """``at`` at ``n_points`` evenly spaced points along a wall (both ends included)."""
        sec = self.analysis.section
        L = sec.length(sec.wall(wall_id))
        return [self.at(wall_id, s, forces, points_per_ply) for s in np.linspace(0.0, L, n_points)]
