# pre_processing/section_integration/thin_walled/layup_sensitivity.py
"""Exact derivatives of thin-walled section properties with respect to ply angles and thicknesses.

A design parameter is one ply of one laminate (``LayupParameter``): its fibre angle (per
degree) or its thickness (per metre). Every step from the ply to the section is
differentiated exactly, with no finite differences:

1. **Laminate.** ``Q_bar(theta) = T^T Q T`` gives ``dQ_bar/dtheta`` from ``dT/dtheta``.
   A thickness change moves every interface ``n_j`` by ``[j > k] - 1/2`` (the stack stays
   centred on the wall's mid-surface), so the moments
   ``sum_i Q_bar_i (n_{i+1}^p - n_i^p) / p`` change by
   ``sum_i Q_bar_i (n_{i+1}^{p-1} dn_{i+1} - n_i^{p-1} dn_i)``. The same holds for the
   mass moments ``int rho n^p``.
2. **Wall law.** The condensed law ``K_c = P^T K P`` (``P = [I; -K_rr^-1 K_rk]``) has
   ``dK_c = P^T dK P``, exactly, because the released resultants are zero.
3. **Section.** The Saint-Venant solution (``ThinWalledSectionAnalysis._solution``) is two
   linear solves whose matrices are all linear in the wall laws. Differentiating them gives
   the derivatives of the warping fields, then
   ``dF = sum w (dPhi^T K Phi + Phi^T K dPhi + Phi^T dK Phi)`` and ``dS = -S dF S``.
   (The shortcut ``dF = -sum Phi^T dK Phi`` is not exact here: the rate-of-warping
   problem is not a minimum of the section energy.)
4. **Mass.** The section mass per length and its moments are linear in the wall mass
   moments, so ``d mass_matrix`` follows directly.

Public API
----------
LayupParameter, laminate_derivative, wall_law_derivative, SectionDerivative,
section_derivative, element_section_derivative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

from pre_processing.section_integration.thin_walled.laminate import WallLaminate
from pre_processing.section_integration.thin_walled.wall_stiffness import WALL_RESULTANTS, WALL_STRAINS

__all__ = ["LayupParameter", "laminate_derivative", "wall_law_derivative", "SectionDerivative", "section_derivative",
           "element_section_derivative"]

KINDS = ("angle", "thickness")


@dataclass(frozen=True)
class LayupParameter:
    """Ply ``ply`` (0-based, bottom first) of laminate ``laminate_id``: its ``angle`` (per degree)
    or ``thickness`` (per metre)."""

    laminate_id: int
    ply: int
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")

    @property
    def label(self) -> str:
        return f"L{self.laminate_id}.p{self.ply + 1}.{self.kind}"


def _dT(theta: float) -> np.ndarray:
    """``d T_eps / d theta`` (per radian) of ``laminate.strain_rotation``."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [-2 * c * s, 2 * c * s, c * c - s * s],
        [2 * c * s, -2 * c * s, -(c * c - s * s)],
        [-2 * (c * c - s * s), 2 * (c * c - s * s), -4 * c * s],
    ])


def _interface_derivative(lam: WallLaminate, ply: int) -> np.ndarray:
    """``d n_j / d t_ply`` for every interface ``j``."""
    n = len(lam.plies)
    return np.array([(1.0 if j > ply else 0.0) - 0.5 for j in range(n + 1)])


def laminate_derivative(lam: WallLaminate, param: LayupParameter) -> Tuple[np.ndarray, np.ndarray]:
    """``d ABD`` (6, 6) and ``d [m0, m1, m2]`` (the mass moments ``int rho n^p dn``, p = 0, 1, 2)."""
    from pre_processing.section_integration.thin_walled.laminate import strain_rotation

    k = int(param.ply)
    if not 0 <= k < len(lam.plies):
        raise ValueError(f"laminate {param.laminate_id} has {len(lam.plies)} plies, asked for ply {k + 1}")
    z = lam.interfaces
    dM = [np.zeros((3, 3)) for _ in range(3)]
    dm = np.zeros(3)
    if param.kind == "angle":
        p = lam.plies[k]
        T = strain_rotation(p.angle_rad)
        dTk = _dT(p.angle_rad) * np.pi / 180.0
        Q = p.material.reduced_stiffness()
        dQ = dTk.T @ Q @ T + T.T @ Q @ dTk
        for i, power in enumerate((1, 2, 3)):
            dM[i] = dQ * (z[k + 1] ** power - z[k] ** power) / power
    else:
        dz = _interface_derivative(lam, k)
        for i, power in enumerate((1, 2, 3)):
            for j, p in enumerate(lam.plies):
                dM[i] += p.rotated_stiffness() * (z[j + 1] ** (power - 1) * dz[j + 1] - z[j] ** (power - 1) * dz[j])
        for power in range(3):
            dm[power] = sum(p.material.rho * (z[j + 1] ** power * dz[j + 1] - z[j] ** power * dz[j])
                            for j, p in enumerate(lam.plies))
    dA, dB, dD = dM
    return np.block([[dA, dB], [dB, dD]]), dm


def wall_law_derivative(lam: WallLaminate, dABD: np.ndarray, hoop: str = "free") -> np.ndarray:
    """``d wall_law(lam, hoop)`` (4, 4) for a change ``dABD`` of the laminate stiffness."""
    released = ("N_ss",) if hoop == "restrained" else ("N_ss", "M_ss")
    rel = [WALL_RESULTANTS.index(r) for r in released]
    kept = [i for i in range(6) if i not in rel]
    K = lam.ABD()
    P = np.zeros((6, len(kept)))
    P[kept, np.arange(len(kept))] = 1.0
    P[rel] = -np.linalg.solve(K[np.ix_(rel, rel)], K[np.ix_(rel, kept)])
    dKc = P.T @ dABD @ P
    dKc = 0.5 * (dKc + dKc.T)
    strains = [WALL_STRAINS[i] for i in kept]
    keep = [strains.index(n) for n in WALL_STRAINS if n in strains]
    if hoop == "restrained":
        keep = [strains.index(n) for n in ("eps_xx", "eps_ss", "gamma_xs", "kappa_xx", "kappa_xs")
                if n in strains]
    return dKc[np.ix_(keep, keep)]


@dataclass(frozen=True)
class SectionDerivative:
    """Derivatives of a thin-walled section for one ``LayupParameter``.

    ``flexibility``, ``stiffness`` (6, 6, right-handed, about the origin as
    ``ThinWalledSectionAnalysis``), ``mass`` (the derivative of each ``mass_properties``
    integral: ``m``, ``my``, ``mz``, ``yy``, ``zz``, ``yz``) and ``mass_matrix`` (6, 6, as
    ``SectionAnalysis.mass_matrix``).
    """

    parameter: LayupParameter
    flexibility: npt.NDArray[np.float64]
    stiffness: npt.NDArray[np.float64]
    mass: Dict[str, float]
    mass_matrix: npt.NDArray[np.float64]


def _assemble(an, laws: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
    """The section matrices of ``an._matrices`` with the wall laws replaced by ``laws``
    (every matrix is linear in them; the warping constraint ``D`` is left out)."""
    nd = an.n_dof
    out = {k: np.zeros(s) for k, s in (("E", (nd, nd)), ("C", (nd, nd)), ("R", (nd, 6)), ("L", (nd, 6)),
                                        ("Ly", (nd, 6)), ("A", (6, 6)), ("Q", (6, 6)))}
    lid_of = {w.wall_id: w.laminate_id for w in an.section.walls}
    for g in an._gauss_points:
        K = laws.get(lid_of[g.wall_id])
        if K is None:
            continue
        d, wt = g.dofs, g.weight
        KB, KZ, KY = K @ g.B_local, K @ g.Z, K @ g.Y
        ix = np.ix_(d, d)
        np.add.at(out["E"], ix, wt * g.B_local.T @ KB)
        np.add.at(out["C"], ix, wt * KB.T @ g.Nx_local)
        np.add.at(out["R"], d, wt * g.B_local.T @ KZ)
        np.add.at(out["L"], d, wt * g.Nx_local.T @ KZ)
        np.add.at(out["Ly"], d, wt * g.B_local.T @ KY)
        out["A"] += wt * g.Z.T @ KZ
        out["Q"] += wt * g.Z.T @ KY
    return out


def _phi(g, P0, P1, W0, W1) -> np.ndarray:
    return g.Z @ P0 + g.Y @ P1 + g.B_local @ W0[g.dofs] + g.Nx_local @ W1[g.dofs]


def section_derivative(an, param: LayupParameter) -> SectionDerivative:
    """Exact derivatives of the section stiffness, flexibility and mass for ``param``.

    ``an`` is a ``ThinWalledSectionAnalysis``. A parameter of a laminate the section does not
    use gives zeros.
    """
    from pre_processing.section_integration.thin_walled.section_analysis import _solve

    lam = an.laminates.get(param.laminate_id)
    used = lam is not None and any(w.laminate_id == param.laminate_id for w in an.section.walls)
    if not used:
        z6 = np.zeros((6, 6))
        return SectionDerivative(param, z6, z6.copy(), {k: 0.0 for k in ("m", "my", "mz", "yy", "zz", "yz")},
                                 z6.copy())
    dABD, dm = laminate_derivative(lam, param)
    dlaw = wall_law_derivative(lam, dABD, an.hoop)

    m = an._matrices
    dmx = _assemble(an, {param.laminate_id: dlaw})
    S = an._system
    nd = an.n_dof
    dS = np.zeros_like(S)
    dS[:nd, :nd] = dmx["E"]
    dS[:nd, nd:nd + 6] = dmx["R"]
    dS[nd:nd + 6, :nd] = dmx["R"].T
    dS[nd:nd + 6, nd:nd + 6] = dmx["A"]
    sol = an._solution
    W1, P1, W0, P0 = sol["W1"], sol["P1"], sol["W0"], sol["P0"]
    # dS is zero on the warping-constraint rows and columns, so the constraint multipliers
    # of the solution do not enter dS @ X
    n_tot = S.shape[0]
    X1 = np.zeros((n_tot, 6))
    X1[:nd], X1[nd:nd + 6] = W1, P1
    dX1 = _solve(S, -dS @ X1)
    dW1, dP1 = dX1[:nd], dX1[nd:nd + 6]

    C, Lm, Ly, Q = m["C"], m["L"], m["Ly"], m["Q"]
    dC, dL, dLy, dQ = dmx["C"], dmx["L"], dmx["Ly"], dmx["Q"]
    X0 = np.zeros((n_tot, 6))
    X0[:nd], X0[nd:nd + 6] = W0, P0
    drhs = np.zeros((n_tot, 6))
    drhs[:nd] = -(dC - dC.T) @ W1 - (C - C.T) @ dW1 + (dL - dLy) @ P1 + (Lm - Ly) @ dP1
    drhs[nd:nd + 6] = -(dL - dLy).T @ W1 - (Lm - Ly).T @ dW1 - (dQ - dQ.T) @ P1 - (Q - Q.T) @ dP1
    dX0 = _solve(S, drhs - dS @ X0)
    dW0, dP0 = dX0[:nd], dX0[nd:nd + 6]

    lid_of = {w.wall_id: w.laminate_id for w in an.section.walls}
    dF = np.zeros((6, 6))
    for g in an._gauss_points:
        Phi = _phi(g, P0, P1, W0, W1)
        dPhi = _phi(g, dP0, dP1, dW0, dW1)
        KdPhi = g.K @ dPhi
        dF += g.weight * (dPhi.T @ g.K @ Phi + Phi.T @ KdPhi)
        if lid_of[g.wall_id] == param.laminate_id:
            dF += g.weight * Phi.T @ dlaw @ Phi
    dF = 0.5 * (dF + dF.T)
    Ks = an.stiffness
    dK = -Ks @ dF @ Ks
    dK = 0.5 * (dK + dK.T)
    dmass = _mass_derivative(an, param.laminate_id, dm)
    return SectionDerivative(param, dF, dK, dmass, _mass_matrix(dmass))


def _mass_derivative(an, laminate_id: int, dm: np.ndarray) -> Dict[str, float]:
    """Derivatives of the ``mass_properties`` integrals for wall mass moments changed by ``dm``."""
    sec = an.section
    tot = dict(m=0.0, my=0.0, mz=0.0, yy=0.0, zz=0.0, yz=0.0)
    m0, m1, m2 = dm
    for w in sec.walls:
        if w.laminate_id != laminate_id:
            continue
        a, b = sec.endpoints(w)
        L = sec.length(w)
        ny, nz = sec.normal(w)
        iy, iz = L * (a[0] + b[0]) / 2, L * (a[1] + b[1]) / 2
        iyy = L * (a[0] ** 2 + a[0] * b[0] + b[0] ** 2) / 3
        izz = L * (a[1] ** 2 + a[1] * b[1] + b[1] ** 2) / 3
        iyz = L * (2 * a[0] * a[1] + a[0] * b[1] + b[0] * a[1] + 2 * b[0] * b[1]) / 6
        tot["m"] += m0 * L
        tot["my"] += m0 * iy + m1 * ny * L
        tot["mz"] += m0 * iz + m1 * nz * L
        tot["zz"] += m0 * iyy + 2 * m1 * ny * iy + m2 * ny * ny * L
        tot["yy"] += m0 * izz + 2 * m1 * nz * iz + m2 * nz * nz * L
        tot["yz"] += m0 * iyz + m1 * (ny * iz + nz * iy) + m2 * ny * nz * L
    return tot


def _mass_matrix(t: Dict[str, float]) -> np.ndarray:
    """``SectionAnalysis.mass_matrix`` from the mass integrals (it is linear in them)."""
    M = np.zeros((6, 6))
    M[0, 0] = M[1, 1] = M[2, 2] = t["m"]
    Sy, Sz = t["mz"], t["my"]
    M[0, 4] = M[4, 0] = Sy
    M[0, 5] = M[5, 0] = -Sz
    M[1, 3] = M[3, 1] = -Sy
    M[2, 3] = M[3, 2] = Sz
    M[3, 3] = t["yy"] + t["zz"]
    M[4, 4] = t["yy"]
    M[5, 5] = t["zz"]
    M[4, 5] = M[5, 4] = -t["yz"]
    return M


def element_section_derivative(sa, d: SectionDerivative):
    """A ``SectionAnalysis`` like ``sa`` whose stiffness and mass matrix are the derivatives ``d``
    (element signs).

    Elements whose stiffness and mass are linear in the section's 6 x 6 stiffness and mass
    matrix (the Timoshenko family) form their own derivatives from it.
    """
    from dataclasses import fields

    from pre_processing.section_integration.section_analysis import SectionAnalysis, to_element_voigt

    class _Derivative(SectionAnalysis):
        def mass_matrix(self):
            return self._dM.copy()

    kw = {f.name: getattr(sa, f.name) for f in fields(sa) if f.init}
    kw["stiffness"] = to_element_voigt(d.stiffness)
    out = _Derivative(**kw)
    object.__setattr__(out, "_dM", d.mass_matrix.copy())
    return out
