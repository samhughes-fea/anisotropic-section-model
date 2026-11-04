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
5. **Benscoter** (``benscoter=True``). The torsional warping ``omega`` is one more linear
   solve with the same wall-law matrices, and its rigid part is removed by a 3 x 3 solve in
   the walls' axial stiffness; both are differentiated. The 8 x 8 stiffness
   ``sum w G^T K G`` (``G`` the wall strains of the beam strains, ``omega_s d`` and
   ``omega psi'``) and the warping inertia ``sum w m omega^2`` then follow by the product rule.
6. **Second-order terms** (``geometric=True``). ``geometric_terms`` sums the wall
   resultants of the Saint-Venant wall strains against the section geometry, linear in
   the wall law and in the strain fields, so ``dH`` takes the derivative law and fields.
7. **GBT** (``gbt_section_derivative``). The GBT matrices integrate the cross-section modes
   against the GBT wall law (``N_ss`` released). The distortional modes are eigenvectors of
   ``B x = lambda C x`` and the shear-lag modes of ``E x = mu A x`` in the complement of the
   Saint-Venant warping; their derivatives follow from the eigenproblems, expanded on the
   other eigenvectors, which complete the space. Only the components that leave the span
   of the modes kept are formed: a GBT beam with one section does not see a change of basis
   within a block, its amplitudes being free.

Public API
----------
LayupParameter, laminate_derivative, wall_law_derivative, SectionDerivative,
section_derivative, gbt_section_derivative, element_section_derivative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from onedfem.pre_processing.section_integration.thin_walled.laminate import WallLaminate
from onedfem.pre_processing.section_integration.thin_walled.wall_stiffness import WALL_RESULTANTS, WALL_STRAINS  # noqa: F401

__all__ = ["LayupParameter", "laminate_derivative", "wall_law_derivative", "SectionDerivative", "section_derivative",
           "gbt_section_derivative", "element_section_derivative"]

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
    from onedfem.pre_processing.section_integration.thin_walled.laminate import strain_rotation

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
    #: right-handed ``d benscoter_stiffness`` (8, 8) and ``d warping_inertia``, with ``benscoter=True``
    benscoter_stiffness: Optional[npt.NDArray[np.float64]] = None
    warping_inertia: float = 0.0
    #: right-handed ``d geometric_terms`` (7, 6), with ``geometric=True``
    geometric_terms: Optional[npt.NDArray[np.float64]] = None
    #: ``d axial_points[:, 2]``: the axial wall stiffness weights (``with_pretwist`` is linear in them)
    axial_weights: Optional[npt.NDArray[np.float64]] = None
    #: ``d A`` of the parameter's laminate, ``A`` the axial wall stiffness of ``beam_wall_stiffness``
    #: (the taper coupling ``sum A_l parts[l]`` is linear in it)
    wall_axial: float = 0.0
    #: right-handed GBT stiffness, mass and geometric tensors with the modes held fixed
    #: (``gbt_section_derivative``), for GBT elements
    gbt_stiffness: Optional[npt.NDArray[np.float64]] = None
    gbt_mass: Optional[npt.NDArray[np.float64]] = None
    gbt_geometric: Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = None
    #: ``(dW0, dW1)``: the Saint-Venant warping fields (free DOFs, per unit section force)
    warping_fields: Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = None


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


def section_derivative(an, param: LayupParameter, benscoter: bool = False,
                       geometric: bool = False) -> SectionDerivative:
    """Exact derivatives of the section stiffness, flexibility and mass for ``param``.

    ``an`` is a ``ThinWalledSectionAnalysis``. A parameter of a laminate the section does not
    use gives zeros. ``benscoter=True`` adds the derivatives of the Benscoter stiffness and
    the warping inertia, ``geometric=True`` that of the second-order terms.
    """
    from onedfem.pre_processing.section_integration.thin_walled.section_analysis import _solve

    lam = an.laminates.get(param.laminate_id)
    used = lam is not None and any(w.laminate_id == param.laminate_id for w in an.section.walls)
    if not used:
        z6 = np.zeros((6, 6))
        return SectionDerivative(param, z6, z6.copy(), {k: 0.0 for k in ("m", "my", "mz", "yy", "zz", "yz")},
                                 z6.copy(), np.zeros((8, 8)) if benscoter else None, 0.0,
                                 np.zeros((7, 6)) if geometric else None, np.zeros(len(an._gauss_points)),
                                 _wall_axial_derivative(lam, param) if lam is not None else 0.0)
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
    dK8, dI = None, 0.0
    if benscoter:
        dK8, dI = _benscoter_derivative(an, param.laminate_id, dlaw, dmx, dm[0], (P0, P1, W0, W1),
                                        (dP0, dP1, dW0, dW1), dK)
    dH = None
    if geometric:
        from types import SimpleNamespace

        from onedfem.pre_processing.section_integration.thin_walled.section_analysis import T_RATE

        rows = type(an)._geometric_rows
        dH = np.zeros((7, 6))
        for g in an._gauss_points:
            Phi = _phi(g, P0, P1, W0, W1)
            dPhi = _phi(g, dP0, dP1, dW0, dW1)
            dH += rows(g, dPhi, dPhi @ T_RATE)
            if lid_of[g.wall_id] == param.laminate_id:
                gd = SimpleNamespace(K=dlaw, position=g.position, normal=g.normal, weight=g.weight)
                dH += rows(gd, Phi, Phi @ T_RATE)
    dwE = np.array([g.weight * dlaw[0, 0] if lid_of[g.wall_id] == param.laminate_id else 0.0
                    for g in an._gauss_points])
    return SectionDerivative(param, dF, dK, dmass, _mass_matrix(dmass), dK8, dI, dH, dwE,
                             _wall_axial_derivative(lam, param), warping_fields=(dW0, dW1))


def _wall_axial_derivative(lam: WallLaminate, param: LayupParameter) -> float:
    """``d A`` of ``beam_wall_stiffness(lam)`` (``N_ss = M_ss = 0``), the taper coupling's wall stiffness."""
    dABD, _ = laminate_derivative(lam, param)
    return float(wall_law_derivative(lam, dABD, "free")[0, 0])


def _benscoter_derivative(an, laminate_id, dlaw, dmx, dm0, sol, dsol, dKs):
    """``(d benscoter_stiffness, d warping_inertia)``, right-handed (module docstring, step 5)."""
    from onedfem.pre_processing.section_integration.thin_walled.section_analysis import _solve

    nd = an.n_dof
    m = an._matrices
    omega, c = an._torsion_warping
    # d omega: E omega = -R[:, 5] with the constant fixed; the constraint only sets the
    # constant, which the rigid-part removal below takes out again, so it is not differentiated
    M2 = np.zeros((nd, nd))
    for g in an._gauss_points:
        np.add.at(M2, np.ix_(g.dofs, g.dofs), g.weight * g.thickness * np.outer(g.Nx_local[0], g.Nx_local[0]))
    D1 = M2 @ an._rigid_modes[:, :1]
    S = np.zeros((nd + 1, nd + 1))
    S[:nd, :nd] = m["E"]
    S[:nd, nd:] = D1
    S[nd:, :nd] = D1.T
    x = np.concatenate([omega, _solve(S, np.concatenate([-m["R"][:, 5], [0.0]]))[nd:]])
    rhs = np.zeros(nd + 1)
    rhs[:nd] = -dmx["R"][:, 5] - dmx["E"] @ x[:nd]
    domega = _solve(S, rhs)[:nd]

    lid_of = {w.wall_id: w.laminate_id for w in an.section.walls}
    G3 = np.zeros((3, 3))
    dG3 = np.zeros((3, 3))
    db = np.zeros(3)
    for g in an._gauss_points:
        Kax = g.K[np.ix_([0, 2], [0, 2])]
        y, z = g.position
        Mg = np.array([[1.0, z, -y], [0.0, g.normal[1], -g.normal[0]]])
        rn = y * g.normal[1] - z * g.normal[0]
        ag = np.array([g.Nx_local[0] @ omega[g.dofs], -rn])
        dag = np.array([g.Nx_local[0] @ domega[g.dofs], 0.0])
        G3 += g.weight * Mg.T @ Kax @ Mg
        db += g.weight * Mg.T @ Kax @ dag
        if lid_of[g.wall_id] == laminate_id:
            dKax = dlaw[np.ix_([0, 2], [0, 2])]
            dG3 += g.weight * Mg.T @ dKax @ Mg
            db += g.weight * Mg.T @ dKax @ ag
    dc = np.linalg.solve(G3, db - dG3 @ c)

    P0, P1, W0, W1 = sol
    dP0, dP1, dW0, dW1 = dsol
    Ks = an._stiffness
    dK8 = np.zeros((8, 8))
    dI = 0.0
    for g in an._gauss_points:
        t_y, t_z = g.normal[1], -g.normal[0]
        y, z = g.position
        om = float(g.Nx_local[0] @ omega[g.dofs]) - (c[0] + c[1] * z - c[2] * y)
        om_s = float(g.B_local[1] @ omega[g.dofs]) - (c[1] * t_z - c[2] * t_y)
        dom = float(g.Nx_local[0] @ domega[g.dofs]) - (dc[0] + dc[1] * z - dc[2] * y)
        dom_s = float(g.B_local[1] @ domega[g.dofs]) - (dc[1] * t_z - dc[2] * t_y)
        Phi = _phi(g, P0, P1, W0, W1)
        Gm = np.zeros((4, 8))
        Gm[:, :6] = Phi @ Ks
        Gm[1, 6], Gm[0, 7] = om_s, om
        dGm = np.zeros((4, 8))
        dGm[:, :6] = _phi(g, dP0, dP1, dW0, dW1) @ Ks + Phi @ dKs
        dGm[1, 6], dGm[0, 7] = dom_s, dom
        KdG = g.K @ dGm
        dK8 += g.weight * (dGm.T @ g.K @ Gm + Gm.T @ KdG)
        lam = an.laminates[lid_of[g.wall_id]]
        dI += g.weight * 2.0 * lam.mass_per_area * om * dom
        if lid_of[g.wall_id] == laminate_id:
            dK8 += g.weight * Gm.T @ dlaw @ Gm
            dI += g.weight * dm0 * om * om
    dK8 = 0.5 * (dK8 + dK8.T)
    dK8[:6, :6] = dKs
    return dK8, float(dI)


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


def _gbt_law_derivative(lam: WallLaminate, dABD: np.ndarray) -> np.ndarray:
    """``d gbt_wall_stiffness(lam)`` (5, 5): ``N_ss`` released, ``P^T dABD P``."""
    rel = [WALL_RESULTANTS.index("N_ss")]
    kept = [i for i in range(6) if i not in rel]
    K = lam.ABD()
    P = np.zeros((6, 5))
    P[kept, np.arange(5)] = 1.0
    P[rel] = -np.linalg.solve(K[np.ix_(rel, rel)], K[np.ix_(rel, kept)])
    dK = P.T @ dABD @ P
    return 0.5 * (dK + dK.T)


def _mode_derivative(modes, lid: int, dlaw: np.ndarray, n_modes: int) -> np.ndarray:
    """``dX`` of the first ``n_modes`` distortional modes, up to a change of basis of their span.

    The modes are eigenvectors of ``B x = lambda C x`` on the admissible frame space, whose
    other eigenvectors (the remaining distortional modes) and rigid modes (``B r = 0``)
    complete it. With ``r_k = (dB - lambda_k dC) x_k``, the component of ``dx_k`` on another
    mode ``x_j`` is ``-x_j^T r_k / ((lambda_j - lambda_k) x_j^T C x_j)`` and on the rigid modes
    ``R`` it solves ``R^T C R c = -R^T dC x_k``. The components on the first ``n_modes`` modes
    only turn the basis of their span, which a GBT beam does not see (its amplitudes are free
    within the block), so they are left out.
    """
    lam_all, X_all = modes._modes
    nf = modes.n_frame_dof
    dC, dB = np.zeros((nf, nf)), np.zeros((nf, nf))
    for e, f, wt, l in modes._gauss():
        if l != lid:
            continue
        d = modes._element_dofs(e)
        E2 = np.vstack([f["u"], np.zeros(8), -f["w"], np.zeros(8), np.zeros(8)])
        E0 = np.vstack([np.zeros(8), np.zeros(8), np.zeros(8), -f["w_ss"], np.zeros(8)])
        dC[np.ix_(d, d)] += wt * E2.T @ dlaw @ E2
        dB[np.ix_(d, d)] += wt * E0.T @ dlaw @ E0
    C = modes._frame_matrices["C"]
    R = modes._rigid
    RCR = R.T @ C @ R
    rest = X_all[:, n_modes:]
    cj = np.einsum("ik,ij,jk->k", rest, C, rest)
    out = np.zeros((nf, n_modes))
    for k in range(n_modes):
        x = X_all[:, k]
        r = dB @ x - lam_all[k] * (dC @ x)
        c = -(rest.T @ r) / ((lam_all[n_modes:] - lam_all[k]) * cj)
        out[:, k] = rest @ c + R @ np.linalg.solve(RCR, -(R.T @ (dC @ x)))
    return out


def _warping_derivative(modes, lid: int, dlaw_r: np.ndarray, d: SectionDerivative, n_shear: int) -> np.ndarray:
    """``dOmega`` of the warping modes, up to a change of basis of the Saint-Venant span.

    The Saint-Venant warping modes span ``D = W - Q (Q^T A Q)^-1 Q^T A W`` (``W = [W0, W1]``,
    ``Q`` the rigid axial fields, ``A`` the axial warping stiffness), so ``Omega_sv = D T`` and
    ``dOmega_sv = dD T``. The shear-lag modes ``x_k`` solve ``E x = mu A x + A Q' nu`` with
    ``Q'^T A x = 0`` (``Q' = [Q, Omega_sv]``, ``nu = (Q'^T A Q')^-1 Q'^T E x``); their other
    shear-lag modes and ``Q'`` complete the warping space, and the components of ``dx_k`` on
    them follow from the derivative of these equations, as for the distortional modes.
    """
    an = modes.analysis
    A = modes._axial_warping_stiffness
    lid_of = {w.wall_id: w.laminate_id for w in an.section.walls}
    dA = np.zeros_like(A)
    for g in an._gauss_points:
        if lid_of[g.wall_id] == lid:
            np.add.at(dA, np.ix_(g.dofs, g.dofs), g.weight * dlaw_r[0, 0] * np.outer(g.Nx_local[0], g.Nx_local[0]))
    sol = an._solution
    W = np.hstack([sol["W0"], sol["W1"]])
    dW = np.hstack(d.warping_fields)
    U, sv, _ = np.linalg.svd(an._rigid_modes, full_matrices=False)
    Q = U[:, sv > 1e-12 * sv[0]]
    S = Q.T @ A @ Q
    G = np.linalg.solve(S, Q.T @ A)                       # (Q^T A Q)^-1 Q^T A
    dG = np.linalg.solve(S, Q.T @ dA) - np.linalg.solve(S, Q.T @ dA @ Q @ G)
    D = W - Q @ (G @ W)
    dD = dW - Q @ (G @ dW) - Q @ (dG @ W)
    Om = modes._check_warping(n_shear)
    n_sv = modes.n_sv_warping
    T, *_ = np.linalg.lstsq(D, Om[:, :n_sv], rcond=None)
    out = np.zeros_like(Om)
    out[:, :n_sv] = dD @ T
    if n_shear:
        full, _, mu = modes._warping_basis
        Xs, mus = full[:, n_sv:], mu[n_sv:]
        E = an._matrices["E"]
        dE = _assemble(an, {lid: dlaw_r})["E"]
        Qp = np.hstack([Q, Om[:, :n_sv]])
        dQp = np.hstack([np.zeros_like(Q), out[:, :n_sv]])
        QAQ = Qp.T @ A @ Qp
        rest = Xs[:, n_shear:]
        aj = np.einsum("ik,ij,jk->k", rest, A, rest)
        dAQ = dA @ Qp + A @ dQp
        for k in range(n_shear):
            x = Xs[:, k]
            nu = np.linalg.solve(QAQ, Qp.T @ (E @ x))
            r = dE @ x - mus[k] * (dA @ x) - dAQ @ nu
            c = -(rest.T @ r) / ((mus[n_shear:] - mus[k]) * aj)
            cq = np.linalg.solve(QAQ, -(dQp.T @ (A @ x) + Qp.T @ (dA @ x)))
            out[:, n_sv + k] = rest @ c + Qp @ cq
    return out


def gbt_section_derivative(modes, d: SectionDerivative, n_modes: int, n_shear: int = 0):
    """Derivatives of the GBT section matrices: ``(dK, dM, (dTX, dTY))``.

    ``modes`` is the ``CrossSectionModes`` of the section and ``d`` its ``section_derivative``
    (for the warping fields). The results are in the layout of ``modes.stiffness``,
    ``modes.mass`` and ``modes.geometric_tensor`` for ``n_modes`` and ``n_shear``
    (right-handed). Each is an integral of the mode fields against the GBT wall law
    (``N_ss`` released) or the wall mass, so ``dK = sum w (dH^T K H + H^T K dH + H^T dK H)``
    with ``dH`` the strains of the derivatives of the distortional modes (``_mode_derivative``)
    and of the warping modes (``_warping_derivative``).
    """
    param = d.parameter
    an = modes.analysis
    X, Om = modes._check(n_modes), modes._check_warping(n_shear)
    m, p = X.shape[1], Om.shape[1]
    nE, nM, nq = 6 + 2 * p + 3 * m, 6 + p + 2 * m, 3 + m
    dK, dM = np.zeros((nE, nE)), np.zeros((nM, nM))
    dTX, dTY = np.zeros((nq, nq, nE)), np.zeros((nq, nq, nE))
    lam = an.laminates.get(param.laminate_id)
    if lam is None or not any(w.laminate_id == param.laminate_id for w in an.section.walls):
        return dK, dM, (dTX, dTY)
    dABD, dm = laminate_derivative(lam, param)
    dlaw = _gbt_law_derivative(lam, dABD)
    dX = _mode_derivative(modes, param.laminate_id, dlaw, n_modes)
    dOm = _warping_derivative(modes, param.laminate_id, wall_law_derivative(lam, dABD, an.hoop), d, n_shear)
    lid_of = {w.wall_id: w.laminate_id for w in an.section.walls}
    iq = np.r_[1, 2, 3, 6 + p:6 + p + m]

    def frame(g, Xm, Om_, ex, xi):
        Pi, _ = modes._displacement_map(g, Xm, Om_)
        w_s = modes._fields(ex, xi)["w_s"] @ Xm[modes._element_dofs(ex)]
        return Pi, w_s

    for g in an._gauss_points:
        own = lid_of[g.wall_id] == param.laminate_id
        law = modes._laws[lid_of[g.wall_id]]
        H, ex = modes._strain_map(g, X, Om)
        dH, _ = modes._strain_map(g, dX, dOm)
        dH[:, :6] = 0.0                                   # the rigid fields do not change
        KH = law @ H
        dK += g.weight * (dH.T @ KH + KH.T @ dH)
        if own:
            dK += g.weight * H.T @ dlaw @ H
        xi = modes._locate(g.wall_id, g.position)[1]
        Pi, w_s = frame(g, X, Om, ex, xi)
        dPi, dw_s = frame(g, dX, dOm, ex, xi)
        dPi[:, :6] = 0.0
        rho = an.laminates[lid_of[g.wall_id]].mass_per_area
        dM += g.weight * rho * (dPi.T @ Pi + Pi.T @ dPi)
        if own:
            dM += g.weight * dm[0] * Pi.T @ Pi
        # geometric tensors: X = int N_xx Pq^T Pq, Y = int N_xs Pq^T Ps, (N_xx, N_xs) = law[:2] H
        Rs = law[:2] @ H
        dRs = law[:2] @ dH + (dlaw[:2] @ H if own else 0.0)
        Pq, dPq = Pi[1:, iq], dPi[1:, iq]
        t, nn = ex["t"], ex["n"]
        Ps = np.zeros((2, nq))
        Ps[:, 2] = [-t[1], t[0]]
        Ps[:, 3:] = np.outer(nn, w_s)
        dPs = np.zeros((2, nq))
        dPs[:, 3:] = np.outer(nn, dw_s)
        w = g.weight
        dTX += w * (np.einsum("ia,ib,e->abe", dPq, Pq, Rs[0]) + np.einsum("ia,ib,e->abe", Pq, dPq, Rs[0])
                    + np.einsum("ia,ib,e->abe", Pq, Pq, dRs[0]))
        dTY += w * (np.einsum("ia,ib,e->abe", dPq, Ps, Rs[1]) + np.einsum("ia,ib,e->abe", Pq, dPs, Rs[1])
                    + np.einsum("ia,ib,e->abe", Pq, Ps, dRs[1]))
    return 0.5 * (dK + dK.T), 0.5 * (dM + dM.T), (dTX, dTY)


def element_section_derivative(sa, d: SectionDerivative):
    """A ``SectionAnalysis`` like ``sa`` whose stiffness and mass matrix are the derivatives ``d``
    (element signs).

    Elements whose stiffness and mass are linear in the section's 6 x 6 stiffness and mass
    matrix (the Timoshenko family) form their own derivatives from it.
    """
    from dataclasses import fields

    from onedfem.pre_processing.section_integration.section_analysis import SectionAnalysis, to_element_voigt

    dM = d.mass_matrix.copy()

    class _Derivative(SectionAnalysis):
        # a closure, so copies made by dataclasses.replace (with_pretwist) keep it
        def mass_matrix(self):
            return dM.copy()

    kw = {f.name: getattr(sa, f.name) for f in fields(sa) if f.init}
    dK = np.array(d.stiffness, dtype=np.float64, copy=True)
    if sa.tapered:
        from onedfem.pre_processing.section_integration.thin_walled.taper import SectionTaper

        if sa.taper_parts is None:
            raise NotImplementedError("layup sensitivities of a tapered section need its taper split by laminate "
                                      "([Section_Taper_Laminate] in section_taper.txt)")
        row = sa.taper_parts.get(d.parameter.laminate_id)
        if row is not None:
            dK += d.wall_axial * SectionTaper.from_row(row).matrix()
    kw["stiffness"] = to_element_voigt(dK)
    kw["tapered"], kw["taper_parts"] = False, None
    if d.axial_weights is not None and sa.axial_points is not None:
        ap = np.array(sa.axial_points, dtype=np.float64, copy=True)
        ap[:, 2] = d.axial_weights
        kw["axial_points"] = ap
    if d.benscoter_stiffness is not None:
        S8 = np.diag([1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        dK8 = np.array(d.benscoter_stiffness, dtype=np.float64, copy=True)
        dK8[:6, :6] = dK
        kw["benscoter_stiffness"] = S8 @ dK8 @ S8
        kw["warping_inertia"] = d.warping_inertia
    else:
        kw["warping_inertia"] = 0.0
    if d.gbt_stiffness is not None:
        Sg = np.ones(d.gbt_stiffness.shape[0])
        Sg[1] = -1.0  # element sign of kappa_y
        Kg = Sg[:, None] * d.gbt_stiffness * Sg[None, :]
        if sa.tapered:
            Kg[:6, :6] += kw["stiffness"] - to_element_voigt(d.stiffness)
        Mg = np.array(d.gbt_mass, dtype=np.float64, copy=True)
        Mg[:6, :6] = dM
        kw["gbt_stiffness"], kw["gbt_mass"] = Kg, Mg
        kw["gbt_geometric"] = tuple(T * Sg for T in d.gbt_geometric)
    return _Derivative(**kw)
