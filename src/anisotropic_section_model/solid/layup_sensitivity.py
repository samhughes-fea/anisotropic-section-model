# anisotropic_section_model/solid/layup_sensitivity.py
"""Exact derivatives of a 2D-meshed section's properties with respect to ply angles and thicknesses.

A parameter is one ply of one laminate (``thin_walled.layup_sensitivity.LayupParameter``):
the elements the mesh tags with that ply (``SectionMesh.plies``) turn with its angle, and
the nodes move with its thickness as ``SectionMesh.velocities`` says (each stack stays
centred on its wall's mid-line, as for thin-walled sections, so the two derivatives agree).

Every step is differentiated exactly, with no finite differences:

1. **Material** (angle). ``C = T C_m T^T`` with ``T`` the bond matrix of the material axes
   ``R``; ``T`` is bilinear in ``R``, so ``dT = T(dR, R) + T(R, dR)`` with ``dR`` the
   derivative of the axes, and ``dC = dT C_m T^T + T C_m dT^T``.
2. **Geometry** (thickness). With the node velocity ``V`` each quadrature point moves by
   ``N V``, its Jacobian ``J = dN_nat X`` by ``dJ = dN_nat V``, so the weight ``w det J``
   changes by ``w tr(J^-1 dJ)`` and the shape-function gradients ``G = J^-1 dN_nat`` by
   ``-J^-1 dJ G``. ``B`` is linear in ``G`` and ``Z`` affine in the point; ``N_x`` does not
   change. Tied nodes (``SectionMesh.ties``) sit at fixed natural coordinates of their
   master element only while the master moves with them: their natural coordinates change
   by ``dxi = J^-T (v_node - N V_master)`` (``SectionMesh.tie_points``), and the tie
   weights ``N(xi)`` with them.
3. **Section.** ``E``, ``Cx``, ``R``, ``L`` and ``A`` are sums over the points of products
   of ``w``, ``B``, ``Z``, ``N_x`` and ``C``, reduced by the tie matrix ``T``, so the product
   rule gives their derivatives. The two Saint-Venant solves
   (``SolidSectionAnalysis._solution``) give ``dX = S^-1 (d rhs - dS X)``; the warping
   constraint ``D`` only fixes the rigid-motion gauge of the warping, which the flexibility
   does not depend on, so it is held. Then
   ``dF = sum dw Phi^T C Phi + w (dPhi^T C Phi + Phi^T C dPhi + Phi^T dC Phi)`` and
   ``dK = -K dF K``.
4. **Mass.** ``int rho``, ``int rho y`` ... are sums over the points: the product rule in
   ``w``, ``y`` and ``z`` (zero for an angle).
5. **Benscoter** (``benscoter=True``). The torsional warping solve, the 3 x 3 removal of its
   rigid part in the axial modulus ``E_x = C_xx - C_xi C_ii^-1 C_ix`` and the relaxed
   gradient solve are differentiated; then ``K8 = sum w G^T C G`` and
   ``int rho omega^2`` by the product rule.
6. **Second-order terms** (``geometric=True``) and the axial quadrature ``[y, z, w E_x]``
   are products of the stresses, ``w``, ``y``, ``z`` and ``E_x``.

Public API
----------
material_derivative, solid_section_derivative.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from anisotropic_section_model.solid.elements import quadrature, shape
from anisotropic_section_model.solid.material import _PAIRS, bond_matrix, material_axes
from anisotropic_section_model.solid.section_analysis import _operators
from anisotropic_section_model.thin_walled.layup_sensitivity import LayupParameter, SectionDerivative

__all__ = ["material_derivative", "solid_section_derivative"]

_MASS_KEYS = ("m", "my", "mz", "yy", "zz", "yz")


def _bond_pair(R: np.ndarray, dR: np.ndarray) -> np.ndarray:
    """``T(R, dR) + T(dR, R)``: the derivative of ``bond_matrix`` along ``dR``."""
    T = np.zeros((6, 6))
    for I, (i, j) in enumerate(_PAIRS):
        for J, (k, m) in enumerate(_PAIRS):
            v = R[i, k] * dR[j, m] + dR[i, k] * R[j, m]
            if k != m:
                v += R[i, m] * dR[j, k] + dR[i, m] * R[j, k]
            T[I, J] = v
    return T


def material_derivative(material, theta_deg: float, alpha_deg: float) -> npt.NDArray[np.float64]:
    """``d C / d theta`` (6, 6, per degree) of ``SolidMaterial.stiffness_in_section``."""
    th, al = np.deg2rad(theta_deg), np.deg2rad(alpha_deg)
    x = np.array([1.0, 0.0, 0.0])
    s = np.array([0.0, np.cos(al), np.sin(al)])
    de1 = -np.sin(th) * x + np.cos(th) * s
    de2 = -np.cos(th) * x - np.sin(th) * s
    dR = np.deg2rad(1.0) * np.column_stack([de1, de2, np.zeros(3)])
    R = material_axes(theta_deg, alpha_deg)
    T, dT = bond_matrix(R), _bond_pair(R, dR)
    Cm = material.stiffness()
    dC = dT @ Cm @ T.T + T @ Cm @ dT.T
    return 0.5 * (dC + dC.T)


# ---------------------------------------------------------------------- the design change
class _Rate:
    """The derivative of one quadrature point's data: ``dw``, ``dy``, ``dz``, ``dZ``, ``dB``
    (``None`` when the point does not move) and ``dC`` (``None`` when its stiffness does not
    change)."""

    __slots__ = ("dw", "dy", "dz", "dZ", "dB", "dC")

    def __init__(self, dw=0.0, dy=0.0, dz=0.0, dZ=None, dB=None, dC=None):
        self.dw, self.dy, self.dz, self.dZ, self.dB, self.dC = dw, dy, dz, dZ, dB, dC

    @property
    def moves(self) -> bool:
        return self.dB is not None


_STILL = _Rate()


def _point_rates(an, dC: Dict[int, np.ndarray], V: Optional[np.ndarray]) -> List[_Rate]:
    """``_Rate`` of every point of ``an._points`` (step 2 of the module docstring)."""
    mesh = an.mesh
    if V is None:
        return [_Rate(dC=dC[p.element]) if p.element in dC else _STILL for p in an._points]
    out = []
    points = iter(an._points)
    for e, (conn, kind) in enumerate(zip(mesh.elements, mesh.kinds)):
        X, Ve = mesh.nodes[conn], V[conn]
        moving = bool(np.any(Ve))
        for xi, w in zip(*quadrature(kind)):
            p = next(points)
            if p.element != e:
                raise RuntimeError("the quadrature points are not in element order")
            if not moving:
                out.append(_Rate(dC=dC.get(e)))
                continue
            N, dNn = shape(kind, *xi)
            J, dJ = dNn @ X, dNn @ Ve
            JidJ = np.linalg.solve(J, dJ)
            G = np.linalg.solve(J, dNn)
            dG = -JidJ @ G
            dy, dz = N @ Ve
            Z0, dB, _ = _operators(0.0, 0.0, N, dG)
            dZ = _operators(dy, dz, N, dG)[0] - Z0
            out.append(_Rate(p.weight * float(np.trace(JidJ)), float(dy), float(dz), dZ, dB, dC.get(e)))
    return out


def _tie_rate(an, V: np.ndarray) -> Optional[sp.csr_matrix]:
    """``d T`` (n_full, n_dof) of the tie matrix for the node velocity ``V`` (step 2)."""
    mesh = an.mesh
    if not mesh.ties:
        return None
    if mesh.tie_points is None:
        raise ValueError(f"section {mesh.section_id}: the mesh has ties but no SectionMesh.tie_points; "
                         "thickness sensitivities need how each node was tied (SectionMesh.from_thin_walled)")
    rows: Dict[int, Dict[int, float]] = {}       # the ties as they were built (unfiltered)
    drows: Dict[int, Dict[int, float]] = {}
    for n, e, xi, eta in mesh.tie_points:
        conn = mesh.elements[e]
        X, Ve = mesh.nodes[conn], V[conn]
        N, dNn = shape(mesh.kinds[e], xi, eta)
        Jt = (dNn @ X).T                          # columns d(y, z)/dxi, d(y, z)/deta
        free = [k for k, c in enumerate((xi, eta)) if abs(abs(c) - 1.0) > 1e-12]
        dxi = np.zeros(2)
        # a coordinate on the element's edge stays there: the strips' edges move with the
        # nodes on them (a point clamped onto an edge from outside keeps its clamp)
        inside = np.linalg.norm(N @ X - mesh.nodes[n]) < 1e-9 * max(float(np.ptp(X, axis=0).max()), 1e-300)
        if inside:
            dxi = np.linalg.solve(Jt, V[n] - N @ Ve)
        elif free:
            dxi[free] = np.linalg.lstsq(Jt[:, free], V[n] - N @ Ve, rcond=None)[0]
        dN = dNn.T @ dxi
        row: Dict[int, float] = {}
        drow: Dict[int, float] = {}
        for m, c, dc in zip(conn, N, dN):
            m = int(m)
            for mm, cc in rows.get(m, {m: 1.0}).items():
                row[mm] = row.get(mm, 0.0) + c * cc
                drow[mm] = drow.get(mm, 0.0) + dc * cc + c * drows.get(m, {}).get(mm, 0.0)
        rows[int(n)], drows[int(n)] = row, drow
    r, c, v = [], [], []
    for n, drow in drows.items():
        for m, dc in drow.items():
            if m not in an._free_index:
                if abs(dc) > 1e-14:
                    raise RuntimeError(f"tie of node {n} onto node {m}, which is not free")
                continue
            for k in range(3):
                r.append(3 * n + k)
                c.append(3 * an._free_index[m] + k)
                v.append(dc)
    return sp.csr_matrix((v, (r, c)), shape=(an.n_full, an.n_dof))


def _assemble(an, rates: List[_Rate]) -> Dict[str, object]:
    """Derivatives of ``E``, ``Cx``, ``R``, ``L``, ``A`` of ``an._matrices`` before the tie
    reduction (``full``) for the point rates ``rates``."""
    nf = an.n_full
    rr, cc, ev, cv = [], [], [], []
    R = np.zeros((nf, 6))
    L = np.zeros((nf, 6))
    A = np.zeros((6, 6))
    for p, q in zip(an._points, rates):
        if q is _STILL or (q.dC is None and not q.moves):
            continue
        d, w = p.dofs, p.weight
        C, B, Z, Nx = p.C, p.B, p.Z, p.Nx
        CB, CZ = C @ B, C @ Z
        dE = q.dw * B.T @ CB
        dCx = q.dw * CB.T @ Nx
        dR = q.dw * B.T @ CZ
        dL = q.dw * Nx.T @ CZ
        dA = q.dw * Z.T @ CZ
        if q.moves:
            CdB, CdZ = C @ q.dB, C @ q.dZ
            dE += w * (q.dB.T @ CB + CB.T @ q.dB)
            dCx += w * CdB.T @ Nx
            dR += w * (q.dB.T @ CZ + B.T @ CdZ)
            dL += w * Nx.T @ CdZ
            dA += w * (q.dZ.T @ CZ + CZ.T @ q.dZ)
        if q.dC is not None:
            dCB, dCZ = q.dC @ B, q.dC @ Z
            dE += w * B.T @ dCB
            dCx += w * dCB.T @ Nx
            dR += w * B.T @ dCZ
            dL += w * Nx.T @ dCZ
            dA += w * Z.T @ dCZ
        r, c = np.meshgrid(d, d, indexing="ij")
        rr.append(r.ravel())
        cc.append(c.ravel())
        ev.append(dE.ravel())
        cv.append(dCx.ravel())
        R[d] += dR
        L[d] += dL
        A += dA
    if rr:
        r, c = np.concatenate(rr), np.concatenate(cc)
        E = sp.csr_matrix((np.concatenate(ev), (r, c)), shape=(nf, nf))
        Cx = sp.csr_matrix((np.concatenate(cv), (r, c)), shape=(nf, nf))
    else:
        E = Cx = sp.csr_matrix((nf, nf))
    return {"E": E, "Cx": Cx, "R": R, "L": L, "A": A}


def _full_matrices(an):
    """``E``, ``Cx``, ``R``, ``L`` before the tie reduction (needed with a moving tie matrix)."""
    cache = an.__dict__.get("_full_matrices_cache")
    if cache is None:
        nf = an.n_full
        rr, cc, ev, cv = [], [], [], []
        R = np.zeros((nf, 6))
        L = np.zeros((nf, 6))
        for p in an._points:
            d, w = p.dofs, p.weight
            CB, CZ = p.C @ p.B, p.C @ p.Z
            r, c = np.meshgrid(d, d, indexing="ij")
            rr.append(r.ravel())
            cc.append(c.ravel())
            ev.append((w * p.B.T @ CB).ravel())
            cv.append((w * CB.T @ p.Nx).ravel())
            R[d] += w * p.B.T @ CZ
            L[d] += w * p.Nx.T @ CZ
        r, c = np.concatenate(rr), np.concatenate(cc)
        cache = {"E": sp.csr_matrix((np.concatenate(ev), (r, c)), shape=(nf, nf)),
                 "Cx": sp.csr_matrix((np.concatenate(cv), (r, c)), shape=(nf, nf)), "R": R, "L": L}
        an.__dict__["_full_matrices_cache"] = cache
    return cache


def _reduce(an, full: Dict[str, object], dT: Optional[sp.csr_matrix]) -> Dict[str, object]:
    """The derivatives of the reduced matrices ``T^T M T``, ``T^T R``, with ``dT`` the tie
    matrix's (``None`` when it does not change)."""
    T = an.T
    out = {"E": sp.csr_matrix(T.T @ full["E"] @ T), "Cx": sp.csr_matrix(T.T @ full["Cx"] @ T),
           "R": T.T @ full["R"], "L": T.T @ full["L"], "A": full["A"]}
    if dT is not None:
        f = _full_matrices(an)
        for k in ("E", "Cx"):
            out[k] = sp.csr_matrix(out[k] + dT.T @ f[k] @ T + T.T @ f[k] @ dT)
        out["R"] = out["R"] + dT.T @ f["R"]
        out["L"] = out["L"] + dT.T @ f["L"]
    return out


# ---------------------------------------------------------------------- per-point products
def _dphi(an, p, q: _Rate, sol, dsol) -> np.ndarray:
    """``d Phi`` at a point: the fields' derivatives and the point's own."""
    P0, W0f, W1f = sol
    dP0, dW0f, dW1f = dsol
    out = an._phi(p, dP0, dW0f, dW1f)
    if q.moves:
        out = out + q.dZ @ P0 + q.dB @ W0f[p.dofs]
    return out


def _energy_rate(p, q: _Rate, G: np.ndarray, dG: np.ndarray) -> np.ndarray:
    """``d (w G^T C G)`` for ``G`` changing by ``dG``."""
    CG = p.C @ G
    out = p.weight * (dG.T @ CG + CG.T @ dG) + q.dw * G.T @ CG
    if q.dC is not None:
        out = out + p.weight * G.T @ q.dC @ G
    return out


def _geometric_rows_rate(p, q: _Rate, s: np.ndarray, ds: np.ndarray) -> np.ndarray:
    """``d _geometric_rows(p, s)``: the stresses change by ``ds`` and the point by ``q``."""
    from anisotropic_section_model.solid.section_analysis import SolidSectionAnalysis, _XX, _XY, _XZ

    out = SolidSectionAnalysis._geometric_rows(p, ds)
    if q.dw or q.dy or q.dz:
        sxx, txz, txy = s[_XX], s[_XZ], s[_XY]
        y, z, dy, dz = p.y, p.z, q.dy, q.dz
        out = out + q.dw / p.weight * SolidSectionAnalysis._geometric_rows(p, s) + p.weight * np.vstack([
            0.0 * sxx, sxx * dy, sxx * dz, sxx * 2.0 * (y * dy + z * dz), 0.0 * txy, 0.0 * txz,
            txy * dy + txz * dz])
    return out


def _zero(param, an, benscoter, geometric, moves=False) -> SectionDerivative:
    z6 = np.zeros((6, 6))
    n = len(an._points)
    return SectionDerivative(param, z6, z6.copy(), {k: 0.0 for k in _MASS_KEYS}, z6.copy(),
                             np.zeros((8, 8)) if benscoter else None, 0.0,
                             np.zeros((7, 6)) if geometric else None, np.zeros(n), 0.0,
                             axial_positions=np.zeros((n, 2)) if moves else None)


def _axial_modulus_derivative(C: np.ndarray, dC: np.ndarray) -> float:
    ip = [1, 2, 3]
    Cii = C[np.ix_(ip, ip)]
    a = np.linalg.solve(Cii, C[ip, 0])
    return float(dC[0, 0] - 2.0 * dC[0, ip] @ a + a @ dC[np.ix_(ip, ip)] @ a)


def _velocity(mesh, param: LayupParameter) -> np.ndarray:
    if mesh.velocities is None:
        raise ValueError(f"section {mesh.section_id}: the 2D mesh has no node velocities for ply thicknesses "
                         "(SectionMesh.velocities, filled by SectionMesh.from_thin_walled)")
    V = mesh.velocities.get((param.laminate_id, param.ply))
    if V is None:
        return np.zeros_like(mesh.nodes)
    return np.asarray(V, dtype=np.float64)


def _mass_rate(an, rates: List[_Rate]) -> Dict[str, float]:
    mesh = an.mesh
    t = {k: 0.0 for k in _MASS_KEYS}
    for p, q in zip(an._points, rates):
        if not (q.dw or q.dy or q.dz):
            continue
        rho = mesh.materials[mesh.material_ids[p.element]].rho
        w, dw = p.weight * rho, q.dw * rho
        y, z, dy, dz = p.y, p.z, q.dy, q.dz
        t["m"] += dw
        t["my"] += dw * y + w * dy
        t["mz"] += dw * z + w * dz
        t["yy"] += dw * z * z + 2.0 * w * z * dz
        t["zz"] += dw * y * y + 2.0 * w * y * dy
        t["yz"] += dw * y * z + w * (dy * z + y * dz)
    return t


def solid_section_derivative(an, param: LayupParameter, benscoter: bool = False,
                             geometric: bool = False, taper=None) -> SectionDerivative:
    """Exact derivatives of a ``SolidSectionAnalysis`` for a ply-angle or ply-thickness ``param``.

    The fields are those of ``thin_walled.layup_sensitivity.SectionDerivative``
    (right-handed, about the origin). A thickness also gives ``axial_positions``, the
    motion of the axial quadrature points. A parameter that tags no element of the mesh
    gives zeros.

    The mesh must carry ``plies`` tags (``SectionMesh.from_thin_walled`` or a
    ``[Section_Element_Ply]`` block); a thickness needs ``velocities`` (and ``tie_points``
    where nodes are tied), which ``SectionMesh.from_thin_walled`` fills. A ``taper``
    (``solid.taper.solid_section_taper`` of this section) adds the derivative of its
    coupling (``SectionDerivative.taper``).
    """
    d = _solid_section_derivative(an, param, benscoter, geometric)
    if taper is None:
        return d
    from dataclasses import replace

    from anisotropic_section_model.thin_walled.taper import taper_derivative

    return replace(d, taper=taper_derivative(taper, d))


def _solid_section_derivative(an, param: LayupParameter, benscoter: bool, geometric: bool) -> SectionDerivative:
    mesh = an.mesh
    if mesh.plies is None:
        raise ValueError(f"section {mesh.section_id}: the 2D mesh has no ply tags (SectionMesh.plies, "
                         "[Section_Element_Ply] in section_mesh_2d.txt)")
    tag = (param.laminate_id, param.ply)
    elems = [e for e, t in enumerate(mesh.plies) if t is not None and tuple(t) == tag]
    thickness = param.kind == "thickness"
    if not elems:
        return _zero(param, an, benscoter, geometric, moves=thickness)
    if thickness:
        dC: Dict[int, np.ndarray] = {}
        V = _velocity(mesh, param)
        rates = _point_rates(an, dC, V)
        dT = _tie_rate(an, V)
    else:
        dC = {e: material_derivative(mesh.materials[mesh.material_ids[e]], mesh.theta[e], mesh.alpha[e])
              for e in elems}
        V, dT = None, None
        rates = _point_rates(an, dC, None)

    m = an._matrices
    dm = _reduce(an, _assemble(an, rates), dT)
    nd = an.n_dof
    D = an._independent(m["D_modes"])
    lu = an._saddle(D)
    n_tot = nd + 6 + D.shape[1]
    sol = an._solution
    W0f, W1f, P0, P1 = sol["W0"], sol["W1"], sol["P0"], sol["P1"]
    free_dofs = (3 * an._free_nodes[:, None] + np.arange(3)[None, :]).ravel()
    W0, W1 = W0f[free_dofs], W1f[free_dofs]

    def dS_times(W, P):
        out = np.zeros((n_tot, W.shape[1]))
        out[:nd] = dm["E"] @ W + dm["R"] @ P
        out[nd:nd + 6] = dm["R"].T @ W + dm["A"] @ P
        return out

    dX1 = lu.solve(-dS_times(W1, P1))
    dW1, dP1 = dX1[:nd], dX1[nd:nd + 6]
    Cx, L, dCx, dL = m["Cx"], m["L"], dm["Cx"], dm["L"]
    drhs = np.zeros((n_tot, 6))
    drhs[:nd] = -(dCx - dCx.T) @ W1 - (Cx - Cx.T) @ dW1 + dL @ P1 + L @ dP1
    drhs[nd:nd + 6] = -dL.T @ W1 - L.T @ dW1
    dX0 = lu.solve(drhs - dS_times(W0, P0))
    dW0, dP0 = dX0[:nd], dX0[nd:nd + 6]
    dW0f, dW1f = an.T @ dW0, an.T @ dW1
    if dT is not None:
        dW0f, dW1f = dW0f + dT @ W0, dW1f + dT @ W1
    fields, dfields = (P0, W0f, W1f), (dP0, dW0f, dW1f)

    phi = an._phi
    dF = np.zeros((6, 6))
    for p, q in zip(an._points, rates):
        dF += _energy_rate(p, q, phi(p, P0, W0f, W1f), _dphi(an, p, q, fields, dfields))
    dF = 0.5 * (dF + dF.T)
    K = an._stiffness
    dK = -K @ dF @ K
    dK = 0.5 * (dK + dK.T)

    dEx = {e: _axial_modulus_derivative(mesh.element_stiffness(e), dC[e]) for e in dC}
    Ex = an._axial_modulus
    dw = np.array([q.dw * Ex[p.element] + p.weight * dEx.get(p.element, 0.0) for p, q in zip(an._points, rates)])
    dpos = np.array([[q.dy, q.dz] for q in rates]) if thickness else None

    dH = None
    if geometric:
        dH = np.zeros((7, 6))
        for p, q in zip(an._points, rates):
            Phi = phi(p, P0, W0f, W1f)
            ds = p.C @ _dphi(an, p, q, fields, dfields)
            if q.dC is not None:
                ds = ds + q.dC @ Phi
            dH += _geometric_rows_rate(p, q, p.C @ Phi, ds)

    dK8, dI = None, 0.0
    if benscoter:
        dK8, dI = _benscoter_derivative(an, rates, dEx, dm, dT, V, fields, dfields, dK)
    dmass = _mass_rate(an, rates)
    from anisotropic_section_model.thin_walled.layup_sensitivity import _mass_matrix

    return SectionDerivative(param, dF, dK, dmass, _mass_matrix(dmass), dK8, dI, dH, dw, 0.0,
                             axial_positions=dpos)


def _constraint_rate(an, rates, V, dT) -> np.ndarray:
    """``d D_modes[:, :4]^T`` (4, n_dof): ``D = T^T M modes`` with the nodes moving at ``V``."""
    modes = an._full_modes[:, :4]
    dmodes = np.zeros_like(modes)
    dmodes[1::3, 3], dmodes[2::3, 3] = -V[:, 1], V[:, 0]
    nf = an.n_full
    Mm, dMm = np.zeros((nf, 4)), np.zeros((nf, 4))
    for p, q in zip(an._points, rates):
        Mp = np.kron(np.outer(p.N, p.N), np.eye(3))
        d = p.dofs
        Mm[d] += p.weight * Mp @ modes[d]
        dMm[d] += q.dw * Mp @ modes[d] + p.weight * Mp @ dmodes[d]
    out = an.T.T @ dMm
    if dT is not None:
        out = out + dT.T @ Mm
    return np.asarray(out).T


def _benscoter_derivative(an, rates, dEx, dm, dT, V, sol, dsol, dK):
    """``(d benscoter_stiffness, d warping_inertia)``, right-handed (module docstring, step 5)."""
    m = an._matrices
    nd = an.n_dof
    mesh = an.mesh
    P0, W0f, W1f = sol
    free_dofs = (3 * an._free_nodes[:, None] + np.arange(3)[None, :]).ravel()

    D4 = an._independent(m["D_modes"][:, :4])
    lu = an._saddle(D4, with_psi=False)
    n4 = nd + D4.shape[1]

    # torsional warping: E omega = -R[:, 5]. Its in-plane rotation is fixed by the constraint
    # D^T omega = 0 and stays in the strains of psi' (no warping cancels it), so the
    # constraint moves with the nodes: D^T d omega = -dD^T omega
    omega_full, c = an._torsion_warping
    omega = omega_full[free_dofs]
    rhs = np.zeros(n4)
    rhs[:nd] = -dm["R"][:, 5] - dm["E"] @ omega
    if V is not None:
        Dm = m["D_modes"][:, :4]
        Q = np.linalg.lstsq(Dm, D4, rcond=None)[0]             # D4 = Dm Q
        rhs[nd:] = -Q.T @ (_constraint_rate(an, rates, V, dT) @ omega)
    domega_full = an.T @ lu.solve(rhs)[:nd]
    if dT is not None:
        domega_full = domega_full + dT @ omega

    # rigid part c = G^-1 b in the axial modulus
    G = np.zeros((3, 3))
    dG = np.zeros((3, 3))
    db = np.zeros(3)
    for p, q in zip(an._points, rates):
        Ex = an._axial_modulus[p.element]
        dEx_p = dEx.get(p.element, 0.0)
        r = np.array([1.0, p.z, -p.y])
        dr = np.array([0.0, q.dz, -q.dy])
        om = p.N @ omega_full[p.dofs[0::3]]
        dom = p.N @ domega_full[p.dofs[0::3]]
        wE, dwE = p.weight * Ex, q.dw * Ex + p.weight * dEx_p
        G += wE * np.outer(r, r)
        dG += dwE * np.outer(r, r) + wE * (np.outer(dr, r) + np.outer(r, dr))
        db += dwE * r * om + wE * (dr * om + r * dom)
    dc = np.linalg.solve(G, db - dG @ c)
    modes = an._full_modes[:, [0, 4, 5]]
    dmodes = np.zeros_like(modes)
    if V is not None:
        dmodes[0::3, 1] = V[:, 1]
        dmodes[0::3, 2] = -V[:, 0]
    om_t = an._omega_tilde
    dom_t = domega_full - modes @ dc - dmodes @ c

    # relaxed gradient: E g = -Cx omega_tilde
    g = an._relaxed_gradient
    rhs = np.zeros(n4)
    rhs[:nd] = -(dm["Cx"] @ om_t[free_dofs]) - m["Cx"] @ dom_t[free_dofs] - dm["E"] @ g[free_dofs]
    dg = an.T @ lu.solve(rhs)[:nd]
    if dT is not None:
        dg = dg + dT @ g[free_dofs]

    K = an._stiffness
    dK8 = np.zeros((8, 8))
    dI = 0.0
    for p, q in zip(an._points, rates):
        Gp = an._benscoter_operator(p)
        dGp = np.zeros((6, 8))
        dGp[:, :6] = _dphi(an, p, q, sol, dsol) @ K + an._phi(p, P0, W0f, W1f) @ dK
        dGp[:, 6] = p.B @ dom_t[p.dofs]
        dGp[:, 7] = p.Nx @ dom_t[p.dofs] + p.B @ dg[p.dofs]
        if q.moves:
            dGp[:, 6] += q.dB @ om_t[p.dofs]
            dGp[:, 7] += q.dB @ g[p.dofs]
        dK8 += _energy_rate(p, q, Gp, dGp)
        om = p.N @ om_t[p.dofs[0::3]]
        dom = p.N @ dom_t[p.dofs[0::3]]
        rho = mesh.materials[mesh.material_ids[p.element]].rho
        dI += rho * (q.dw * om * om + p.weight * 2.0 * om * dom)
    dK8 = 0.5 * (dK8 + dK8.T)
    dK8[:6, :6] = dK
    return dK8, float(dI)

