# anisotropic_section_model/solid/layup_sensitivity.py
"""Exact derivatives of a 2D-meshed section's properties with respect to ply angles.

A parameter is one ply of one laminate (``thin_walled.layup_sensitivity.LayupParameter``):
the elements the mesh tags with that ply (``SectionMesh.plies``) turn with its angle. Only
``angle`` parameters are supported: a thickness change moves the mesh nodes.

Every step is differentiated exactly, as for thin-walled sections:

1. **Material.** ``C = T C_m T^T`` with ``T`` the bond matrix of the material axes ``R``;
   ``T`` is bilinear in ``R``, so ``dT = T(dR, R) + T(R, dR)`` with ``dR`` the derivative of
   the axes, and ``dC = dT C_m T^T + T C_m dT^T``.
2. **Section.** ``E``, ``Cx``, ``R``, ``L`` and ``A`` are linear in ``C``; the two
   Saint-Venant solves (``SolidSectionAnalysis._solution``) give ``dX = S^-1 (d rhs - dS X)``
   (``dS`` is zero on the warping-constraint rows and columns), then
   ``dF = sum w (dPhi^T C Phi + Phi^T C dPhi + Phi^T dC Phi)`` and ``dK = -K dF K``.
3. **Benscoter** (``benscoter=True``). The torsional warping solve, the 3 x 3 removal of its
   rigid part in the axial modulus ``E_x = C_xx - C_xi C_ii^-1 C_ix`` and the relaxed
   gradient solve are differentiated; then ``K8 = sum w G^T C G`` and
   ``int rho omega^2`` by the product rule.
4. **Second-order terms** (``geometric=True``) and the axial weights ``w E_x`` are linear in
   the stresses and in ``E_x``.

The mass does not depend on ply angles, so its derivative is zero.

Public API
----------
material_derivative, solid_section_derivative.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from anisotropic_section_model.solid.material import _PAIRS, bond_matrix, material_axes
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


def _assemble(an, dC: Dict[int, np.ndarray]) -> Dict[str, object]:
    """``E``, ``Cx``, ``R``, ``L``, ``A`` of ``an._matrices`` with the element stiffness ``dC``
    (zero for elements not listed)."""
    nf = an.n_full
    rr, cc, ev, cv = [], [], [], []
    R = np.zeros((nf, 6))
    L = np.zeros((nf, 6))
    A = np.zeros((6, 6))
    for p in an._points:
        C = dC.get(p.element)
        if C is None:
            continue
        d, w = p.dofs, p.weight
        CB, CZ = C @ p.B, C @ p.Z
        r, c = np.meshgrid(d, d, indexing="ij")
        rr.append(r.ravel())
        cc.append(c.ravel())
        ev.append((w * p.B.T @ CB).ravel())
        cv.append((w * CB.T @ p.Nx).ravel())
        R[d] += w * p.B.T @ CZ
        L[d] += w * p.Nx.T @ CZ
        A += w * p.Z.T @ CZ
    T = an.T
    if rr:
        r, c = np.concatenate(rr), np.concatenate(cc)
        E = T.T @ sp.csr_matrix((np.concatenate(ev), (r, c)), shape=(nf, nf)) @ T
        Cx = T.T @ sp.csr_matrix((np.concatenate(cv), (r, c)), shape=(nf, nf)) @ T
    else:
        E = Cx = sp.csr_matrix((an.n_dof, an.n_dof))
    return {"E": sp.csr_matrix(E), "Cx": sp.csr_matrix(Cx), "R": T.T @ R, "L": T.T @ L, "A": A}


def _zero(param, an, benscoter, geometric) -> SectionDerivative:
    z6 = np.zeros((6, 6))
    return SectionDerivative(param, z6, z6.copy(), {k: 0.0 for k in _MASS_KEYS}, z6.copy(),
                             np.zeros((8, 8)) if benscoter else None, 0.0,
                             np.zeros((7, 6)) if geometric else None, np.zeros(len(an._points)), 0.0)


def _axial_modulus_derivative(C: np.ndarray, dC: np.ndarray) -> float:
    ip = [1, 2, 3]
    Cii = C[np.ix_(ip, ip)]
    a = np.linalg.solve(Cii, C[ip, 0])
    return float(dC[0, 0] - 2.0 * dC[0, ip] @ a + a @ dC[np.ix_(ip, ip)] @ a)


def solid_section_derivative(an, param: LayupParameter, benscoter: bool = False,
                             geometric: bool = False) -> SectionDerivative:
    """Exact derivatives of a ``SolidSectionAnalysis`` for a ply-angle ``param``.

    The fields are those of ``thin_walled.layup_sensitivity.SectionDerivative``
    (right-handed, about the origin); the mass derivatives are zero. A parameter that tags
    no element of the mesh gives zeros. The mesh must carry ``plies`` tags
    (``SectionMesh.from_thin_walled`` or a ``[Section_Element_Ply]`` block).
    """
    if param.kind != "angle":
        raise NotImplementedError("2D sections take ply-angle sensitivities only (a thickness moves the mesh)")
    mesh = an.mesh
    if mesh.plies is None:
        raise ValueError(f"section {mesh.section_id}: the 2D mesh has no ply tags (SectionMesh.plies, "
                         "[Section_Element_Ply] in section_mesh_2d.txt)")
    tag = (param.laminate_id, param.ply)
    elems = [e for e, t in enumerate(mesh.plies) if t is not None and tuple(t) == tag]
    if not elems:
        return _zero(param, an, benscoter, geometric)
    dC = {e: material_derivative(mesh.materials[mesh.material_ids[e]], mesh.theta[e], mesh.alpha[e])
          for e in elems}

    m = an._matrices
    dm = _assemble(an, dC)
    nd = an.n_dof
    D = an._independent(m["D_modes"])
    lu = an._saddle(D)
    n_tot = nd + 6 + D.shape[1]
    sol = an._solution
    W0f, W1f, P0, P1 = sol["W0"], sol["W1"], sol["P0"], sol["P1"]
    # the free warping: T is the identity on the free nodes
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

    phi = an._phi
    dF = np.zeros((6, 6))
    for p in an._points:
        Phi = phi(p, P0, W0f, W1f)
        dPhi = phi(p, dP0, dW0f, dW1f)
        CdPhi = p.C @ dPhi
        dF += p.weight * (dPhi.T @ p.C @ Phi + Phi.T @ CdPhi)
        if p.element in dC:
            dF += p.weight * Phi.T @ dC[p.element] @ Phi
    dF = 0.5 * (dF + dF.T)
    K = an._stiffness
    dK = -K @ dF @ K
    dK = 0.5 * (dK + dK.T)

    dEx = {e: _axial_modulus_derivative(mesh.element_stiffness(e), dC[e]) for e in elems}
    dw = np.array([p.weight * dEx.get(p.element, 0.0) for p in an._points])

    dH = None
    if geometric:
        rows = type(an)._geometric_rows
        dH = np.zeros((7, 6))
        for p in an._points:
            Phi = phi(p, P0, W0f, W1f)
            s = p.C @ phi(p, dP0, dW0f, dW1f)
            if p.element in dC:
                s = s + dC[p.element] @ Phi
            dH += rows(p, s)

    dK8, dI = None, 0.0
    if benscoter:
        dK8, dI = _benscoter_derivative(an, dC, dEx, dm, (P0, W0f, W1f), (dP0, dW0f, dW1f), dK)
    z6 = np.zeros((6, 6))
    return SectionDerivative(param, dF, dK, {k: 0.0 for k in _MASS_KEYS}, z6, dK8, dI, dH, dw, 0.0)


def _benscoter_derivative(an, dC, dEx, dm, sol, dsol, dK):
    """``(d benscoter_stiffness, d warping_inertia)``, right-handed (module docstring, step 3)."""
    m = an._matrices
    nd = an.n_dof
    mesh = an.mesh
    P0, W0f, W1f = sol
    dP0, dW0f, dW1f = dsol
    free_dofs = (3 * an._free_nodes[:, None] + np.arange(3)[None, :]).ravel()

    D4 = an._independent(m["D_modes"][:, :4])
    lu = an._saddle(D4, with_psi=False)
    n4 = nd + D4.shape[1]

    # torsional warping: E omega = -R[:, 5]
    omega_full, c = an._torsion_warping
    omega = omega_full[free_dofs]
    rhs = np.zeros(n4)
    rhs[:nd] = -dm["R"][:, 5] - dm["E"] @ omega
    domega_full = an.T @ lu.solve(rhs)[:nd]

    # rigid part c = G^-1 b in the axial modulus
    G = np.zeros((3, 3))
    dG = np.zeros((3, 3))
    db = np.zeros(3)
    for p in an._points:
        Ex = an._axial_modulus[p.element]
        dEx_p = dEx.get(p.element, 0.0)
        r = np.array([1.0, p.z, -p.y])
        om = p.N @ omega_full[p.dofs[0::3]]
        dom = p.N @ domega_full[p.dofs[0::3]]
        G += p.weight * Ex * np.outer(r, r)
        dG += p.weight * dEx_p * np.outer(r, r)
        db += p.weight * (dEx_p * r * om + Ex * r * dom)
    dc = np.linalg.solve(G, db - dG @ c)
    modes = an._full_modes[:, [0, 4, 5]]
    om_t = an._omega_tilde
    dom_t = domega_full - modes @ dc

    # relaxed gradient: E g = -Cx omega_tilde
    g = an._relaxed_gradient
    rhs = np.zeros(n4)
    rhs[:nd] = -(dm["Cx"] @ om_t[free_dofs]) - m["Cx"] @ dom_t[free_dofs] - dm["E"] @ g[free_dofs]
    dg = an.T @ lu.solve(rhs)[:nd]

    K = an._stiffness
    dK8 = np.zeros((8, 8))
    dI = 0.0
    for p in an._points:
        Gp = an._benscoter_operator(p)
        dGp = np.zeros((6, 8))
        dGp[:, :6] = an._phi(p, dP0, dW0f, dW1f) @ K + an._phi(p, P0, W0f, W1f) @ dK
        dGp[:, 6] = p.B @ dom_t[p.dofs]
        dGp[:, 7] = p.Nx @ dom_t[p.dofs] + p.B @ dg[p.dofs]
        CG = p.C @ Gp
        dK8 += p.weight * (dGp.T @ CG + CG.T @ dGp)
        if p.element in dC:
            dK8 += p.weight * Gp.T @ dC[p.element] @ Gp
        om = p.N @ om_t[p.dofs[0::3]]
        dom = p.N @ dom_t[p.dofs[0::3]]
        dI += p.weight * mesh.materials[mesh.material_ids[p.element]].rho * 2.0 * om * dom
    dK8 = 0.5 * (dK8 + dK8.T)
    dK8[:6, :6] = dK
    return dK8, float(dI)
