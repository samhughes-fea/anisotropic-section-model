# pre_processing/section_integration/solid/cross_section_modes.py
"""Cross-section deformation modes of sections meshed in 2D: GBT on a continuum section.

The same generalised beam as ``thin_walled.cross_section_modes`` (and the same strain and
mass orders, so ``LinearGBTBeamElement3D`` takes either), with the walls replaced by the
2D mesh of ``SolidSectionAnalysis``. Each section point moves as::

    u_x       = u - y theta_z + z theta_y + Omega_j b_j + u_k a_k'
    (u_y, u_z) = (v - z theta_x, w + y theta_x) + S psi6 + d_k a_k

- ``Omega_j`` (axial warping, amplitudes ``b_j``) span the axial part of the Saint-Venant
  warping of the six beam strains and their rates (``W0``, ``W1``), less the rigid axial
  fields ``1, y, z``. With ``b`` free a uniform or linearly varying force state is the
  Saint-Venant one; ``b = 0`` at a support keeps the section plane there.
- ``S psi6`` is the in-plane (Poisson) part of the Saint-Venant warping per unit beam
  strain, slaved to the beam strains. Without it the plane section would be held in
  plane strain and EA would be up to 35 % stiff.
- ``d_k`` (amplitudes ``a_k``) are distortional modes: the lowest in-plane modes of the
  section's own plane-strain stiffness against its area (``K d = lambda M d``), the three
  rigid in-plane motions removed. ``u_k`` is the axial warping that goes with them, the
  field that minimises the out-of-plane shear energy of ``a_k'`` (the continuum form of the
  Vlasov hypothesis of no membrane shear), with zero mean.
- The axial strain of ``a_k''`` (``N_x u_k``) comes with a slaved through-thickness strain
  that makes the stress across the wall zero (``sigma_nn = 0``, ``n`` normal to the wall
  direction ``alpha`` of each element), as in plate theory. Without it the wall's
  longitudinal bending is held in plane strain: ``C`` is ``(1 - nu) / ((1 + nu)(1 - 2 nu))``
  times the plate value instead of ``1 / (1 - nu^2)``, 22 % stiff for ``nu = 0.3``, and
  local buckling 6 to 10 % high. The membrane part of ``N_x u_k`` keeps its hoop strain
  restrained (the thin-walled modes release it, ``N_ss = 0``), so for a membrane-dominated
  mode ``C`` is up to ``1 / (1 - nu^2)`` above the thin-walled value.

3D strains, Voigt ``[xx, yy, zz, yz, xz, xy]``, per generalised strain
``[psi6, b'(p), b(p), a''(m), a'(m), a(m)]``, with ``B`` and ``N_x`` the section's
in-plane-gradient and x-derivative operators::

    psi6: Z + B S          b': N_x Omega      b: B Omega
    a'':  N_x u_k          a': B u_k + N_x d_k          a: B d_k

The stiffness is ``int H^T C H dA``, so material coupling (Poisson, anisotropy) between
distortion and the beam strains is kept. Extra shear-lag modes are not offered for 2D
sections (``n_shear_modes = 0``), and wall loads need a thin-walled section.

Public API
----------
SolidCrossSectionModes.
"""

from __future__ import annotations

from functools import cached_property
from typing import Tuple

import numpy as np
import numpy.typing as npt
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["SolidCrossSectionModes"]

#: distortional modes computed at most
MAX_MODES = 12


class SolidCrossSectionModes:
    """GBT modes, stiffness and mass of a ``SolidSectionAnalysis`` (right-handed)."""

    def __init__(self, analysis, max_modes: int = MAX_MODES) -> None:
        self.analysis = analysis
        self.max_modes = int(max_modes)

    # ------------------------------------------------------------------ #
    # Assembled section operators (full nodal vectors, 3 per node: g_x, g_y, g_z)
    # ------------------------------------------------------------------ #
    @cached_property
    def _operators(self):
        an = self.analysis
        nf = an.n_full
        rows, cols = [], []
        vals = {k: [] for k in ("Exx", "Exd", "Kd", "Ax", "Md")}
        by_el = {}
        for p in an._points:
            by_el.setdefault(p.element, []).append(p)
        for pts in by_el.values():
            d = pts[0].dofs
            nd = d.size
            blocks = {k: np.zeros((nd, nd)) for k in vals}
            xs, ys = np.arange(0, nd, 3), np.r_[np.arange(1, nd, 3), np.arange(2, nd, 3)]
            for p in pts:
                w = p.weight
                Bx = p.B[:, xs]                     # gradient of g_x: the out-of-plane shears
                Nd = p.Nx[:, ys]                    # d_k into the out-of-plane shears
                Bd = p.B[:, ys]                     # in-plane strains of d_k
                b = blocks
                b["Exx"][np.ix_(xs, xs)] += w * Bx.T @ p.C @ Bx
                b["Exd"][np.ix_(xs, ys)] += w * Bx.T @ p.C @ Nd
                b["Kd"][np.ix_(ys, ys)] += w * Bd.T @ p.C @ Bd
                b["Ax"][np.ix_(xs, xs)] += w * p.C[0, 0] * np.outer(p.N, p.N)
                b["Md"][np.ix_(ys, ys)] += w * np.kron(np.eye(2), np.outer(p.N, p.N))
            r, c = np.meshgrid(d, d, indexing="ij")
            rows.append(r.ravel())
            cols.append(c.ravel())
            for k in vals:
                vals[k].append(blocks[k].ravel())
        r, c = np.concatenate(rows), np.concatenate(cols)
        T = an.T
        return {k: (T.T @ sp.csr_matrix((np.concatenate(v), (r, c)), shape=(nf, nf)) @ T).tocsr()
                for k, v in vals.items()}

    def _components(self, which: str) -> np.ndarray:
        """Free DOFs of the ``x`` or the in-plane (``yz``) components (free DOF ``3 k + c``)."""
        j = np.arange(self.analysis.n_dof)
        return j[j % 3 == 0] if which == "x" else j[j % 3 != 0]

    def _free(self, Wf: np.ndarray) -> np.ndarray:
        """Free-DOF values of full nodal fields (the free nodes' own rows)."""
        nodes = self.analysis._free_nodes
        return Wf[(3 * nodes[:, None] + np.arange(3)[None, :]).ravel()]

    # ------------------------------------------------------------------ #
    # Warping modes
    # ------------------------------------------------------------------ #
    @cached_property
    def _warping(self) -> np.ndarray:
        """``Omega`` (n_dof, p): the axial Saint-Venant warping less ``1, y, z``, ``A``-orthonormal."""
        an = self.analysis
        A = self._operators["Ax"]
        sol = an._solution
        x = self._components("x")
        free = self._free
        W = free(np.hstack([sol["W0"], sol["W1"]]))
        Wx = np.zeros_like(W)
        Wx[x] = W[x]
        y, z = an.mesh.nodes[:, 0], an.mesh.nodes[:, 1]
        R = np.zeros((an.n_full, 3))
        R[0::3, 0], R[0::3, 1], R[0::3, 2] = 1.0, y, z
        R = free(R)
        Q = _orthonormal(R, A)
        Wx = Wx - Q @ (Q.T @ (A @ Wx))
        return _orthonormal(Wx, A)

    @property
    def n_sv_warping(self) -> int:
        return self._warping.shape[1]

    @property
    def n_shear_modes(self) -> int:
        return 0

    # ------------------------------------------------------------------ #
    # Poisson (in-plane Saint-Venant) warping per unit beam strain
    # ------------------------------------------------------------------ #
    @cached_property
    def _poisson(self) -> np.ndarray:
        an = self.analysis
        sol = an._solution
        S = (sol["W0"] @ an.stiffness).copy()          # full nodal warping per unit strain
        S[0::3] = 0.0
        return S

    # ------------------------------------------------------------------ #
    # Distortional modes
    # ------------------------------------------------------------------ #
    @cached_property
    def _modes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(lambda, D, U)``: eigenvalues, in-plane fields and their axial warping (free DOFs)."""
        an = self.analysis
        ops = self._operators
        yz = self._components("yz")
        x = self._components("x")
        K = ops["Kd"][yz][:, yz]
        M = ops["Md"][yz][:, yz]
        y, z = an.mesh.nodes[:, 0], an.mesh.nodes[:, 1]
        Rf = np.zeros((an.n_full, 3))
        Rf[1::3, 0] = 1.0
        Rf[2::3, 1] = 1.0
        Rf[1::3, 2], Rf[2::3, 2] = -z, y
        Rr = self._free(Rf)[yz]
        k = min(self.max_modes + 8, K.shape[0] - 2)
        scale = float(abs(K.diagonal()).max()) / float(abs(M.diagonal()).max())
        lam, X = spla.eigsh(K.tocsc(), k=k, M=M.tocsc(), sigma=-1e-6 * scale, which="LM")
        # remove the rigid in-plane motions (M-orthogonal complement), then re-solve in the span
        Q = _orthonormal(Rr, M)
        X = X - Q @ (Q.T @ (M @ X))
        X = _orthonormal(X, M)
        lam, Y = sla.eigh(X.T @ (K @ X), X.T @ (M @ X))
        keep = lam > 1e-8 * max(lam.max(), 1e-300)
        lam, X = lam[keep], X @ Y[:, keep]
        lam, X = lam[:self.max_modes], X[:, :self.max_modes]
        D = np.zeros((an.n_dof, X.shape[1]))
        D[yz] = X
        D = D / np.abs(D).max(axis=0)
        D = D * np.sign(D[np.argmax(np.abs(D), axis=0), np.arange(D.shape[1])])
        # axial warping of each mode: min int (grad u + d)^T C (grad u + d), with int u = 0
        Exx = ops["Exx"][x][:, x]
        Exd = ops["Exd"][x][:, yz]
        ones = np.ones(x.size)
        a = ops["Ax"][x][:, x] @ ones
        S = sp.bmat([[Exx, sp.csr_matrix(a[:, None])], [sp.csr_matrix(a[None, :]), None]], format="csc")
        rhs = np.vstack([-(Exd @ D[yz]), np.zeros((1, D.shape[1]))])
        sol = spla.splu(S).solve(rhs)
        U = np.zeros_like(D)
        U[x] = sol[:x.size]
        # take out the axial and bending content of u_k with the rigid-section modes it
        # belongs to (axial: u = 1; translations: u = -y with d = e_y, u = -z with d = e_z),
        # so the pair stays free of extra shear and does not couple with eps or kappa (the
        # thin-walled modes are orthogonal to the rigid-section modes in the same way)
        Ax = ops["Ax"]
        R = np.zeros((an.n_full, 3))
        R[0::3, 0], R[0::3, 1], R[0::3, 2] = 1.0, an.mesh.nodes[:, 0], an.mesh.nodes[:, 1]
        Q = self._free(R)
        c = np.linalg.solve(Q.T @ (Ax @ Q), Q.T @ (Ax @ U))       # U ~ Q c
        U = U - Q @ c
        ey = np.zeros(an.n_dof)
        ez = np.zeros(an.n_dof)
        ey[1::3], ez[2::3] = 1.0, 1.0
        D = D + np.outer(ey, c[1]) + np.outer(ez, c[2])
        # and the torsion that goes with the rigid rotation (d = (-z, y), u = the Saint-Venant
        # torsion warping): no coupling with the Saint-Venant torsion shear
        yv, zv = an.mesh.nodes[:, 0], an.mesh.nodes[:, 1]
        rot = np.zeros(an.n_full)
        rot[1::3], rot[2::3] = -zv, yv
        om = np.zeros(an.n_full)
        om[0::3] = (an._solution["W0"] @ an.stiffness)[0::3, 5]
        rot, om = self._free(rot), self._free(om)
        T = an.T
        rf, of, Df, Uf = T @ rot, T @ om, T @ D, T @ U
        num = np.zeros(D.shape[1])
        den = 0.0
        for p in an._points:
            dd = p.dofs
            sT = p.Z[:, 5] + p.B @ of[dd]
            Cs = p.weight * (p.C @ sT)
            den += Cs @ sT
            num += Cs @ (p.B @ Uf[dd] + p.Nx @ Df[dd])
        beta = num / den
        D = D - np.outer(rot, beta)
        U = U - np.outer(om, beta)
        return lam, D, U

    @property
    def n_modes(self) -> int:
        return self._modes[1].shape[1]

    @property
    def eigenvalues(self) -> npt.NDArray[np.float64]:
        return self._modes[0].copy()

    def mode(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Full nodal in-plane field ``d_k`` and axial warping ``u_k`` (n_full each)."""
        T = self.analysis.T
        return T @ self._modes[1][:, k], T @ self._modes[2][:, k]

    def _check(self, n_modes: int, n_shear: int) -> None:
        if n_shear:
            raise ValueError("2D sections offer no extra shear-lag modes (gbt_shear_modes must be 0)")
        if not 0 <= n_modes <= self.n_modes:
            raise ValueError(f"asked for {n_modes} cross-section modes; the section has {self.n_modes}")

    # ------------------------------------------------------------------ #
    # GBT section stiffness and mass
    # ------------------------------------------------------------------ #
    def _fields(self, n_modes: int):
        T = self.analysis.T
        Om = T @ self._warping
        D = T @ self._modes[1][:, :n_modes]
        U = T @ self._modes[2][:, :n_modes]
        return Om, D, U, self._poisson

    def strain_map(self, p, n_modes: int) -> npt.NDArray[np.float64]:
        """``H`` (6, 6 + 2p + 3m): 3D strains at section point ``p`` per GBT strain."""
        Om, D, U, S = self._fields(n_modes)
        return self._H(p, Om, D, U, S)

    def _wall_normal(self, element: int) -> np.ndarray:
        """Voigt strain of a unit ``eps_nn`` across the wall of ``element`` (``n`` at ``alpha + 90``)."""
        a = np.radians(self.analysis.mesh.alpha[element])
        ny, nz = -np.sin(a), np.cos(a)
        return np.array([0.0, ny * ny, nz * nz, 2.0 * ny * nz, 0.0, 0.0])

    def _H(self, p, Om, D, U, S):
        P, m = Om.shape[1], D.shape[1]
        oa = 6 + 2 * P
        d = p.dofs
        H = np.zeros((6, oa + 3 * m))
        H[:, :6] = p.Z + p.B @ S[d]
        H[:, 6:6 + P] = p.Nx @ Om[d]
        H[:, 6 + P:oa] = p.B @ Om[d]
        Ha = p.Nx @ U[d]
        if m:                                   # sigma_nn = 0: the wall is free to thin
            n = self._wall_normal(p.element)
            Cn = p.C @ n
            Ha = Ha - np.outer(n, (Ha.T @ Cn) / (n @ Cn))
        H[:, oa:oa + m] = Ha
        H[:, oa + m:oa + 2 * m] = p.B @ U[d] + p.Nx @ D[d]
        H[:, oa + 2 * m:] = p.B @ D[d]
        return H

    def stiffness(self, n_modes: int, n_shear: int = 0) -> npt.NDArray[np.float64]:
        """Right-handed GBT section stiffness, square of size ``6 + 2p + 3m``."""
        self._check(n_modes, n_shear)
        Om, D, U, S = self._fields(n_modes)
        n = 6 + 2 * Om.shape[1] + 3 * n_modes
        K = np.zeros((n, n))
        for p in self.analysis._points:
            H = self._H(p, Om, D, U, S)
            K += p.weight * H.T @ p.C @ H
        return 0.5 * (K + K.T)

    def mass(self, n_modes: int, n_shear: int = 0) -> npt.NDArray[np.float64]:
        """Section mass for ``[u, v, w, theta_x, theta_y, theta_z, b(p), a(m), a'(m)]``."""
        self._check(n_modes, n_shear)
        an = self.analysis
        Om, D, U, _ = self._fields(n_modes)
        P, m = Om.shape[1], n_modes
        n = 6 + P + 2 * m
        M = np.zeros((n, n))
        mesh = an.mesh
        for p in an._points:
            rho = mesh.materials[mesh.material_ids[p.element]].rho
            N, d = p.N, p.dofs
            y, z = p.y, p.z
            Pi = np.zeros((3, n))
            Pi[0, [0, 4, 5]] = [1.0, z, -y]
            Pi[1, [1, 3]] = [1.0, -z]
            Pi[2, [2, 3]] = [1.0, y]
            Pi[0, 6:6 + P] = N @ Om[d[0::3]]
            Pi[1, 6 + P:6 + P + m] = N @ D[d[1::3]]
            Pi[2, 6 + P:6 + P + m] = N @ D[d[2::3]]
            Pi[0, 6 + P + m:] = N @ U[d[0::3]]
            M += p.weight * rho * Pi.T @ Pi
        return 0.5 * (M + M.T)

    def shape_moments(self, n_modes: int, n_shear: int = 0) -> npt.NDArray[np.float64]:
        """Moments of each field of the mass order, as ``CrossSectionModes.shape_moments``."""
        from pre_processing.section_integration.thin_walled.cross_section_modes import _box, _monomials

        self._check(n_modes, n_shear)
        an = self.analysis
        Om, D, U, _ = self._fields(n_modes)
        P, m = Om.shape[1], n_modes
        n = 6 + P + 2 * m
        c, h = _box(np.array([[p.y, p.z] for p in an._points]))
        out = np.zeros((n, 30))
        for p in an._points:
            N, d = p.N, p.dofs
            y, z = p.y, p.z
            Pi = np.zeros((3, n))
            Pi[0, [0, 4, 5]] = [1.0, z, -y]
            Pi[1, [1, 3]] = [1.0, -z]
            Pi[2, [2, 3]] = [1.0, y]
            Pi[0, 6:6 + P] = N @ Om[d[0::3]]
            Pi[1, 6 + P:6 + P + m] = N @ D[d[1::3]]
            Pi[2, 6 + P:6 + P + m] = N @ D[d[2::3]]
            Pi[0, 6 + P + m:] = N @ U[d[0::3]]
            out += p.weight * np.kron(Pi.T, _monomials((np.array([y, z]) - c) / h)[None, :]).reshape(n, 30)
        return out

    def geometric_tensor(self, n_modes: int, n_shear: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Second-order terms of the in-plane motion ``q = [v, w, theta_x, a(m)]``, as
        ``CrossSectionModes.geometric_tensor``: ``X = int sigma_xx Pi^T Pi dA`` and
        ``Y = int (tau_xy Pi^T Pi_,y + tau_xz Pi^T Pi_,z) dA`` per unit GBT strain (each
        ``(3 + m, 3 + m, 6 + 2p + 3m)``, right-handed). The Poisson part ``S psi6`` of the
        in-plane motion is left out of ``Pi``, as in the beam's own second-order terms.
        """
        self._check(n_modes, n_shear)
        Om, D, U, S = self._fields(n_modes)
        m = n_modes
        nq, nE = 3 + m, 6 + 2 * Om.shape[1] + 3 * m
        TX = np.zeros((nq, nq, nE))
        TY = np.zeros((nq, nq, nE))
        for p in self.analysis._points:
            sig = p.C @ self._H(p, Om, D, U, S)                    # 3D stresses per unit E
            d = p.dofs
            dNy, dNz = p.B[1, 1::3], p.B[2, 2::3]                   # dN/dy, dN/dz
            Pi = np.zeros((2, nq))
            Pi[0, [0, 2]] = [1.0, -p.z]
            Pi[1, [1, 2]] = [1.0, p.y]
            Pi[0, 3:] = p.N @ D[d[1::3]]
            Pi[1, 3:] = p.N @ D[d[2::3]]
            Py = np.zeros((2, nq))
            Pz = np.zeros((2, nq))
            Py[1, 2], Pz[0, 2] = 1.0, -1.0
            Py[0, 3:], Py[1, 3:] = dNy @ D[d[1::3]], dNy @ D[d[2::3]]
            Pz[0, 3:], Pz[1, 3:] = dNz @ D[d[1::3]], dNz @ D[d[2::3]]
            TX += p.weight * np.einsum("ia,ib,e->abe", Pi, Pi, sig[0])
            TY += p.weight * (np.einsum("ia,ib,e->abe", Pi, Py, sig[5]) + np.einsum("ia,ib,e->abe", Pi, Pz, sig[4]))
        return TX, TY

    def brazier_tensor(self, n_modes: int, n_shear: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Brazier terms ``(T, Q)`` as ``CrossSectionModes.brazier_tensor``: ``T = int g sigma_xx dA``
        and ``Q = int E_x g g dA`` with ``g_k = (d_z, -d_y)`` the mode's in-plane motion and
        ``E_x`` the axial modulus (in-plane stresses free), right-handed."""
        self._check(n_modes, n_shear)
        Om, D, U, S = self._fields(n_modes)
        m = n_modes
        nE = 6 + 2 * Om.shape[1] + 3 * m
        T = np.zeros((m, 2, nE))
        Q = np.zeros((m, m, 2, 2))
        Ex = self.analysis._axial_modulus
        for p in self.analysis._points:
            sig = p.C @ self._H(p, Om, D, U, S)
            d = p.dofs
            gk = np.stack([p.N @ D[d[2::3]], -(p.N @ D[d[1::3]])], axis=1)      # (m, 2)
            T += p.weight * np.einsum("kc,e->kce", gk, sig[0])
            Q += p.weight * Ex[p.element] * np.einsum("kc,ld->klcd", gk, gk)
        return T, Q


def _orthonormal(W: np.ndarray, A) -> np.ndarray:
    """Columns spanning ``W``, orthonormal in ``A`` (dependent columns dropped)."""
    AW = A @ W
    nrm = np.sqrt(np.maximum(np.einsum("ik,ik->k", W, AW), 0.0))
    big = nrm > 1e-12 * max(nrm.max(), 1e-300)
    W = W[:, big] / nrm[big]
    G = W.T @ (A @ W)
    lam, V = np.linalg.eigh(0.5 * (G + G.T))
    keep = lam > 1e-10 * max(lam.max(), 1e-300)
    return W @ V[:, keep] / np.sqrt(lam[keep])
