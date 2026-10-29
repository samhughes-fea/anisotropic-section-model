# pre_processing/section_integration/thin_walled/cross_section_modes.py
"""Cross-section deformation modes of thin-walled sections (conventional GBT modes).

The section is treated as a plane frame of its walls, split into ``n_sub``
elements per wall (the same subdivision as the ``ThinWalledSectionAnalysis``).
Each frame node carries an in-plane displacement ``(dy, dz)``, an in-plane
rotation ``beta`` (right-handed about ``x``) and an axial warping ``u``. On an
element with tangent ``t`` and normal ``n = (-t_z, t_y)``:

- tangential displacement ``v`` is linear in ``s``, normal displacement ``w`` is a
  cubic Hermite in ``(n . d, beta)`` (``dw/ds = beta``), warping ``u`` is linear;
- the Vlasov hypotheses hold: no hoop membrane strain (``dv/ds = 0``) and no
  membrane shear (``du/ds + v = 0``).

A mode is a nodal vector satisfying these constraints. Along the beam a mode
``k`` with amplitude ``a_k(x)`` displaces the walls by

    u_x = u_k(s) a_k',   (u_y, u_z) = (v_k t + w_k n) a_k,

so its wall strains, in the order of the 5 x 5 GBT wall law
``[eps_xx, gamma_xs, kappa_xx, kappa_ss, kappa_xs]``, are

    a_k'':  [u_k, 0, -w_k, 0, 0]
    a_k' :  [0, u_k,s + v_k (= 0), 0, 0, -2 w_k,s]
    a_k  :  [0, 0, 0, -w_k,ss, 0]

The admissible space contains the rigid-section modes (axial, two translations
with their plane-section warping and, for open sections, the rotation with its
sectorial warping). Those are carried by the beam DOFs, so the distortional
modes are the admissible vectors orthogonal to them in the warping stiffness
``C``, ordered by the generalised eigenproblem ``B x = lambda C x`` (``B`` the
transverse wall-bending stiffness), lowest first.

Warping modes (shear-deformable warping)
----------------------------------------
In the GBT element all out-of-plane warping is carried by warping modes
``Omega_j(s)`` with their own amplitudes ``b_j(x)``: ``u_x += Omega_j b_j``, so
``eps_xx += Omega_j b_j'`` and ``gamma_xs += Omega_j,s b_j``. The first modes span
the Saint-Venant warping of the six beam strains (shear warping and torsional
warping), so the Saint-Venant solution is recovered away from the ends; further
shear-lag modes come from ``E x = mu A x`` (membrane shear vs axial stiffness) in
the complement. A support with ``b = 0`` keeps the section plane there, which is
what produces shear lag and the warping-restraint end effects. The modes use the
warping discretisation of the ``ThinWalledSectionAnalysis`` (``n_sub`` cubic
elements per wall).

Wall loads
----------
A surface load ``p = p_x e_x + p_s t + p_n n`` on a wall does work
``int p . u ds`` per unit length, which gives the generalised load on
``[u, v, w, theta_x, theta_y, theta_z, b, a, a']``.

Public API
----------
CrossSectionModes.
"""

from __future__ import annotations

from functools import cached_property
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh, null_space

from pre_processing.section_integration.thin_walled.section_analysis import ThinWalledSectionAnalysis
from pre_processing.section_integration.thin_walled.wall_stiffness import gbt_wall_stiffness

#: GBT wall strains (5 x 5 law, ``N_ss`` released)
GBT_WALL_STRAINS = ("eps_xx", "gamma_xs", "kappa_xx", "kappa_ss", "kappa_xs")
_BEAM_ROWS = [0, 1, 2, 4]  # the 4 beam wall strains inside the 5 GBT ones
_NODE_DOF = 4               # dy, dz, beta, u


def _hermite(xi: float, le: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cubic Hermite on ``s = xi le`` (xi in [0, 1]) for ``(w_i, w_i', w_j, w_j')``: values, d/ds, d2/ds2."""
    x = xi
    H = np.array([1 - 3 * x**2 + 2 * x**3, le * (x - 2 * x**2 + x**3), 3 * x**2 - 2 * x**3, le * (-x**2 + x**3)])
    dH = np.array([-6 * x + 6 * x**2, le * (1 - 4 * x + 3 * x**2), 6 * x - 6 * x**2, le * (-2 * x + 3 * x**2)]) / le
    d2H = np.array([-6 + 12 * x, le * (-4 + 6 * x), 6 - 12 * x, le * (-2 + 6 * x)]) / le**2
    return H, dH, d2H


class CrossSectionModes:
    """Conventional (Vlasov) cross-section modes and the GBT section matrices.

    Parameters
    ----------
    analysis : ThinWalledSectionAnalysis
        Saint-Venant analysis of the section. Its ``n_sub`` sets the frame
        subdivision (``n_sub > 1`` adds local plate-bending modes). Its hoop
        option must be ``"restrained"``: the modes carry the hoop bending.

    Attributes
    ----------
    n_rigid : int
        Rigid-section modes in the admissible space (4 open, 3 closed).
    n_modes : int
        Distortional modes available.
    eigenvalues : ndarray (n_modes,)
        ``B_kk / C_kk`` of each distortional mode, ascending.
    """

    def __init__(self, analysis: ThinWalledSectionAnalysis) -> None:
        if analysis.hoop != "restrained":
            raise ValueError("cross-section modes need hoop='restrained' (the modes carry the hoop bending)")
        self.analysis = analysis
        self.section = analysis.section
        self.n_sub = analysis.n_sub
        self._laws = {lid: gbt_wall_stiffness(lam).matrix for lid, lam in analysis.laminates.items()}
        self._build_frame()

    # ------------------------------------------------------------------ #
    # Frame discretisation
    # ------------------------------------------------------------------ #
    def _build_frame(self) -> None:
        sec = self.section
        node_of_point = {pid: i for i, pid in enumerate(sorted(sec.points))}
        xy = [np.asarray(sec.points[pid], dtype=float) for pid in sorted(sec.points)]
        self._elements: List[Dict] = []
        self._wall_elements: Dict[int, List[int]] = {}
        for w in sec.walls:
            a, _ = sec.endpoints(w)
            t, n = sec.tangent(w), sec.normal(w)
            le = sec.length(w) / self.n_sub
            nodes = [node_of_point[w.start]]
            for k in range(1, self.n_sub):
                xy.append(a + k * le * t)
                nodes.append(len(xy) - 1)
            nodes.append(node_of_point[w.end])
            self._wall_elements[w.wall_id] = []
            for k in range(self.n_sub):
                self._wall_elements[w.wall_id].append(len(self._elements))
                self._elements.append(dict(wall=w, i=nodes[k], j=nodes[k + 1], t=t, n=n, le=le, a=a + k * le * t))
        self.node_xy = np.array(xy)
        self.n_nodes = len(xy)
        self.n_frame_dof = _NODE_DOF * self.n_nodes

    def _element_dofs(self, e: Dict) -> np.ndarray:
        i, j = e["i"], e["j"]
        return np.r_[_NODE_DOF * i:_NODE_DOF * i + 4, _NODE_DOF * j:_NODE_DOF * j + 4]

    def _fields(self, e: Dict, xi: float) -> Dict[str, np.ndarray]:
        """Rows (8,) mapping the element's nodal vector to ``u, u_s, v, w, w_s, w_ss`` at ``xi`` in [0, 1]."""
        t, n, le = e["t"], e["n"], e["le"]
        H, dH, d2H = _hermite(xi, le)
        out = {k: np.zeros(8) for k in ("u", "u_s", "v", "w", "w_s", "w_ss")}
        for side, base, Lv in ((0, 0, 1.0 - xi), (1, 4, xi)):
            out["v"][base:base + 2] = Lv * t
            out["u"][base + 3] = Lv
            out["u_s"][base + 3] = (-1.0 if side == 0 else 1.0) / le
            for key, sh in (("w", H), ("w_s", dH), ("w_ss", d2H)):
                out[key][base:base + 2] += sh[2 * side] * n
                out[key][base + 2] += sh[2 * side + 1]
        return out

    # ------------------------------------------------------------------ #
    # Admissible space and modes
    # ------------------------------------------------------------------ #
    @cached_property
    def _constraints(self) -> npt.NDArray[np.float64]:
        G = np.zeros((2 * len(self._elements), self.n_frame_dof))
        for r, e in enumerate(self._elements):
            d = self._element_dofs(e)
            f0, f1 = self._fields(e, 0.0), self._fields(e, 1.0)
            G[2 * r, d] = f1["v"] - f0["v"]           # inextensible in s
            G[2 * r + 1, d] = f0["u_s"] + f0["v"]     # no membrane shear (v constant)
        return G

    @cached_property
    def _admissible(self) -> npt.NDArray[np.float64]:
        G = self._constraints
        return null_space(G, rcond=1e-10 * max(1.0, np.abs(G).max()))

    def _gauss(self):
        """Yield ``(element, fields, weight, laminate_id)`` on 5 Gauss points per frame element."""
        xg, wg = np.polynomial.legendre.leggauss(5)
        for e in self._elements:
            for x, w in zip(xg, wg):
                xi = 0.5 * (1.0 + x)
                yield e, self._fields(e, xi), 0.5 * w * e["le"], e["wall"].laminate_id

    @cached_property
    def _frame_matrices(self) -> Dict[str, npt.NDArray[np.float64]]:
        """``C`` (a'' energy), ``B`` (a energy) on the frame DOFs."""
        nf = self.n_frame_dof
        C = np.zeros((nf, nf))
        B = np.zeros((nf, nf))
        for e, f, wt, lid in self._gauss():
            K = self._laws[lid]
            d = self._element_dofs(e)
            E2 = np.vstack([f["u"], np.zeros(8), -f["w"], np.zeros(8), np.zeros(8)])
            E0 = np.vstack([np.zeros(8), np.zeros(8), np.zeros(8), -f["w_ss"], np.zeros(8)])
            C[np.ix_(d, d)] += wt * E2.T @ K @ E2
            B[np.ix_(d, d)] += wt * E0.T @ K @ E0
        return {"C": 0.5 * (C + C.T), "B": 0.5 * (B + B.T)}

    @cached_property
    def _rigid(self) -> npt.NDArray[np.float64]:
        """Admissible rigid-section modes (columns, frame DOFs): axial, y, z and, if admissible, rotation."""
        G = self._constraints
        u_cols = np.arange(3, self.n_frame_dof, _NODE_DOF)
        modes = []
        ax = np.zeros(self.n_frame_dof)
        ax[u_cols] = 1.0
        modes.append(ax)
        y, z = self.node_xy[:, 0], self.node_xy[:, 1]
        for dy, dz, beta in ((np.ones_like(y), 0 * y, 0 * y), (0 * y, np.ones_like(y), 0 * y), (-z, y, np.ones_like(y))):
            q = np.zeros(self.n_frame_dof)
            q[0::_NODE_DOF], q[1::_NODE_DOF], q[2::_NODE_DOF] = dy, dz, beta
            Gu = G[:, u_cols]
            u, *_ = np.linalg.lstsq(Gu, -G @ q, rcond=None)
            q[u_cols] = u
            if np.linalg.norm(G @ q) <= 1e-9 * max(1.0, np.linalg.norm(q)) * max(1.0, np.abs(G).max()):
                modes.append(q)
        return np.column_stack(modes)

    @property
    def n_rigid(self) -> int:
        return self._rigid.shape[1]

    @cached_property
    def _modes(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        N = self._admissible
        C, B = self._frame_matrices["C"], self._frame_matrices["B"]
        # coefficients of the rigid modes in the admissible basis
        R, *_ = np.linalg.lstsq(N, self._rigid, rcond=None)
        Cn, Bn = N.T @ C @ N, N.T @ B @ N
        # distortional space: C-orthogonal to the rigid modes
        P = null_space((Cn @ R).T)
        if P.shape[1] == 0:
            return np.zeros(0), np.zeros((self.n_frame_dof, 0))
        lam, Y = eigh(P.T @ Bn @ P, P.T @ Cn @ P)
        X = N @ P @ Y
        # one refinement pass on the mode basis sharpens the B- and C-orthogonality
        for _ in range(2):
            lam, Y = eigh(X.T @ B @ X, X.T @ C @ X)
            X = X @ Y
        X = np.column_stack([self._normalise(X[:, k]) for k in range(X.shape[1])])
        lam = np.einsum("ik,ij,jk->k", X, B, X) / np.einsum("ik,ij,jk->k", X, C, X)
        return lam, X

    def _normalise(self, q: np.ndarray) -> np.ndarray:
        """Scale to unit largest in-plane displacement; the largest frame component positive."""
        peak = 0.0
        for e, f, _, _ in self._gauss():
            d = self._element_dofs(e)
            peak = max(peak, abs(f["v"] @ q[d]), abs(f["w"] @ q[d]))
        q = q / peak
        k = np.argmax(np.abs(q))
        return q * np.sign(q[k])

    @property
    def eigenvalues(self) -> npt.NDArray[np.float64]:
        return self._modes[0].copy()

    @property
    def n_modes(self) -> int:
        return self._modes[1].shape[1]

    def mode(self, k: int) -> Dict[str, npt.NDArray[np.float64]]:
        """Mode ``k`` (0-based) at the frame nodes: ``dy, dz, beta, u`` (each (n_nodes,)) and ``xy``."""
        q = self._modes[1][:, k]
        return {"dy": q[0::4].copy(), "dz": q[1::4].copy(), "beta": q[2::4].copy(), "u": q[3::4].copy(),
                "xy": self.node_xy.copy()}

    def frame_matrices(self, n_modes: int) -> Dict[str, npt.NDArray[np.float64]]:
        """Classical GBT matrices of the first ``n_modes`` distortional modes: ``C`` (a''), ``D`` (a'), ``B`` (a)."""
        X = self._check(n_modes)
        m = X.shape[1]
        C = np.zeros((m, m))
        D = np.zeros((m, m))
        B = np.zeros((m, m))
        for e, f, wt, lid in self._gauss():
            K = self._laws[lid]
            q = X[self._element_dofs(e)]
            E2 = np.vstack([f["u"] @ q, 0 * f["u"] @ q, -f["w"] @ q, 0 * f["u"] @ q, 0 * f["u"] @ q])
            E1 = np.vstack([0 * f["u"] @ q, (f["u_s"] + f["v"]) @ q, 0 * f["u"] @ q, 0 * f["u"] @ q, -2 * f["w_s"] @ q])
            E0 = np.vstack([0 * f["u"] @ q, 0 * f["u"] @ q, 0 * f["u"] @ q, -f["w_ss"] @ q, 0 * f["u"] @ q])
            C += wt * E2.T @ K @ E2
            D += wt * E1.T @ K @ E1
            B += wt * E0.T @ K @ E0
        return {"C": C, "D": D, "B": B}

    def _check(self, n_modes: int) -> npt.NDArray[np.float64]:
        if n_modes < 0 or n_modes > self.n_modes:
            raise ValueError(f"asked for {n_modes} cross-section modes; the section has {self.n_modes}")
        return self._modes[1][:, :n_modes]

    # ------------------------------------------------------------------ #
    # Warping modes (shear-deformable warping)
    # ------------------------------------------------------------------ #
    @cached_property
    def _axial_warping_stiffness(self) -> npt.NDArray[np.float64]:
        """``A = int Nx_0^T K_00 Nx_0 ds`` on the warping DOFs of the Saint-Venant analysis."""
        an = self.analysis
        A = np.zeros((an.n_dof, an.n_dof))
        for g in an._gauss_points:
            A += g.weight * g.K[0, 0] * np.outer(g.Nx[0], g.Nx[0])
        return 0.5 * (A + A.T)

    @cached_property
    def _warping_basis(self) -> Tuple[npt.NDArray[np.float64], int, npt.NDArray[np.float64]]:
        """Warping modes on the Saint-Venant warping DOFs: ``(Omega, n_sv, mu)``.

        The first ``n_sv`` columns span the Saint-Venant warping of the six beam
        strains and their rates (``W0``, ``W1``), so uniform and linearly varying
        section forces are reproduced exactly. The rest are shear-lag modes from
        ``E x = mu A x`` in the complement, lowest first. All columns are
        ``A``-orthogonal to the rigid axial fields ``1, y, z`` and to each other.
        """
        an = self.analysis
        A = self._axial_warping_stiffness
        E = an._matrices["E"]
        R = an._rigid_modes
        sol = an._solution

        def deflate(W, Q):
            if Q.shape[1] == 0:
                return W
            return W - Q @ np.linalg.solve(Q.T @ A @ Q, Q.T @ A @ W)

        def orthonormal(W):
            # scale columns first: the force and rate fields differ by many orders
            nrm = np.sqrt(np.maximum(np.einsum("ik,ij,jk->k", W, A, W), 0.0))
            big = nrm > 1e-12 * max(nrm.max(), 1e-300)
            W = W[:, big] / nrm[big]
            G = W.T @ A @ W
            lam, V = np.linalg.eigh(0.5 * (G + G.T))
            keep = lam > 1e-10 * max(lam.max(), 1e-300)
            return W @ V[:, keep] / np.sqrt(lam[keep])

        Rr = orthonormal(R)
        Wsv = orthonormal(deflate(np.hstack([sol["W0"], sol["W1"]]), Rr))
        # order the Saint-Venant modes by their shear stiffness too
        mu_sv, Y = eigh(Wsv.T @ E @ Wsv, Wsv.T @ A @ Wsv)
        Wsv = Wsv @ Y
        Q = np.hstack([Rr, Wsv])
        P = null_space((A @ Q).T)
        if P.shape[1]:
            X = P
            for _ in range(3):
                mu_x, Yx = eigh(X.T @ E @ X, X.T @ A @ X)
                X = X @ Yx
        else:
            X, mu_x = np.zeros((an.n_dof, 0)), np.zeros(0)
        Om = np.hstack([Wsv, X])
        Om = Om / np.abs(Om).max(axis=0)
        Om = Om * np.sign(Om[np.argmax(np.abs(Om), axis=0), np.arange(Om.shape[1])])
        mu = np.einsum("ik,ij,jk->k", Om, E, Om) / np.einsum("ik,ij,jk->k", Om, A, Om)
        return Om, Wsv.shape[1], mu

    @property
    def n_sv_warping(self) -> int:
        """Warping modes always carried: the span of the Saint-Venant warping."""
        return self._warping_basis[1]

    @property
    def n_shear_modes(self) -> int:
        """Extra (shear-lag) warping modes available beyond the Saint-Venant ones."""
        return self._warping_basis[0].shape[1] - self._warping_basis[1]

    @property
    def warping_eigenvalues(self) -> npt.NDArray[np.float64]:
        """``E_jj / A_jj`` of each warping mode (1 / length^2): Saint-Venant ones, then shear-lag ones."""
        return self._warping_basis[2].copy()

    def warping_mode(self, j: int) -> npt.NDArray[np.float64]:
        """Warping mode ``j`` on the Saint-Venant warping DOFs (``analysis.point_dof`` for section points)."""
        return self._warping_basis[0][:, j].copy()

    def _check_warping(self, n_shear: int) -> npt.NDArray[np.float64]:
        if n_shear < 0 or n_shear > self.n_shear_modes:
            raise ValueError(f"asked for {n_shear} shear-lag modes; the section has {self.n_shear_modes}")
        return self._warping_basis[0][:, :self.n_sv_warping + n_shear]

    def n_warping(self, n_shear: int = 0) -> int:
        """Warping amplitudes per node for ``n_shear`` extra shear-lag modes."""
        return self.n_sv_warping + n_shear

    # ------------------------------------------------------------------ #
    # GBT section stiffness, mass and wall loads
    # ------------------------------------------------------------------ #
    def _locate(self, wall_id: int, position: np.ndarray) -> Tuple[Dict, float]:
        els = self._wall_elements[wall_id]
        e0 = self._elements[els[0]]
        s = float((position - e0["a"]) @ e0["t"])
        k = min(int(s / e0["le"]), self.n_sub - 1)
        e = self._elements[els[k]]
        return e, s / e["le"] - k

    def stiffness(self, n_modes: int, n_shear: int = 0) -> npt.NDArray[np.float64]:
        """Right-handed GBT section stiffness, square of size ``6 + 2p + 3m``.

        Strains ``[psi6, b'(p), b(p), a''(m), a'(m), a(m)]`` with ``p = n_warping(n_shear)``.
        The beam strains act with plane-section kinematics; all warping is carried by
        the amplitudes ``b`` of the warping modes, so a restrained end (``b = 0``) gives
        shear lag and warping-restraint effects. Wall strains::

            eps_xx   = Z_0 psi6 + Omega b' + u_k a_k''
            gamma_xs = Z_1 psi6 + Omega_s b
            kappa_xx = Z_2 psi6 - w_k a_k''
            kappa_ss = -w_k,ss a_k
            kappa_xs = -2 phi - 2 w_k,s a_k'
        """
        X = self._check(n_modes)
        Om = self._check_warping(n_shear)
        m, p = X.shape[1], Om.shape[1]
        n = 6 + 2 * p + 3 * m
        K = np.zeros((n, n))
        for g in self.analysis._gauss_points:
            H, e = self._strain_map(g, X, Om)
            K += g.weight * H.T @ self._laws[e["wall"].laminate_id] @ H
        return 0.5 * (K + K.T)

    def _strain_map(self, g, X, Om) -> Tuple[npt.NDArray[np.float64], Dict]:
        """``H`` (5, 6 + 2p + 3m): GBT wall strains per strain of ``stiffness`` at a section point."""
        m, p = X.shape[1], Om.shape[1]
        n = 6 + 2 * p + 3 * m
        o_a = 6 + 2 * p
        H = np.zeros((5, n))
        H[_BEAM_ROWS, :6] = g.Z
        H[0, 6:6 + p] = g.Nx[0] @ Om
        H[1, 6 + p:o_a] = g.B[1] @ Om
        e, xi = self._locate(g.wall_id, g.position)
        f = self._fields(e, xi)
        q = X[self._element_dofs(e)]
        H[0, o_a:o_a + m] = f["u"] @ q
        H[2, o_a:o_a + m] = -(f["w"] @ q)
        H[1, o_a + m:o_a + 2 * m] = (f["u_s"] + f["v"]) @ q
        H[4, o_a + m:o_a + 2 * m] = -2.0 * (f["w_s"] @ q)
        H[3, o_a + 2 * m:] = -(f["w_ss"] @ q)
        return H, e

    def wall_strain_map(self, wall_id: int, s: float, n_modes: int, n_shear: int = 0) -> npt.NDArray[np.float64]:
        """Right-handed GBT wall strains (5, ...) per strain of ``stiffness(n_modes, n_shear)`` at ``s`` on a wall.

        Rows ``[eps_xx, gamma_xs, kappa_xx, kappa_ss, kappa_xs]``.
        """
        g = self.analysis.point_at(wall_id, s)
        return self._strain_map(g, self._check(n_modes), self._check_warping(n_shear))[0]

    def _displacement_map(self, g, X, Om) -> Tuple[npt.NDArray[np.float64], Dict]:
        """``Pi`` (3, 6 + p + 2m): wall displacement per ``[u, v, w, theta, b(p), a(m), a'(m)]``."""
        m, p = X.shape[1], Om.shape[1]
        n = 6 + p + 2 * m
        y, z = g.position
        e, xi = self._locate(g.wall_id, g.position)
        f = self._fields(e, xi)
        q = X[self._element_dofs(e)]
        t, nn = e["t"], e["n"]
        v_k, w_k, u_k = f["v"] @ q, f["w"] @ q, f["u"] @ q
        Pi = np.zeros((3, n))
        Pi[0, [0, 4, 5]] = [1.0, z, -y]
        Pi[1, [1, 3]] = [1.0, -z]
        Pi[2, [2, 3]] = [1.0, y]
        Pi[0, 6:6 + p] = g.Nx[0] @ Om
        o = 6 + p
        Pi[1, o:o + m] = v_k * t[0] + w_k * nn[0]
        Pi[2, o:o + m] = v_k * t[1] + w_k * nn[1]
        Pi[0, o + m:] = u_k
        return Pi, e

    def mass(self, n_modes: int, n_shear: int = 0) -> npt.NDArray[np.float64]:
        """Section mass for ``[u, v, w, theta_x, theta_y, theta_z, b(p), a(m), a'(m)]``.

        Kinetic energy ``1/2 int rho_A |u_dot|^2 ds`` with the wall displacement
        ``u_x = u - y theta_z + z theta_y + Omega_j b_j + u_k a_k'`` and
        ``(u_y, u_z) = (v - z theta_x, w + y theta_x) + d_k a_k``.
        """
        X, Om = self._check(n_modes), self._check_warping(n_shear)
        n = 6 + Om.shape[1] + 2 * X.shape[1]
        M = np.zeros((n, n))
        for g in self.analysis._gauss_points:
            Pi, e = self._displacement_map(g, X, Om)
            rho = self.analysis.laminates[e["wall"].laminate_id].mass_per_area
            M += g.weight * rho * Pi.T @ Pi
        return 0.5 * (M + M.T)

    def wall_load(self, loads: Dict[int, Tuple[float, float, float]], n_modes: int, n_shear: int = 0) -> npt.NDArray[np.float64]:
        """Generalised load per length from uniform wall surface loads, in the order of ``mass``.

        ``loads`` maps ``wall_id`` to ``(p_x, p_s, p_n)`` (force per area along ``x``,
        the wall tangent and the wall normal).
        """
        unknown = sorted(set(loads) - {w.wall_id for w in self.section.walls})
        if unknown:
            raise ValueError(f"wall loads name unknown walls {unknown}")
        X, Om = self._check(n_modes), self._check_warping(n_shear)
        f = np.zeros(6 + Om.shape[1] + 2 * X.shape[1])
        for g in self.analysis._gauss_points:
            if g.wall_id not in loads:
                continue
            Pi, e = self._displacement_map(g, X, Om)
            px, ps, pn = loads[g.wall_id]
            f += g.weight * Pi.T @ np.array([px, ps * e["t"][0] + pn * e["n"][0], ps * e["t"][1] + pn * e["n"][1]])
        return f
