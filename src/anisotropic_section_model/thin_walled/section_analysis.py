# pre_processing/section_integration/thin_walled/section_analysis.py
"""Saint-Venant analysis of anisotropic thin-walled sections.

Builds the fully coupled 6 x 6 Timoshenko stiffness of a thin-walled section
from the laminates of its walls, by the Giavotto method applied to wall
mid-lines. Open, closed and multi-cell sections are handled by the same
equations: no shear-flow bookkeeping, cell loops or cut points are needed,
because the warping is solved for directly.

Beam kinematics (right-handed, rotations by the right-hand rule)
----------------------------------------------------------------
Section rigid in its own plane, plus out-of-plane warping::

    u_x(x, y, z) = u - y theta_z + z theta_y + warping
    u_y = v - z theta_x,   u_z = w + y theta_x

Generalised strains, in this order (``STRAINS``)::

    psi = [eps, kappa_y, kappa_z, gamma_xy, gamma_xz, phi]
    eps = u',  kappa_y = theta_y',  kappa_z = theta_z',
    gamma_xy = v' - theta_z,  gamma_xz = w' + theta_y,  phi = theta_x'

and the work-conjugate section forces (``FORCES``)::

    F = [N, M_y, M_z, V_y, V_z, T]
    N = int sigma,  M_y = int z sigma,  M_z = -int y sigma,
    V_y = int tau_xy,  V_z = int tau_xz,  T = int (y tau_xz - z tau_xy)

Beam equilibrium with no distributed load: ``N', V', T' = 0``,
``M_y' = V_z`` and ``M_z' = -V_y``.

Wall kinematics (Kirchhoff walls)
---------------------------------
Each wall carries an axial warping ``w(s)`` of its mid-surface. Through the
thickness the wall follows Kirchhoff plate kinematics (``gamma_xn = 0``). On a
straight wall with tangent ``t``, normal ``n = (-t_z, t_y)``,
``h = y t_z - z t_y`` and ``r_n = y n_z - z n_y``, the wall strains are::

    eps_xx   = eps + z kappa_y - y kappa_z                       + dw/dx
    gamma_xs = t_y gamma_xy + t_z gamma_xz + h phi               + dw/ds
    kappa_xx = n_z kappa_y - n_y kappa_z
               - (n_y gamma_xy' + n_z gamma_xz' + r_n phi')
    kappa_xs = -2 phi

or ``e_w = Z psi + Y psi' + B w_hat + N_x w_hat'`` in discrete form. The
``Y`` term is the wall bending caused by a varying transverse shear strain;
its reaction is the wall's Kirchhoff transverse shear force
``Q_x = dM_xx/dx``. The wall law is the condensed in-plane wall stiffness
(4 x 4, order ``WALL_STRAINS``).

Weak form (Giavotto, Borri et al. 1983; Blasques and Stolpe 2012)
----------------------------------------------------------------
For section forces that vary linearly along the beam, warping and strains are
``w_hat = w0 + x w1`` and ``psi = psi0 + x psi1``. With ``E = int B^T K B``,
``C = int B^T K N_x``, ``R = int B^T K Z``, ``L = int N_x^T K Z``,
``L_y = int B^T K Y``, ``A = int Z^T K Z``, ``Q = int Z^T K Y`` and ``D`` the
warping modes that equal rigid motions (``w = 1, y, z``)::

    [E   R  D] [w1  ]   [0     ]
    [R^T A  0] [psi1] = [T F0  ]
    [D^T 0  0] [lam1]   [0     ]

    [E   R  D] [w0  ]   [-(C - C^T) w1 + (L - L_y) psi1         ]
    [R^T A  0] [psi0] = [F0 - (L - L_y)^T w1 - (Q - Q^T) psi1   ]
    [D^T 0  0] [lam0]   [0                                      ]

``T`` maps forces to their rate along the beam (``M_y' = V_z``,
``M_z' = -V_y``). With ``e_w = Phi F0`` at ``x = 0``, the flexibility is
``int Phi^T K Phi ds`` and the stiffness is its inverse. The derivation is in
docs/proofs/section/saint_venant_thin_walled.md.

Discretisation: cubic Lagrange elements for ``w`` along each wall (``n_sub``
per wall), shared at section points. On straight walls the exact warping is
at most cubic, so one element per wall is exact.

Public API
----------
STRAINS, FORCES, ThinWalledSectionAnalysis.

See Also
--------
docs/element_library/thin_walled_section_analysis.md
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Mapping, Tuple

import numpy as np
import numpy.typing as npt

from pre_processing.section_integration.thin_walled.laminate import WallLaminate
from pre_processing.section_integration.thin_walled.thin_walled_geometry import (
    ThinWalledSection,
    Wall,
)
from pre_processing.section_integration.thin_walled.wall_stiffness import (
    condensed_wall_stiffness,
)

STRAINS: Tuple[str, ...] = ("eps", "kappa_y", "kappa_z", "gamma_xy", "gamma_xz", "phi")
FORCES: Tuple[str, ...] = ("N", "M_y", "M_z", "V_y", "V_z", "T")
CLASSICAL = (0, 1, 2, 5)  # [eps, kappa_y, kappa_z, phi]
WALL_STRAINS: Tuple[str, ...] = ("eps_xx", "gamma_xs", "kappa_xx", "kappa_xs")

HOOP_OPTIONS = ("restrained", "free")

# Force-rate map: F' = T_RATE @ F for a beam with no distributed load.
T_RATE = np.zeros((6, 6))
T_RATE[1, 4] = 1.0   # M_y' = V_z
T_RATE[2, 3] = -1.0  # M_z' = -V_y

_XI_NODES = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])
_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(5)


def _cubic_lagrange(xi: float) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Cubic Lagrange shape functions and d/dxi at ``xi`` (nodes -1, -1/3, 1/3, 1)."""
    N = np.ones(4)
    dN = np.zeros(4)
    for i in range(4):
        others = [j for j in range(4) if j != i]
        den = np.prod([_XI_NODES[i] - _XI_NODES[j] for j in others])
        N[i] = np.prod([xi - _XI_NODES[j] for j in others]) / den
        dN[i] = sum(
            np.prod([xi - _XI_NODES[j] for j in others if j != k]) for k in others
        ) / den
    return N, dN


def _solve(S: npt.NDArray[np.float64], rhs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Solve the section system; a strain with no wall strain (shear-rigid direction) takes the minimum-norm value."""
    try:
        return np.linalg.solve(S, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(S, rhs, rcond=None)[0]


def wall_law(laminate: WallLaminate, hoop: str = "restrained") -> npt.NDArray[np.float64]:
    """Wall stiffness (4, 4) in order ``WALL_STRAINS``.

    ``hoop="restrained"`` (default): the section keeps its shape, so the wall's
    hoop curvature is zero (``kappa_ss = 0``) while the hoop force is free
    (``N_ss = 0``). ``hoop="free"``: the hoop moment is free too (``M_ss = 0``),
    which is softer. For walls with ``B = 0`` the two differ only in the wall
    plate-bending terms (through ``D12`` and ``D26``); for unsymmetric walls the
    membrane terms differ too.
    """
    if hoop not in HOOP_OPTIONS:
        raise ValueError(f"hoop must be one of {HOOP_OPTIONS}, got {hoop!r}")
    if hoop == "restrained":
        ws = condensed_wall_stiffness(laminate, ("N_ss",))
        keep = [ws.index(n) for n in WALL_STRAINS]
        return ws.matrix[np.ix_(keep, keep)].copy()
    ws = condensed_wall_stiffness(laminate, ("N_ss", "M_ss"))
    assert ws.strains == WALL_STRAINS
    return ws.matrix.copy()


@dataclass(frozen=True)
class _GaussPoint:
    Z: npt.NDArray[np.float64]       # (4, 6)  strains from psi
    Y: npt.NDArray[np.float64]       # (4, 6)  strains from psi'
    B: npt.NDArray[np.float64]       # (4, ndof) strains from warping
    Nx: npt.NDArray[np.float64]      # (4, ndof) strains from warping'
    K: npt.NDArray[np.float64]       # (4, 4)
    weight: float                    # ds weight
    wall_id: int
    position: npt.NDArray[np.float64]
    normal: npt.NDArray[np.float64]
    thickness: float


class ThinWalledSectionAnalysis:
    """Saint-Venant stiffness and section properties of a thin-walled section.

    Parameters
    ----------
    section : ThinWalledSection
        Mid-line geometry. Coordinates ``(y, z)``; results are about its origin.
    laminates : mapping int -> WallLaminate
        Laminate of each ``laminate_id`` used by the walls.
    hoop : {"restrained", "free"}, optional
        Hoop behaviour of the walls; see ``wall_law``. Default ``"restrained"``.
    n_sub : int, optional
        Cubic elements per wall (default 1, exact for straight walls).

    Attributes
    ----------
    n_dof : int
        Warping unknowns (``w`` at the nodes of the wall mesh).
    point_dof : dict
        Warping unknown of each section point.

    Notes
    -----
    All results are in the order ``STRAINS`` / ``FORCES`` and about the
    section's coordinate origin. See the module docstring for the weak form.
    """

    def __init__(
        self,
        section: ThinWalledSection,
        laminates: Mapping[int, WallLaminate],
        hoop: str = "restrained",
        n_sub: int = 1,
    ) -> None:
        if n_sub < 1:
            raise ValueError(f"n_sub must be at least 1, got {n_sub}")
        missing = sorted({w.laminate_id for w in section.walls} - set(laminates))
        if missing:
            raise ValueError(f"no laminate given for laminate ids {missing}")
        self.section = section
        self.laminates = dict(laminates)
        self.hoop = hoop
        self.n_sub = int(n_sub)
        self._wall_laws = {
            lid: wall_law(self.laminates[lid], hoop) for lid in {w.laminate_id for w in section.walls}
        }
        self._build_dofs()

    # ------------------------------------------------------------------ #
    # Discretisation
    # ------------------------------------------------------------------ #
    def _build_dofs(self) -> None:
        sec = self.section
        point_dof: Dict[int, int] = {pid: i for i, pid in enumerate(sorted(sec.points))}
        n = len(point_dof)
        n_nodes = 3 * self.n_sub + 1
        self._wall_dofs: Dict[int, List[int]] = {}
        self._wall_node_s: Dict[int, npt.NDArray[np.float64]] = {}
        for w in sec.walls:
            wd = [point_dof[w.start]]
            for _ in range(n_nodes - 2):
                wd.append(n)
                n += 1
            wd.append(point_dof[w.end])
            self._wall_dofs[w.wall_id] = wd
            self._wall_node_s[w.wall_id] = np.linspace(0.0, sec.length(w), n_nodes)
        self.n_dof = n
        self.point_dof = point_dof

    def _point(self, w, k: int, xg: float, weight: float = 0.0) -> _GaussPoint:
        """Strain operators at natural coordinate ``xg`` of element ``k`` of wall ``w``."""
        sec = self.section
        a, _ = sec.endpoints(w)
        t = sec.tangent(w)
        nrm = sec.normal(w)
        le = sec.length(w) / self.n_sub
        dof = self._wall_dofs[w.wall_id][3 * k: 3 * k + 4]
        Nsh, dN = _cubic_lagrange(xg)
        s = le * (k + 0.5 * (1.0 + xg))
        p = a + s * t
        y, z = p
        h = y * t[1] - z * t[0]
        rn = y * nrm[1] - z * nrm[0]
        Z = np.zeros((4, 6))
        Z[0, 0], Z[0, 1], Z[0, 2] = 1.0, z, -y
        Z[1, 3], Z[1, 4], Z[1, 5] = t[0], t[1], h
        Z[2, 1], Z[2, 2] = nrm[1], -nrm[0]
        Z[3, 5] = -2.0
        Y = np.zeros((4, 6))
        Y[2, 3], Y[2, 4], Y[2, 5] = -nrm[0], -nrm[1], -rn
        B = np.zeros((4, self.n_dof))
        B[1, dof] += dN * 2.0 / le
        Nx = np.zeros((4, self.n_dof))
        Nx[0, dof] += Nsh
        return _GaussPoint(Z, Y, B, Nx, self._wall_laws[w.laminate_id], weight * 0.5 * le, w.wall_id, p, nrm,
                           self.laminates[w.laminate_id].thickness)

    @cached_property
    def _gauss_points(self) -> List[_GaussPoint]:
        out: List[_GaussPoint] = []
        for w in self.section.walls:
            for k in range(self.n_sub):
                for xg, wg in zip(_GAUSS_X, _GAUSS_W):
                    out.append(self._point(w, k, xg, wg))
        return out

    def point_at(self, wall_id: int, s: float) -> _GaussPoint:
        """Strain operators at distance ``s`` along wall ``wall_id`` (zero weight)."""
        w = self.section.wall(wall_id)
        L = self.section.length(w)
        if not -1e-12 * L <= s <= L * (1 + 1e-12):
            raise ValueError(f"s = {s} is outside wall {wall_id} (length {L})")
        le = L / self.n_sub
        k = min(max(int(s / le), 0), self.n_sub - 1)
        return self._point(w, k, 2.0 * (s / le - k) - 1.0)

    @cached_property
    def _rigid_modes(self) -> npt.NDArray[np.float64]:
        """Warping fields equal to rigid motions: ``w = 1``, ``w = y``, ``w = z`` at the nodes."""
        sec = self.section
        modes = np.zeros((self.n_dof, 3))
        for w in sec.walls:
            a, _ = sec.endpoints(w)
            t = sec.tangent(w)
            for j, s in enumerate(self._wall_node_s[w.wall_id]):
                p = a + s * t
                modes[self._wall_dofs[w.wall_id][j]] = [1.0, p[0], p[1]]
        return modes

    @cached_property
    def _matrices(self) -> Dict[str, npt.NDArray[np.float64]]:
        nd = self.n_dof
        E = np.zeros((nd, nd))
        C = np.zeros((nd, nd))
        R = np.zeros((nd, 6))
        Lm = np.zeros((nd, 6))
        Ly = np.zeros((nd, 6))
        A = np.zeros((6, 6))
        Q = np.zeros((6, 6))
        M2 = np.zeros((nd, nd))
        for g in self._gauss_points:
            KB = g.K @ g.B
            KZ = g.K @ g.Z
            KY = g.K @ g.Y
            E += g.weight * g.B.T @ KB
            C += g.weight * KB.T @ g.Nx
            R += g.weight * g.B.T @ KZ
            Lm += g.weight * g.Nx.T @ KZ
            Ly += g.weight * g.B.T @ KY
            A += g.weight * g.Z.T @ KZ
            Q += g.weight * g.Z.T @ KY
            M2 += g.weight * g.thickness * np.outer(g.Nx[0], g.Nx[0])
        D = M2 @ self._rigid_modes
        # Collinear walls make the modes dependent (e.g. w = z vanishes on a strip at z = 0).
        U, sv, _ = np.linalg.svd(D, full_matrices=False)
        D = U[:, sv > 1e-12 * sv[0]]
        return {"E": E, "C": C, "R": R, "L": Lm, "Ly": Ly, "A": A, "Q": Q, "D": D}

    @cached_property
    def _system(self) -> npt.NDArray[np.float64]:
        m = self._matrices
        nd = self.n_dof
        nc = m["D"].shape[1]
        S = np.zeros((nd + 6 + nc, nd + 6 + nc))
        S[:nd, :nd] = m["E"]
        S[:nd, nd:nd + 6] = m["R"]
        S[nd:nd + 6, :nd] = m["R"].T
        S[nd:nd + 6, nd:nd + 6] = m["A"]
        S[:nd, nd + 6:] = m["D"]
        S[nd + 6:, :nd] = m["D"].T
        return S

    @cached_property
    def _solution(self) -> Dict[str, npt.NDArray[np.float64]]:
        m = self._matrices
        nd = self.n_dof
        C, Lm, Ly, Q = m["C"], m["L"], m["Ly"], m["Q"]
        S = self._system

        n_tot = S.shape[0]
        rhs1 = np.zeros((n_tot, 6))
        rhs1[nd:nd + 6] = T_RATE
        X1 = _solve(S, rhs1)
        W1, P1 = X1[:nd], X1[nd:nd + 6]

        rhs0 = np.zeros((n_tot, 6))
        rhs0[:nd] = -(C - C.T) @ W1 + (Lm - Ly) @ P1
        rhs0[nd:nd + 6] = np.eye(6) - (Lm - Ly).T @ W1 - (Q - Q.T) @ P1
        X0 = _solve(S, rhs0)
        W0, P0 = X0[:nd], X0[nd:nd + 6]

        flex = np.zeros((6, 6))
        for g in self._gauss_points:
            Phi = g.Z @ P0 + g.Y @ P1 + g.B @ W0 + g.Nx @ W1
            flex += g.weight * Phi.T @ g.K @ Phi
        flex = 0.5 * (flex + flex.T)
        return {"W0": W0, "W1": W1, "P0": P0, "P1": P1, "flexibility": flex}

    # ------------------------------------------------------------------ #
    # Stiffness
    # ------------------------------------------------------------------ #
    @property
    def flexibility(self) -> npt.NDArray[np.float64]:
        """Section compliance (6, 6): ``psi = flexibility @ F`` about the origin."""
        return self._solution["flexibility"].copy()

    @cached_property
    def _stiffness(self) -> npt.NDArray[np.float64]:
        F = self._solution["flexibility"]
        ev = np.linalg.eigvalsh(F)
        if ev[0] <= 1e-13 * ev[-1]:
            raise ValueError(
                f"section {self.section.section_id}: the 6 x 6 stiffness is unbounded because the "
                "walls cannot shear in some direction (for example all walls are collinear, so "
                "Kirchhoff walls carry no transverse shear across their thickness). "
                "classical_stiffness is still available."
            )
        K = np.linalg.inv(F)
        return 0.5 * (K + K.T)

    @property
    def stiffness(self) -> npt.NDArray[np.float64]:
        """Timoshenko section stiffness (6, 6): ``F = stiffness @ psi``, order ``STRAINS``."""
        return self._stiffness.copy()

    @property
    def classical_stiffness(self) -> npt.NDArray[np.float64]:
        """Shear-rigid stiffness (4, 4) for ``[eps, kappa_y, kappa_z, phi]``.

        Euler-Bernoulli bending with Saint-Venant torsion: the inverse of the
        flexibility block of ``[N, M_y, M_z, T]``.
        """
        F4 = self._solution["flexibility"][np.ix_(CLASSICAL, CLASSICAL)]
        K = np.linalg.inv(F4)
        return 0.5 * (K + K.T)

    def strain_response(self, forces: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Generalised strains (6,) for section forces (6,) in order ``FORCES``."""
        return self._solution["flexibility"] @ np.asarray(forces, dtype=np.float64)

    def wall_strains(self, forces: npt.ArrayLike) -> List[Tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Wall strains at the Gauss points for section forces ``forces`` (constant shear).

        Returns a list of ``(wall_id, position (2,), strains (4,))`` in order
        ``WALL_STRAINS``.
        """
        sol = self._solution
        f = np.asarray(forces, dtype=np.float64)
        p0, p1 = sol["P0"] @ f, sol["P1"] @ f
        w0, w1 = sol["W0"] @ f, sol["W1"] @ f
        return [
            (g.wall_id, g.position.copy(), g.Z @ p0 + g.Y @ p1 + g.B @ w0 + g.Nx @ w1)
            for g in self._gauss_points
        ]

    def wall_strains_at(self, wall_id: int, s: float, forces: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Wall strains (4,) in order ``WALL_STRAINS`` at distance ``s`` along a wall.

        The section forces vary along the beam as with no distributed load
        (``F' = T F``), as in ``wall_strains``.
        """
        sol = self._solution
        f = np.asarray(forces, dtype=np.float64)
        g = self.point_at(wall_id, s)
        return g.Z @ (sol["P0"] @ f) + g.Y @ (sol["P1"] @ f) + g.B @ (sol["W0"] @ f) + g.Nx @ (sol["W1"] @ f)

    # ------------------------------------------------------------------ #
    # Centres
    # ------------------------------------------------------------------ #
    @property
    def shear_centre(self) -> Tuple[float, float]:
        """Point ``(y, z)`` where a transverse force causes no twist.

        A force ``(V_y, V_z)`` at ``(y, z)`` carries torque ``T = y V_z - z V_y``.
        """
        F = self._solution["flexibility"]
        y_s = -F[5, 4] / F[5, 5]
        z_s = F[5, 3] / F[5, 5]
        return float(y_s), float(z_s)

    @property
    def tension_centre(self) -> Tuple[float, float]:
        """Point ``(y, z)`` where an axial force causes no bending curvature.

        An axial force ``N`` at ``(y, z)`` gives ``M_y = z N`` and ``M_z = -y N``.
        """
        F = self._solution["flexibility"]
        # kappa_y, kappa_z rows (1, 2): F[i,0] + F[i,1] z - F[i,2] y = 0
        M = np.array([[-F[1, 2], F[1, 1]], [-F[2, 2], F[2, 1]]])
        rhs = -np.array([F[1, 0], F[2, 0]])
        y_t, z_t = np.linalg.solve(M, rhs)
        return float(y_t), float(z_t)

    # ------------------------------------------------------------------ #
    # Torsional warping and the Vlasov warping constant
    # ------------------------------------------------------------------ #
    @cached_property
    def _torsion_warping(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Warping per unit twist rate and its rigid-mode coefficients.

        ``omega`` solves the warping equation with ``psi = e_phi``. The axial
        strain of a twist-rate gradient is then ``eps_xx = omega``,
        ``kappa_xx = -r_n`` (Kirchhoff secondary warping). The parts equal to
        extension and bending are removed in the axial energy, which puts the
        pole at the shear centre and the constant at the tension centre.
        """
        m = self._matrices
        nd = self.n_dof
        nc = m["D"].shape[1]
        S = np.zeros((nd + nc, nd + nc))
        S[:nd, :nd] = m["E"]
        S[:nd, nd:] = m["D"]
        S[nd:, :nd] = m["D"].T
        rhs = np.zeros(nd + nc)
        rhs[:nd] = -m["R"][:, 5]
        omega = np.linalg.solve(S, rhs)[:nd]
        # rigid axial fields [eps_xx, kappa_xx] for eps, kappa_y, kappa_z: [1,0], [z,n_z], [-y,-n_y]
        G = np.zeros((3, 3))
        b = np.zeros(3)
        for g in self._gauss_points:
            Kax = g.K[np.ix_([0, 2], [0, 2])]
            y, z = g.position
            Mg = np.array([[1.0, z, -y], [0.0, g.normal[1], -g.normal[0]]])
            rn = y * g.normal[1] - z * g.normal[0]
            ag = np.array([g.Nx[0] @ omega, -rn])
            G += g.weight * Mg.T @ Kax @ Mg
            b += g.weight * Mg.T @ Kax @ ag
        c = np.linalg.solve(G, b)
        return omega, c

    def _warping_axial_field(self, g: _GaussPoint) -> npt.NDArray[np.float64]:
        omega, c = self._torsion_warping
        y, z = g.position
        Mg = np.array([[1.0, z, -y], [0.0, g.normal[1], -g.normal[0]]])
        rn = y * g.normal[1] - z * g.normal[0]
        return np.array([g.Nx[0] @ omega, -rn]) - Mg @ c

    def omega_at(self, wall_id: int, s: float) -> float:
        """Torsional warping ``omega`` (as ``point_warping``) at distance ``s`` along a wall."""
        omega, c = self._torsion_warping
        w = self.section.wall(wall_id)
        a, _ = self.section.endpoints(w)
        t = self.section.tangent(w)
        le = self.section.length(w) / self.n_sub
        k = min(max(int(s / le), 0), self.n_sub - 1)
        xi = 2.0 * (s / le - k) - 1.0
        Nsh, _ = _cubic_lagrange(xi)
        y, z = a + s * t
        return float(Nsh @ omega[self._wall_dofs[wall_id][3 * k:3 * k + 4]] - (c[0] + c[1] * z - c[2] * y))

    def point_warping(self) -> Dict[int, float]:
        """Torsional warping ``omega`` at each section point (per unit twist rate).

        Taken about the shear centre and orthogonal to extension and bending.
        """
        omega, c = self._torsion_warping
        out = {}
        for pid, d in self.point_dof.items():
            y, z = self.section.points[pid]
            out[pid] = float(omega[d] - (c[0] + c[1] * z - c[2] * y))
        return out

    @property
    def warping_constant(self) -> float:
        """Vlasov warping stiffness ``E Gamma`` (primary plus secondary warping)."""
        total = 0.0
        for g in self._gauss_points:
            a = self._warping_axial_field(g)
            total += g.weight * a @ g.K[np.ix_([0, 2], [0, 2])] @ a
        return float(total)

    # ------------------------------------------------------------------ #
    # Benscoter (restrained warping with an independent warping amplitude)
    # ------------------------------------------------------------------ #
    def _omega_at(self, g: _GaussPoint) -> Tuple[float, float]:
        """Torsional warping ``omega`` (about the shear centre, orthogonalised) and ``d omega / ds`` at a Gauss point."""
        omega, c = self._torsion_warping
        y, z = g.position
        t_y, t_z = g.normal[1], -g.normal[0]
        om = float(g.Nx[0] @ omega) - (c[0] + c[1] * z - c[2] * y)
        om_s = float(g.B[1] @ omega) - (c[1] * t_z - c[2] * t_y)
        return om, om_s

    def _benscoter_wall_strains(self, g: _GaussPoint) -> npt.NDArray[np.float64]:
        """Wall strains (4, 8) at a Gauss point per unit Benscoter strain ``[psi6, d, psi']``."""
        sol = self._solution
        Phi = g.Z @ sol["P0"] + g.Y @ sol["P1"] + g.B @ sol["W0"] + g.Nx @ sol["W1"]
        om, om_s = self._omega_at(g)
        G = np.zeros((4, 8))
        G[:, :6] = Phi @ self._stiffness
        G[1, 6] = om_s   # warping deviation d = psi - phi shears the walls
        G[0, 7] = om     # warping amplitude gradient psi' stretches them
        return G

    @cached_property
    def _benscoter(self) -> Tuple[npt.NDArray[np.float64], float]:
        K6 = self._stiffness
        K8 = np.zeros((8, 8))
        inertia = 0.0
        for g in self._gauss_points:
            G = self._benscoter_wall_strains(g)
            om, _ = self._omega_at(g)
            K8 += g.weight * G.T @ g.K @ G
            lam = self.laminates[self.section.wall(g.wall_id).laminate_id]
            inertia += g.weight * lam.mass_per_area * om * om
        K8 = 0.5 * (K8 + K8.T)
        K8[:6, :6] = K6
        return K8, float(inertia)

    @property
    def benscoter_stiffness(self) -> npt.NDArray[np.float64]:
        """Benscoter section stiffness (8, 8), right-handed, about the origin.

        Strains ``[eps, kappa_y, kappa_z, gamma_xy, gamma_xz, phi, d, psi']`` where the
        warping is the Saint-Venant warping of ``psi[:6]`` plus ``omega * d`` with
        ``d = psi - phi`` (``psi`` the warping amplitude, ``omega`` the torsional warping
        about the shear centre). Then ``eps_xx += omega psi'`` and ``gamma_xs += omega_s d``.
        The leading 6 x 6 block is ``stiffness``; ``K[6, 6]`` is the warping shear stiffness
        (``G (I_p - J)``-like) and ``K[7, 7]`` the primary ``E Gamma``. Secondary
        (through-thickness) warping is not included.
        """
        return self._benscoter[0].copy()

    # ------------------------------------------------------------------ #
    # Third-order shear (Reddy, Levinson): shear warping slaved to the shear strain
    # ------------------------------------------------------------------ #
    @cached_property
    def _shear_warping(self) -> npt.NDArray[np.float64]:
        """Saint-Venant warping (n_dof, 2) per unit ``[gamma_xy, gamma_xz]``.

        ``W0 K[:, 3:5]`` with its rigid part (``1, y, z``) removed in the walls' axial
        stiffness, so the axial stress of the slaved warping has no resultant ``N``,
        ``M_y`` or ``M_z`` (the plane-section rotation stays the Timoshenko one).
        """
        Psi = self._solution["W0"] @ self._stiffness[:, 3:5]
        Rm = self._rigid_modes
        M = np.zeros((self.n_dof, self.n_dof))
        for g in self._gauss_points:
            M += g.weight * g.K[0, 0] * np.outer(g.Nx[0], g.Nx[0])
        A = Rm.T @ M @ Rm
        c = np.linalg.lstsq(A, Rm.T @ M @ Psi, rcond=1e-12)[0]
        return Psi - Rm @ c

    def _third_order_wall_strains(self, g: _GaussPoint) -> npt.NDArray[np.float64]:
        """Wall strains (4, 8) at a Gauss point per unit third-order strain ``[psi6, chi_y, chi_z]``."""
        sol = self._solution
        Phi = g.Z @ sol["P0"] + g.Y @ sol["P1"] + g.B @ sol["W0"] + g.Nx @ sol["W1"]
        G = np.zeros((4, 8))
        G[:, :6] = Phi @ self._stiffness
        G[:, 6:] = g.Nx @ self._shear_warping  # eps_xx = Psi(s) . chi
        return G

    @cached_property
    def _third_order(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        K8 = np.zeros((8, 8))
        for g in self._gauss_points:
            G = self._third_order_wall_strains(g)
            K8 += g.weight * G.T @ g.K @ G
        K8 = 0.5 * (K8 + K8.T)
        K8[:6, :6] = self._stiffness
        F = self._solution["flexibility"]
        R = (F @ T_RATE @ self._stiffness)[3:5, :]
        return K8, R

    @property
    def third_order_stiffness(self) -> npt.NDArray[np.float64]:
        """Third-order shear section stiffness (8, 8), right-handed, about the origin.

        Strains ``[eps, kappa_y, kappa_z, gamma_xy, gamma_xz, phi, chi_y, chi_z]``. The
        section warps with the Saint-Venant warping of the six beam strains, and the
        shear part of that warping, ``Psi(s) gamma`` with ``Psi = W0 K[:, 3:5]``, is
        slaved to the local shear strain. Where the shear strain changes along the
        beam faster than in the Saint-Venant solution it stretches the walls:
        ``eps_xx += Psi(s) . chi`` with ``chi = gamma' - third_order_rate @ psi6``.
        For a solid rectangle ``Psi`` is Reddy's cubic ``z - 4 z^3 / 3h^2`` field and
        ``K[7, 7] = E I_y / 84``. The leading 6 x 6 block is ``stiffness``.
        """
        return self._third_order[0].copy()

    def third_order_wall_strains_at(self, wall_id: int, s: float) -> npt.NDArray[np.float64]:
        """Wall strains (4, 8) at ``s`` along a wall per unit right-handed third-order strain."""
        return self._third_order_wall_strains(self.point_at(wall_id, s))

    @property
    def third_order_rate(self) -> npt.NDArray[np.float64]:
        """``R`` (2, 6): the shear-strain rate of the Saint-Venant solution, ``gamma'_SV = R psi6``."""
        return self._third_order[1].copy()

    @property
    def warping_inertia(self) -> float:
        """Mass moment of the warping amplitude, ``int rho_A omega^2 ds`` per length."""
        return self._benscoter[1]

    def wall_load_resultant(self, loads: Mapping[int, Tuple[float, float, float]]) -> npt.NDArray[np.float64]:
        """Generalised load per length (7,) on ``[u, v, w, theta_x, theta_y, theta_z, psi]``.

        ``loads`` maps ``wall_id`` to a uniform surface load ``(p_x, p_s, p_n)`` along
        ``x``, the wall tangent and the wall normal. The load is the work-conjugate of
        the rigid-section displacement plus the torsional warping ``omega psi``, so its
        first six entries are ``[N, F_y, F_z, M_x, M_y, M_z]`` per length about the origin.
        """
        unknown = sorted(set(loads) - {w.wall_id for w in self.section.walls})
        if unknown:
            raise ValueError(f"wall loads name unknown walls {unknown}")
        f = np.zeros(7)
        for g in self._gauss_points:
            if g.wall_id not in loads:
                continue
            px, ps, pn = loads[g.wall_id]
            n_y, n_z = g.normal
            t_y, t_z = n_z, -n_y
            py, pz = ps * t_y + pn * n_y, ps * t_z + pn * n_z
            y, z = g.position
            om, _ = self._omega_at(g)
            f += g.weight * np.array([px, py, pz, y * pz - z * py, z * px, -y * px, om * px])
        return f

    # ------------------------------------------------------------------ #
    # Mass
    # ------------------------------------------------------------------ #
    @cached_property
    def mass_properties(self) -> Dict[str, float]:
        """Mass per length, mass centre and mass moments of inertia per length.

        Through-thickness distribution is included exactly for straight walls.
        Keys: ``m``, ``y_m``, ``z_m``, ``I_yy`` (int rho z^2), ``I_zz`` (int rho y^2),
        ``I_yz`` (int rho y z), all about the origin, and ``I_p = I_yy + I_zz``.
        """
        sec = self.section
        tot = dict(m=0.0, my=0.0, mz=0.0, yy=0.0, zz=0.0, yz=0.0)
        for w in sec.walls:
            lam = self.laminates[w.laminate_id]
            n_if = lam.interfaces
            rho = [p.material.rho for p in lam.plies]
            m0 = sum(r * (n_if[k + 1] - n_if[k]) for k, r in enumerate(rho))
            m1 = sum(r * (n_if[k + 1] ** 2 - n_if[k] ** 2) / 2 for k, r in enumerate(rho))
            m2 = sum(r * (n_if[k + 1] ** 3 - n_if[k] ** 3) / 3 for k, r in enumerate(rho))
            a, b = sec.endpoints(w)
            L = sec.length(w)
            ny, nz = sec.normal(w)
            iy = L * (a[0] + b[0]) / 2
            iz = L * (a[1] + b[1]) / 2
            iyy = L * (a[0] ** 2 + a[0] * b[0] + b[0] ** 2) / 3
            izz = L * (a[1] ** 2 + a[1] * b[1] + b[1] ** 2) / 3
            iyz = L * (2 * a[0] * a[1] + a[0] * b[1] + b[0] * a[1] + 2 * b[0] * b[1]) / 6
            tot["m"] += m0 * L
            tot["my"] += m0 * iy + m1 * ny * L
            tot["mz"] += m0 * iz + m1 * nz * L
            tot["zz"] += m0 * iyy + 2 * m1 * ny * iy + m2 * ny * ny * L
            tot["yy"] += m0 * izz + 2 * m1 * nz * iz + m2 * nz * nz * L
            tot["yz"] += m0 * iyz + m1 * (ny * iz + nz * iy) + m2 * ny * nz * L
        m = tot["m"]
        return {
            "m": m,
            "y_m": tot["my"] / m if m > 0 else 0.0,
            "z_m": tot["mz"] / m if m > 0 else 0.0,
            "I_yy": tot["yy"],
            "I_zz": tot["zz"],
            "I_yz": tot["yz"],
            "I_p": tot["yy"] + tot["zz"],
        }


def translation_matrix(dy: float, dz: float) -> npt.NDArray[np.float64]:
    """Strain map ``psi_new = P @ psi`` when the reference point moves by ``(dy, dz)``.

    A section described about a point at ``(dy, dz)`` (old coordinates) has
    ``eps_new = eps + dz kappa_y - dy kappa_z``, ``gamma_xy_new = gamma_xy - dz phi`` and
    ``gamma_xz_new = gamma_xz + dy phi``; curvatures and twist are unchanged. Then
    ``K_old = P.T @ K_new @ P``.
    """
    P = np.eye(6)
    P[0, 1] = dz
    P[0, 2] = -dy
    P[3, 5] = -dz
    P[4, 5] = dy
    return P


def stiffness_about(K: npt.ArrayLike, y0: float, z0: float) -> npt.NDArray[np.float64]:
    """Stiffness (6, 6) about the point ``(y0, z0)`` from a stiffness about the origin."""
    P = translation_matrix(y0, z0)
    Pinv = np.linalg.inv(P)
    return Pinv.T @ np.asarray(K, dtype=np.float64) @ Pinv
