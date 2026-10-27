"""Generalised Beam Theory (GBT) for thin-walled blade sections: warping and distortion.

Saint-Venant and Benscoter/Vlasov torsion keep the cross-section rigid in its own
plane. A thin-walled section is not rigid: a torque introduced as forces on the skins
or webs rather than as the Saint-Venant shear flows also distorts it (a box turns
into a rhombus, an airfoil's cells shear against each other), and the walls bend
across the contour to resist. Generalised Beam Theory adds those cross-section
deformation modes as further beam degrees of freedom, each with its own amplitude
along the span.

``GeneralizedBeam`` builds the beam from a ``ThinWalledSection``:

Kinematics. Each section point carries an axial warping displacement w and an
in-plane displacement (u_x, u_y) with an in-plane rotation r. In each straight wall
w varies linearly, the tangential displacement is constant (the walls do not stretch
round the contour, GBT's first Vlasov hypothesis) and the normal displacement is a
cubic (Hermite) curve fixed by the end displacements and rotations. The wall strains
are those of a Kirchhoff plate strip:

    eps_zz = dw/dz,  gamma_zs = dw/ds + dv/dz,
    kappa_zz = -d2u_n/dz2, kappa_ss = -d2u_n/ds2, kappa_zs = -2 d2u_n/ds dz

with the full 5 x 5 wall stiffness of ``condensed_wall_stiffness`` (no hoop stress),
so unbalanced and unsymmetric walls keep all their couplings. Membrane shear is kept
(GBT with shear modes), which closed sections need: their torsion and torsional
warping live in the shear modes.

Modes. The in-plane displacement fields that keep every wall inextensible are split
into the three rigid-body modes (translations in x and y, and rotation about the
shear centre, whose amplitude is the twist) and the distortional modes. These are
made free of work with the Saint-Venant shear flows (so a torque applied as those
shear flows twists the section without distorting it) and are ordered by the
transverse bending stiffness of the walls, from the softest (global distortion of the
cells) to local plate modes. Warping is either the four classical modes (axial,
two bending, and the torsional warping function omega), the classical modes plus a
number of further warping modes (shear lag), or every section point free.

Choosing the modes gives the hierarchy of torsion theories:

    warping="classical", distortional_modes=0   Benscoter: restrained warping, bimoment,
                                                Timoshenko bending, all couplings
    warping="full"                              + shear lag / secondary warping
    distortional_modes=n                        + n distortional modes (GBT)

Along the span, the in-plane amplitudes are cubic Hermite and the warping amplitudes
quadratic, so the beam is free of shear locking in the Vlasov limit. Ends can be
clamped (no displacement, no warping, walls clamped against bending), held by a
diaphragm (rib: no distortion, warping free) or have warping restrained; loads are
torques (as Saint-Venant shear flows), point forces at section points and loads
distributed along walls.

The section must be prismatic. Linear static analysis only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # optional sparse solver
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla
except ImportError:  # pragma: no cover - dense fallback
    _sp = None
    _spla = None

from .laminate_kinematics import integrate_abd
from .laminate_mechanics import Laminate
from .laminate_thin_wall import ThinWalledSection

_GAUSS_S = np.polynomial.legendre.leggauss(4)


def _null_space(A: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    if A.size == 0:
        return np.eye(A.shape[1])
    _, s, vt = np.linalg.svd(A)
    tol = rtol * (s[0] if s.size else 1.0)
    rank = int(np.sum(s > tol))
    return vt[rank:].T


def _hermite(xi: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cubic Hermite functions on [0, 1] (value, slope x length, value, slope x length) and derivatives."""
    H = np.array([1 - 3 * xi**2 + 2 * xi**3, xi - 2 * xi**2 + xi**3, 3 * xi**2 - 2 * xi**3, -xi**2 + xi**3])
    dH = np.array([-6 * xi + 6 * xi**2, 1 - 4 * xi + 3 * xi**2, 6 * xi - 6 * xi**2, -2 * xi + 3 * xi**2])
    ddH = np.array([-6 + 12 * xi, -4 + 6 * xi, 6 - 12 * xi, -2 + 6 * xi])
    return H, dH, ddH


@dataclass
class _WallPoint:
    """Strain operators at one integration point of a wall (rows over the 4N section DOFs)."""

    wall: int
    weight: float          # Gauss weight x wall length
    E0: np.ndarray         # (5, 4N) on d
    E1: np.ndarray         # on d'
    E2: np.ndarray         # on d''
    v: np.ndarray          # tangential displacement row
    un: np.ndarray         # normal displacement row
    w: np.ndarray          # warping row
    s_fraction: float


class SectionModes:
    """The cross-section deformation modes of a ``ThinWalledSection`` for GBT.

    ``in_plane`` is (3N, n_u): columns over [u_x, u_y, r] of every section point;
    columns 0-2 are the rigid modes (x, y, rotation about the shear centre) and the
    rest distortional modes, softest first, with their transverse bending stiffness
    ``distortional_stiffness`` (eigenvalue: bending energy per unit displacement
    squared). ``warping`` is (N, n_w), the axial displacement of every point.
    """

    def __init__(self, section: ThinWalledSection, warping: Union[str, int] = "full",
                 distortional_modes: Optional[int] = None, extensible: bool = False):
        self.section = section
        self.extensible = bool(extensible)
        N = len(section.points)
        self.n_points = N
        if self.extensible:  # full ABD, hoop strain kept: [eps_zz, eps_ss, gamma_zs, kappa_zz, kappa_ss, kappa_zs]
            self.strain_rows = [0, 1, 2, 3, 4, 5]
            self.wall_stiffness = np.array([integrate_abd(Laminate(w.laminate.plies)) for w in section.walls])
        else:  # no hoop stress, walls inextensible round the contour
            self.strain_rows = [0, 2, 3, 4, 5]
            self.wall_stiffness = section.stiffness
        self.points_ = self._wall_points(section, self.extensible, self.strain_rows)
        nd = 4 * N
        # in-plane metric M (displacement squared), transverse bending B, warping C and shear G
        M = np.zeros((nd, nd))
        B = np.zeros((nd, nd))
        Cw = np.zeros((nd, nd))
        Gw = np.zeros((nd, nd))
        f_rot = np.zeros(nd)
        r = {k: self.strain_rows.index(k) for k in self.strain_rows}
        for p in self.points_:
            K = self.wall_stiffness[p.wall]
            M += p.weight * (np.outer(p.v, p.v) + np.outer(p.un, p.un))
            kss = p.E0[r[4]]
            B += p.weight * K[r[4], r[4]] * np.outer(kss, kss)
            if self.extensible:
                ess = p.E0[r[1]]
                B += p.weight * K[r[1], r[1]] * np.outer(ess, ess)
            Cw += p.weight * section.Et[p.wall] * np.outer(p.w, p.w)
            ws = p.E0[r[2]]
            Gw += p.weight * section.Gt[p.wall] * np.outer(ws, ws)
            f_rot += p.weight * section.shear_flow[p.wall] * p.v
        u = slice(N, nd)
        M, B, f_rot = M[u, u], B[u, u], f_rot[u]
        Cw, Gw = Cw[:N, :N], Gw[:N, :N]

        # inextensible walls: (U_j - U_i) . t = 0
        G = np.zeros((len(section.walls), 3 * N))
        for k, (i, j) in enumerate(zip(section.start, section.end)):
            t = section.tangent[k]
            G[k, 3 * j:3 * j + 2] += t
            G[k, 3 * i:3 * i + 2] -= t
        Phi = np.eye(3 * N) if self.extensible else _null_space(G)
        x = section.points[:, 0]
        y = section.points[:, 1]
        xs, ys = section.shear_centre
        R = np.zeros((3 * N, 3))
        R[0::3, 0] = 1.0
        R[1::3, 1] = 1.0
        R[0::3, 2] = -(y - ys)
        R[1::3, 2] = x - xs
        R[2::3, 2] = 1.0
        cons = [M @ R[:, 0], M @ R[:, 1]]
        if np.linalg.norm(f_rot) > 1e-12 * max(np.linalg.norm(M @ R[:, 2]), 1e-300) and section.n_cells > 0:
            cons.append(f_rot)
        else:
            cons.append(M @ R[:, 2])
        Z = _null_space(np.array(cons) @ Phi)
        D = Phi @ Z
        Bd = D.T @ B @ D
        Md = D.T @ M @ D
        lam, Y = _generalized_eigh(0.5 * (Bd + Bd.T), 0.5 * (Md + Md.T))
        D = D @ Y
        # unit amplitude: largest point translation of 1 (length units)
        for c in range(D.shape[1]):
            trans = np.hypot(D[0::3, c], D[1::3, c])
            k = int(np.argmax(trans))
            sgn = np.sign(D[3 * k, c] if abs(D[3 * k, c]) > abs(D[3 * k + 1, c]) else D[3 * k + 1, c]) or 1.0
            scale = trans[k] if trans[k] > 0 else np.max(np.abs(D[:, c]))
            D[:, c] /= sgn * scale
        n_all = D.shape[1]
        nd_keep = n_all if distortional_modes is None else int(min(max(distortional_modes, 0), n_all))
        self.all_distortional = D
        self.all_distortional_stiffness = lam
        self.in_plane = np.column_stack([R, D[:, :nd_keep]])
        self.distortional_stiffness = lam[:nd_keep]
        self.n_distortional = nd_keep

        # warping modes
        omega = section.omega
        classical = np.column_stack([np.ones(N), x - section.centroid[0], y - section.centroid[1], omega])
        if warping == "full":
            W = np.eye(N)
            labels_w = [f"warping at point {i}" for i in range(N)]
        else:
            n_extra = 0 if warping == "classical" else int(warping)
            if n_extra < 0:
                raise ValueError("warping must be 'classical', 'full' or a number of extra modes")
            W = classical.copy()
            labels_w = ["axial", "bending about y (x warping)", "bending about x (y warping)", "torsional warping"]
            if n_extra:
                # complement of the classical modes in the Et-weighted metric, softest in shear first
                Zc = _null_space(classical.T @ Cw)
                lw, Yw = _generalized_eigh(Zc.T @ Gw @ Zc, Zc.T @ Cw @ Zc)
                extra = Zc @ Yw[:, :n_extra]
                W = np.column_stack([W, extra])
                labels_w += [f"warping mode {k + 1}" for k in range(extra.shape[1])]
        self.warping = W
        self.warping_labels = labels_w
        self.in_plane_labels = ["translation x", "translation y", "torsion (rotation)"] + \
            [f"distortion {k + 1}" for k in range(nd_keep)]

    @property
    def n_warping(self) -> int:
        return self.warping.shape[1]

    @property
    def n_in_plane(self) -> int:
        return self.in_plane.shape[1]

    def in_plane_shape(self, mode: int, n_per_wall: int = 8) -> List[np.ndarray]:
        """Deformed contour of in-plane mode ``mode`` (index into ``in_plane``), one (n, 2) array per wall.

        The displacement is returned (add it, scaled, to ``wall_points``).
        """
        return self._contour(self.in_plane[:, mode], n_per_wall)

    def wall_points(self, n_per_wall: int = 8) -> List[np.ndarray]:
        sec = self.section
        out = []
        for k in range(len(sec.walls)):
            s = np.linspace(0.0, 1.0, n_per_wall)
            p0, p1 = sec.points[sec.start[k]], sec.points[sec.end[k]]
            out.append(p0[None, :] + s[:, None] * (p1 - p0)[None, :])
        return out

    def _contour(self, u: np.ndarray, n_per_wall: int) -> List[np.ndarray]:
        sec = self.section
        out = []
        for k in range(len(sec.walls)):
            i, j = sec.start[k], sec.end[k]
            b, t, n = sec.length[k], sec.tangent[k], sec.normal[k]
            Ui, Uj = u[3 * i:3 * i + 3], u[3 * j:3 * j + 3]
            pts = []
            for xi in np.linspace(0.0, 1.0, n_per_wall):
                H, _, _ = _hermite(xi)
                v = ((1 - xi) * Ui[:2] + xi * Uj[:2]) @ t if self.extensible else 0.5 * (Ui[:2] + Uj[:2]) @ t
                un = H[0] * (Ui[:2] @ n) + H[1] * b * Ui[2] + H[2] * (Uj[:2] @ n) + H[3] * b * Uj[2]
                pts.append(v * t + un * n)
            out.append(np.array(pts))
        return out

    @staticmethod
    def _wall_points(section: ThinWalledSection, extensible: bool = False,
                     rows: Sequence[int] = (0, 2, 3, 4, 5)) -> List[_WallPoint]:
        N = len(section.points)
        nd = 4 * N
        pts = []
        gx, gw = _GAUSS_S
        for k in range(len(section.walls)):
            i, j = int(section.start[k]), int(section.end[k])
            b, t, n = section.length[k], section.tangent[k], section.normal[k]
            ui, uj = N + 3 * i, N + 3 * j
            for xg, wg in zip(gx, gw):
                xi = 0.5 * (xg + 1.0)
                H, dH, ddH = _hermite(xi)
                rw = np.zeros(nd)
                rw[i] += 1.0 - xi
                rw[j] += xi
                rws = np.zeros(nd)
                rws[i] -= 1.0 / b
                rws[j] += 1.0 / b
                rv = np.zeros(nd)
                rv[ui:ui + 2] += (1.0 - xi if extensible else 0.5) * t
                rv[uj:uj + 2] += (xi if extensible else 0.5) * t
                ress = np.zeros(nd)
                ress[ui:ui + 2] -= t / b
                ress[uj:uj + 2] += t / b

                def normal_row(c):
                    r = np.zeros(nd)
                    r[ui:ui + 2] += c[0] * n
                    r[ui + 2] += c[1] * b
                    r[uj:uj + 2] += c[2] * n
                    r[uj + 2] += c[3] * b
                    return r

                run = normal_row(H)
                runs = normal_row(dH) / b
                runss = normal_row(ddH) / b**2
                E0 = np.zeros((6, nd))
                E1 = np.zeros((6, nd))
                E2 = np.zeros((6, nd))
                E0[1] = ress
                E0[2] = rws
                E0[4] = -runss
                E1[0] = rw
                E1[2] = rv
                E1[5] = -2.0 * runs
                E2[3] = -run
                rows_ = list(rows)
                pts.append(_WallPoint(k, 0.5 * wg * b, E0[rows_], E1[rows_], E2[rows_], rv, run, rw, xi))
        return pts


def _generalized_eigh(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenpairs of A x = lambda B x (B positive definite), ascending, B-orthonormal."""
    if A.shape[0] == 0:
        return np.zeros(0), np.zeros((0, 0))
    L = np.linalg.cholesky(0.5 * (B + B.T))
    Li = np.linalg.inv(L)
    C = Li @ A @ Li.T
    lam, V = np.linalg.eigh(0.5 * (C + C.T))
    return lam, Li.T @ V


@dataclass
class _Constraint:
    rows: List[np.ndarray]


class GeneralizedBeam:
    """GBT beam of a prismatic ``ThinWalledSection`` from z = 0 to ``length``.

    Parameters
    ----------
    section:
        The cross-section.
    length:
        Span of the beam.
    n_elements:
        Elements along the span, graded towards both ends (supports and loads snap to
        element ends, or for warping and axial loads also to element mid-points).
    z_nodes:
        Element end positions instead of the graded default (z_nodes[0] is the root).
    warping:
        ``"classical"`` (axial, bending and torsional warping), ``"full"`` (every
        section point) or an int: the classical modes plus that many more.
    distortional_modes:
        Number of distortional modes kept (softest first); ``None`` keeps all of them
        and 0 keeps the section rigid in its plane.
    extensible:
        Let the walls stretch round the contour (GBT's transverse extension modes) with
        the full wall ABD, instead of the inextensible, hoop-stress-free walls of
        classical GBT. Adds one mode per wall; matters where loads go into the skins
        locally.
    """

    def __init__(self, section: ThinWalledSection, length: float, n_elements: int = 40,
                 warping: Union[str, int] = "full", distortional_modes: Optional[int] = None,
                 modes: Optional[SectionModes] = None, z_nodes: Optional[Sequence[float]] = None,
                 extensible: bool = False):
        self.section = section
        self.length = float(length)
        self.n_elements = int(n_elements)
        self.modes = modes or SectionModes(section, warping, distortional_modes, extensible)
        self._section_matrices()
        ne = self.n_elements
        if z_nodes is None:
            # graded towards both ends, where the warping and wall-bending boundary layers sit
            u = np.linspace(0.0, 1.0, ne + 1)
            z_nodes = self.length * 0.5 * (1.0 - np.cos(np.pi * u))
        self.z_nodes = np.asarray(z_nodes, dtype=float)
        if len(self.z_nodes) < 2 or np.any(np.diff(self.z_nodes) <= 0):
            raise ValueError("z_nodes must increase")
        self.length = float(self.z_nodes[-1] - self.z_nodes[0])
        self.n_elements = ne = len(self.z_nodes) - 1
        mids = 0.5 * (self.z_nodes[:-1] + self.z_nodes[1:])
        self.z_warp = np.empty(2 * ne + 1)
        self.z_warp[0::2] = self.z_nodes
        self.z_warp[1::2] = mids
        nu, nw = self.modes.n_in_plane, self.modes.n_warping
        self.nu, self.nw = nu, nw
        # DOFs numbered along the span (end node j, then the mid-point after it) to keep the band narrow
        self._pair = 2 * nu + 2 * nw
        self.ndof = ne * self._pair + 2 * nu + nw
        self._constraints: Dict[Tuple[str, int], List[np.ndarray]] = {}
        self._loads = np.zeros(self.ndof)

    # ------------------------------------------------------------ section level
    def _section_matrices(self) -> None:
        m = self.modes
        N = m.n_points
        Psi = np.zeros((4 * N, m.n_warping + m.n_in_plane))
        Psi[:N, :m.n_warping] = m.warping
        Psi[N:, m.n_warping:] = m.in_plane
        self._Psi = Psi
        nm = Psi.shape[1]
        S = np.zeros((3, 3, nm, nm))
        for p in m.points_:
            K = m.wall_stiffness[p.wall]
            Es = [p.E0 @ Psi, p.E1 @ Psi, p.E2 @ Psi]
            KE = [K @ E for E in Es]
            for a in range(3):
                for c in range(3):
                    S[a, c] += p.weight * Es[a].T @ KE[c]
        self.S = S

    # ------------------------------------------------------------ dof layout
    def _u_index(self, j: int, slope: bool) -> np.ndarray:
        base = j * self._pair + (self.nu if slope else 0)
        return np.arange(base, base + self.nu)

    def _w_index(self, m: int) -> np.ndarray:
        j, odd = divmod(m, 2)
        base = j * self._pair + 2 * self.nu + (self.nw if odd else 0)
        return np.arange(base, base + self.nw)

    def _node(self, z: float) -> int:
        j = int(np.argmin(np.abs(self.z_nodes - z)))
        if abs(self.z_nodes[j] - z) > 1e-9 * max(self.length, 1.0):
            raise ValueError(f"z = {z} is not an element end; put it in z_nodes (see GeneralizedBeam.graded_nodes)")
        return j

    def _warp_node(self, z: float) -> int:
        m = int(np.argmin(np.abs(self.z_warp - z)))
        if abs(self.z_warp[m] - z) > 1e-9 * max(self.length, 1.0):
            raise ValueError(f"z = {z} is not an element end or mid-point; put it in z_nodes")
        return m

    @staticmethod
    def graded_nodes(length: float, n_elements: int, stations: Sequence[float] = ()) -> np.ndarray:
        """Element ends from 0 to ``length`` through ``stations`` (ribs, loads), graded towards each.

        Each interval between stations gets elements in proportion to its length (at
        least two), spaced more finely towards both of its ends.
        """
        keys = np.unique(np.concatenate([[0.0, float(length)], np.asarray(stations, dtype=float)]))
        keys = keys[(keys >= 0.0) & (keys <= length)]
        spans = np.diff(keys)
        counts = np.maximum(2, np.round(n_elements * spans / length)).astype(int)
        out = [0.0]
        for a, b, n in zip(keys[:-1], keys[1:], counts):
            u = np.linspace(0.0, 1.0, n + 1)[1:]
            out.extend(a + (b - a) * 0.5 * (1.0 - np.cos(np.pi * u)))
        return np.array(out)

    def _add_rows(self, group: Tuple[str, int], rows: np.ndarray) -> None:
        self._constraints.setdefault(group, []).append(np.atleast_2d(rows))

    # ------------------------------------------------------------ supports
    def clamp(self, z: float = 0.0) -> "GeneralizedBeam":
        """Fully fixed section: no displacement, no warping, walls clamped against bending."""
        j = self._node(z)
        self._add_rows(("u", j), np.eye(self.nu))
        self._add_rows(("du", j), self._wall_slope_rows())
        self._add_rows(("w", 2 * j), np.eye(self.nw))
        return self

    def diaphragm(self, z: float) -> "GeneralizedBeam":
        """A rib rigid in its own plane: no distortion; warping and rigid motion free."""
        j = self._node(z)
        if self.nu > 3:
            self._add_rows(("u", j), np.eye(self.nu)[3:])
        return self

    def restrain_warping(self, z: float) -> "GeneralizedBeam":
        """Stop all warping (an end plate) at ``z``."""
        self._add_rows(("w", self._warp_node(z)), np.eye(self.nw))
        return self

    def fix_rigid(self, z: float, translation: bool = True, twist: bool = True) -> "GeneralizedBeam":
        """Fix the rigid in-plane translations and/or the twist at ``z``."""
        rows = []
        if translation:
            rows += [np.eye(self.nu)[0], np.eye(self.nu)[1]]
        if twist:
            rows.append(np.eye(self.nu)[2])
        if rows:
            self._add_rows(("u", self._node(z)), np.array(rows))
        return self

    def _wall_slope_rows(self) -> np.ndarray:
        """Rows that set d u_n / dz = 0 along every wall (Hermite end values and rotations)."""
        sec = self.section
        N = len(sec.points)
        rows = []
        for k in range(len(sec.walls)):
            n = sec.normal[k]
            for p in (sec.start[k], sec.end[k]):
                r = np.zeros(3 * N)
                r[3 * p:3 * p + 2] = n
                rows.append(r)
        for p in range(N):
            r = np.zeros(3 * N)
            r[3 * p + 2] = 1.0
            rows.append(r)
        return np.array(rows) @ self.modes.in_plane

    # ------------------------------------------------------------ loads
    def add_torque(self, z: float, torque: float) -> "GeneralizedBeam":
        """A torque about the shear centre applied as Saint-Venant shear flows (no distortion)."""
        self._loads[self._u_index(self._node(z), False)[2]] += torque
        return self

    def add_distributed_torque(self, torque_per_length: float) -> "GeneralizedBeam":
        """A uniform torque per length along the whole span, as Saint-Venant shear flows."""
        for e in range(self.n_elements):
            h = self.z_nodes[e + 1] - self.z_nodes[e]
            for j, coef in ((e, 0.5 * h), (e + 1, 0.5 * h)):
                self._loads[self._u_index(j, False)[2]] += coef * torque_per_length
            self._loads[self._u_index(e, True)[2]] += h * h / 12.0 * torque_per_length
            self._loads[self._u_index(e + 1, True)[2]] -= h * h / 12.0 * torque_per_length
        return self

    def add_point_force(self, z: float, point: Union[int, Sequence[float]], force: Sequence[float]) -> "GeneralizedBeam":
        """Force (Fx, Fy[, Fz]) at a section point (an index, or the nearest point to (x, y))."""
        k = self._point_index(point)
        N = self.modes.n_points
        f = np.zeros(3 * N)
        f[3 * k] = force[0]
        f[3 * k + 1] = force[1]
        self._loads[self._u_index(self._node(z), False)] += self.modes.in_plane.T @ f
        if len(force) > 2 and force[2]:
            fw = np.zeros(N)
            fw[k] = force[2]
            self._loads[self._w_index(self._warp_node(z))] += self.modes.warping.T @ fw
        return self

    def add_wall_load(self, z: float, walls: Sequence[int], force: Sequence[float]) -> "GeneralizedBeam":
        """A section force (Fx, Fy) at ``z`` spread uniformly over the length of ``walls``."""
        walls = [int(w) for w in walls]
        total = float(sum(self.section.length[w] for w in walls))
        f = np.zeros(4 * self.modes.n_points)
        F = np.asarray(force, dtype=float)[:2]
        for p in self.modes.points_:
            if p.wall in walls:
                t, n = self.section.tangent[p.wall], self.section.normal[p.wall]
                f += p.weight / total * ((F @ t) * p.v + (F @ n) * p.un)
        N = self.modes.n_points
        self._loads[self._u_index(self._node(z), False)] += self.modes.in_plane.T @ f[N:]
        return self

    def walls_named(self, prefix: str) -> List[int]:
        return [k for k, w in enumerate(self.section.walls) if w.name.startswith(prefix)]

    def _point_index(self, point) -> int:
        if np.ndim(point) == 0:
            return int(point)
        return int(np.argmin(np.linalg.norm(self.section.points - np.asarray(point, float)[None, :2], axis=1)))

    # ------------------------------------------------------------ solve
    def _element_matrix(self, h: float) -> np.ndarray:
        nu, nw = self.nu, self.nw
        gz, gwz = np.polynomial.legendre.leggauss(5)
        # slots: 4 Hermite (nu each) then 3 quadratic (nw each)
        sizes = [nu] * 4 + [nw] * 3
        offs = np.concatenate([[0], np.cumsum(sizes)])
        Ke = np.zeros((offs[-1], offs[-1]))
        S = self.S
        wsl, usl = slice(0, nw), slice(nw, nw + nu)
        for x, wg in zip(gz, gwz):
            zeta = 0.5 * (x + 1.0)
            wgt = 0.5 * wg * h
            H, dH, ddH = _hermite(zeta)
            Hs = [H * np.array([1, h, 1, h]), dH * np.array([1, h, 1, h]) / h, ddH * np.array([1, h, 1, h]) / h**2]
            L = np.array([2 * (zeta - 0.5) * (zeta - 1), -4 * zeta * (zeta - 1), 2 * zeta * (zeta - 0.5)])
            dL = np.array([4 * zeta - 3, -8 * zeta + 4, 4 * zeta - 1]) / h
            Ls = [L, dL, np.zeros(3)]
            for a in range(3):
                for c in range(3):
                    Sac = S[a, c]
                    if not np.any(Sac):
                        continue
                    for si in range(7):
                        fi = Hs[a][si] if si < 4 else Ls[a][si - 4]
                        if fi == 0.0:
                            continue
                        bi = usl if si < 4 else wsl
                        for sj in range(7):
                            fj = Hs[c][sj] if sj < 4 else Ls[c][sj - 4]
                            if fj == 0.0:
                                continue
                            bj = usl if sj < 4 else wsl
                            Ke[offs[si]:offs[si + 1], offs[sj]:offs[sj + 1]] += wgt * fi * fj * Sac[bi, bj]
        return Ke

    def _element_dofs(self, e: int) -> np.ndarray:
        return np.concatenate([self._u_index(e, False), self._u_index(e, True), self._u_index(e + 1, False),
                               self._u_index(e + 1, True), self._w_index(2 * e), self._w_index(2 * e + 1),
                               self._w_index(2 * e + 2)])

    def solve(self) -> "GBTSolution":
        cache: Dict[float, np.ndarray] = {}
        rows, cols, vals = [], [], []
        for e in range(self.n_elements):
            h = float(self.z_nodes[e + 1] - self.z_nodes[e])
            key = round(h / self.length, 12)
            if key not in cache:
                Kh = self._element_matrix(h)
                cache[key] = 0.5 * (Kh + Kh.T)
            Ke = cache[key]
            d = self._element_dofs(e)
            rows.append(np.repeat(d, len(d)))
            cols.append(np.tile(d, len(d)))
            vals.append(Ke.ravel())
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        # constraint elimination: x = T y, one basis per constrained group
        T_rows, T_cols, T_vals = [], [], []
        col = 0
        groups = {}
        for j in range(self.n_elements + 1):
            groups[("u", j)] = self._u_index(j, False)
            groups[("du", j)] = self._u_index(j, True)
        for m in range(2 * self.n_elements + 1):
            groups[("w", m)] = self._w_index(m)
        for key, idx in sorted(groups.items(), key=lambda kv: kv[1][0] if len(kv[1]) else -1):
            if key in self._constraints:
                Z = _null_space(np.vstack(self._constraints[key]))
            else:
                Z = None
            if Z is None:
                T_rows.append(idx)
                T_cols.append(np.arange(col, col + len(idx)))
                T_vals.append(np.ones(len(idx)))
                col += len(idx)
            elif Z.shape[1]:
                r, c = np.nonzero(np.abs(Z) > 0)
                T_rows.append(idx[r])
                T_cols.append(col + c)
                T_vals.append(Z[r, c])
                col += Z.shape[1]
        T_rows, T_cols, T_vals = np.concatenate(T_rows), np.concatenate(T_cols), np.concatenate(T_vals)
        if _sp is not None:
            K = _sp.coo_matrix((vals, (rows, cols)), shape=(self.ndof, self.ndof)).tocsr()
            T = _sp.coo_matrix((T_vals, (T_rows, T_cols)), shape=(self.ndof, col)).tocsr()
            Kr = (T.T @ K @ T).tocsc()
            Fr = T.T @ self._loads
            y = _spla.spsolve(Kr, Fr, permc_spec="MMD_AT_PLUS_A")
            U = T @ y
        else:  # pragma: no cover
            K = np.zeros((self.ndof, self.ndof))
            np.add.at(K, (rows, cols), vals)
            T = np.zeros((self.ndof, col))
            T[T_rows, T_cols] = T_vals
            y = np.linalg.solve(T.T @ K @ T, T.T @ self._loads)
            U = T @ y
        return GBTSolution(self, np.asarray(U).ravel())


class GBTSolution:
    """Modal amplitudes along the span and the section response they give."""

    def __init__(self, beam: GeneralizedBeam, U: np.ndarray):
        self.beam = beam
        self.U = U

    def amplitudes(self, z: float, order: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """(warping, in-plane) modal amplitudes, or their ``order``-th z derivative, at ``z``."""
        b = self.beam
        zn = b.z_nodes
        e = int(np.clip(np.searchsorted(zn, z, side="right") - 1, 0, b.n_elements - 1))
        h = zn[e + 1] - zn[e]
        zeta = (z - zn[e]) / h
        H, dH, ddH = _hermite(zeta)
        sc = np.array([1, h, 1, h])
        Hs = [H * sc, dH * sc / h, ddH * sc / h**2][order]
        L = [np.array([2 * (zeta - 0.5) * (zeta - 1), -4 * zeta * (zeta - 1), 2 * zeta * (zeta - 0.5)]),
             np.array([4 * zeta - 3, -8 * zeta + 4, 4 * zeta - 1]) / h, np.zeros(3)][order]
        U = self.U
        au = (Hs[0] * U[b._u_index(e, False)] + Hs[1] * U[b._u_index(e, True)]
              + Hs[2] * U[b._u_index(e + 1, False)] + Hs[3] * U[b._u_index(e + 1, True)])
        aw = L[0] * U[b._w_index(2 * e)] + L[1] * U[b._w_index(2 * e + 1)] + L[2] * U[b._w_index(2 * e + 2)]
        return aw, au

    def twist(self, z) -> np.ndarray:
        """Amplitude of the rigid rotation mode (about the shear centre)."""
        return np.array([self.amplitudes(float(zz))[1][2] for zz in np.atleast_1d(z)])

    def distortion(self, z, mode: int = 0) -> np.ndarray:
        """Amplitude of distortional mode ``mode`` (0 = softest) along z."""
        return np.array([self.amplitudes(float(zz))[1][3 + mode] for zz in np.atleast_1d(z)])

    def section_displacement(self, z: float) -> np.ndarray:
        """(N, 4): u_x, u_y, w and in-plane rotation r of every section point at z."""
        aw, au = self.amplitudes(z)
        m = self.beam.modes
        u = m.in_plane @ au
        w = m.warping @ aw
        return np.column_stack([u[0::3], u[1::3], w, u[2::3]])

    def mean_twist(self, z, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Least-squares rotation of the section points (uniform weights), as an RBE3 measures it."""
        P = self.beam.section.points
        out = []
        for zz in np.atleast_1d(z):
            d = self.section_displacement(float(zz))
            w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
            w = w / w.sum()
            c = w @ P
            r = P - c
            ut = d[:, :2] - w @ d[:, :2]
            out.append(float(np.sum(w * (r[:, 0] * ut[:, 1] - r[:, 1] * ut[:, 0])) / np.sum(w * np.sum(r * r, axis=1))))
        return np.array(out)

    def wall_resultants(self, z: float) -> List[np.ndarray]:
        """Per wall: resultants at the wall's Gauss points, (n_gauss, 5) [N_zz, N_zs, M_zz, M_ss, M_zs].

        With extensible walls the rows are (n_gauss, 6) [N_zz, N_ss, N_zs, M_zz, M_ss, M_zs].
        """
        b = self.beam
        a = [np.concatenate(self.amplitudes(z, k)) for k in range(3)]
        d = [b._Psi @ ak for ak in a]
        out: Dict[int, List[np.ndarray]] = {}
        for p in b.modes.points_:
            e = p.E0 @ d[0] + p.E1 @ d[1] + p.E2 @ d[2]
            out.setdefault(p.wall, []).append(b.modes.wall_stiffness[p.wall] @ e)
        return [np.array(out[k]) for k in range(len(b.section.walls))]

    def bimoment(self, z) -> np.ndarray:
        """B = integral N_zz omega ds (omega: Saint-Venant warping function about the shear centre)."""
        b = self.beam
        om = b.section.omega
        out = []
        for zz in np.atleast_1d(z):
            res = self.wall_resultants(float(zz))
            counter = {}
            total = 0.0
            for p in b.modes.points_:
                g = counter.get(p.wall, 0)
                counter[p.wall] = g + 1
                i, j = b.section.start[p.wall], b.section.end[p.wall]
                total += p.weight * res[p.wall][g, 0] * ((1 - p.s_fraction) * om[i] + p.s_fraction * om[j])
            out.append(total)
        return np.array(out)


__all__ = ["SectionModes", "GeneralizedBeam", "GBTSolution"]
