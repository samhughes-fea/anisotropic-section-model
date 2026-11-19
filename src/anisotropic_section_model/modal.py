"""Natural frequencies and mode shapes of beams and blades from their 6 x 6 sections.

``ModalBeam`` is a linear free-vibration model of a beam whose sections carry a 6 x 6
Timoshenko stiffness and a 6 x 6 mass matrix per unit length, both fully coupled, in
the order and sign convention of ``TimoshenkoBeam`` and BeamDyn: stiffness
[Fx, Fy, N, Mx, My, Mz] against [gamma_x, gamma_y, eps, kappa_x, kappa_y, kappa_z], and
mass on [u_x, u_y, u_z, theta_x, theta_y, theta_z]. The properties may change along the
span (they are interpolated linearly between stations), each station's local frame may
be turned about the axis (a blade's twist), and the reference axis may be curved
(prebend, sweep). Point masses and springs can be added, and any DOFs fixed.

Model
-----
The linearised geometrically exact beam (Reissner, Simo), which is also what BeamDyn
linearises: with u and theta the displacement and small rotation of the reference axis
x0(s) in the beam's axes, the sectional strains in the local frame Lambda0(s) are

    gamma = Lambda0^T (u' + x0' x theta),   kappa = Lambda0^T theta'

with ' the derivative along the arc length s. Kinetic energy is
1/2 ∫ [u_dot, theta_dot]^T M [u_dot, theta_dot] ds with M rotated by Lambda0 into the
beam's axes. Every coupling in the two 6 x 6 matrices (extension-twist, bending-twist,
shear-bending, offsets of the mass centre and of the elastic axis) is kept.

It is discretised with high-order Lagrange elements on Gauss-Lobatto-Legendre nodes (the
spectral elements BeamDyn uses), integrated with Gauss points: displacement and
rotation share the same interpolation, and order 6 (the default) is free of shear
locking for any practical slenderness. The eigenproblem K phi = omega^2 M phi is solved
densely (``scipy.linalg.eigh``, low modes as the largest of M phi = mu K phi); modes are
normalised to unit modal mass.

``prismatic_frequencies`` is the exact answer for a prismatic beam: it integrates the
12 first-order equations of the same theory (displacements, rotations, section forces
and moments) across the span with a matrix exponential and finds the frequencies where
the end conditions can be met. It is the reference the finite elements are tested
against.

Axes
----
The arrays are in whatever axes the caller's 6 x 6 matrices are in, the beam's axis
being z. ``ModalBeam.from_beamdyn`` takes a ``BeamDynBlade``, which is in BeamDyn's
blade frame (x towards the suction side, so flapwise; y towards the trailing edge, so
edgewise). This module does no frame conversion of its own beyond the station twist
and the axis curvature.

Sections are taken normal to the curved axis, as BeamDyn takes them. The thin-walled
and shell sections in this package are cut normal to z; for a prebent blade the
difference is second order in the slope of the axis.

Scope: linear, undamped, non-rotating (no centrifugal stiffening or gyroscopic terms).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DOF_NAMES = ("ux", "uy", "uz", "theta_x", "theta_y", "theta_z")
#: mode families, from where a mode's kinetic energy is (see ``ModalResult.energy_fractions``)
FAMILIES = ("x-bending", "y-bending", "axial", "torsion")
_ENDS = ("clamped", "free", "pinned")


# ----------------------------------------------------------------- small helpers
def _skew(v: np.ndarray) -> np.ndarray:
    """[v]x with [v]x w = v x w."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _rot_z(angle_deg: float) -> np.ndarray:
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _align_z(t: np.ndarray) -> np.ndarray:
    """The smallest rotation taking e_z to the unit vector t (Rodrigues)."""
    ez = np.array([0.0, 0.0, 1.0])
    c = float(ez @ t)
    v = np.cross(ez, t)
    if c > 1.0 - 1e-15:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        raise ValueError("the reference axis turns back on itself")
    V = _skew(v)
    return np.eye(3) + V + V @ V / (1.0 + c)


def _block(R: np.ndarray) -> np.ndarray:
    out = np.zeros((6, 6))
    out[:3, :3] = R
    out[3:, 3:] = R
    return out


def gll_points(order: int) -> np.ndarray:
    """The ``order + 1`` Gauss-Lobatto-Legendre points on [-1, 1]."""
    if order < 1:
        raise ValueError("element order must be at least 1")
    if order == 1:
        return np.array([-1.0, 1.0])
    P = np.polynomial.legendre.Legendre.basis(order)
    inner = np.sort(np.real(P.deriv().roots()))
    return np.concatenate([[-1.0], inner, [1.0]])


def _lagrange(nodes: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lagrange basis values and derivatives, (len(x), len(nodes)) each."""
    n = len(nodes)
    x = np.atleast_1d(x)
    N = np.ones((len(x), n))
    dN = np.zeros((len(x), n))
    for a in range(n):
        others = [b for b in range(n) if b != a]
        den = np.prod([nodes[a] - nodes[b] for b in others])
        for i, xi in enumerate(x):
            terms = np.array([xi - nodes[b] for b in others])
            N[i, a] = np.prod(terms) / den
            d = 0.0
            for k in range(len(others)):
                d += np.prod(np.delete(terms, k))
            dN[i, a] = d / den
    return N, dN


def _as_stations(values, n: int, name: str) -> np.ndarray:
    A = np.asarray(values, dtype=float)
    if A.shape == (6, 6):
        A = np.broadcast_to(A, (n, 6, 6)).copy()
    if A.shape != (n, 6, 6):
        raise ValueError(f"{name} must be 6 x 6 or one 6 x 6 per station ({n}, 6, 6)")
    return 0.5 * (A + np.swapaxes(A, -1, -2))


# ----------------------------------------------------------------- the model
class ModalBeam:
    """A beam or blade for natural frequencies and mode shapes.

    Parameters
    ----------
    s:
        Station positions along the reference axis (arc length, m), increasing. Two
        stations and one 6 x 6 pair make a prismatic beam.
    stiffness, mass:
        6 x 6 Timoshenko stiffness and mass per unit length, one pair or one per
        station, in each station's local frame (see the module docstring for the order).
    frame_angle:
        Angle (degrees) each station's local frame is turned about the axis, positive
        anticlockwise about the tangent (right-handed); one value or one per station.
        A ``BeamDynBlade``'s twist enters as ``-twist`` (``from_beamdyn``).
    axis:
        (k, 3) points of a curved reference axis from root to tip, k >= 2; by default a
        straight axis along z. The stations' ``s`` are arc lengths along it, measured
        from its first point.
    order:
        Polynomial order of the spectral elements (default 6).
    max_element_length:
        Elements are cut at every station, support and point mass, then split until
        none is longer than this (default: a tenth of the span).
    """

    def __init__(self, s, stiffness, mass, frame_angle=0.0, axis=None, order: int = 6,
                 max_element_length: Optional[float] = None):
        self.s = np.asarray(s, dtype=float).ravel()
        n = len(self.s)
        if n < 2 or np.any(np.diff(self.s) <= 0.0):
            raise ValueError("station positions s must increase (at least two)")
        self.stiffness = _as_stations(stiffness, n, "stiffness")
        self.mass = _as_stations(mass, n, "mass")
        self.frame_angle = np.broadcast_to(np.asarray(frame_angle, dtype=float), (n,)).copy()
        self.order = int(order)
        self.length = float(self.s[-1] - self.s[0])
        self.max_element_length = (self.length / 10.0 if max_element_length is None
                                   else float(max_element_length))
        self._axis = None if axis is None else self._build_axis(np.asarray(axis, dtype=float))
        self._fixed: Dict[float, set] = {}
        self._point_mass: List[Tuple[float, np.ndarray]] = []
        self._springs: List[Tuple[float, np.ndarray]] = []
        self.family_names = dict(zip(FAMILIES, FAMILIES))

    # ------------------------------------------------------------ construction
    @classmethod
    def prismatic(cls, stiffness, mass, length: float, **options) -> "ModalBeam":
        """A straight prismatic beam of one section, z from 0 to ``length``."""
        return cls([0.0, float(length)], stiffness, mass, **options)

    @classmethod
    def from_beamdyn(cls, blade, curved: bool = True, **options) -> "ModalBeam":
        """A ``BeamDynBlade`` (stations, matrices, key points, twist), root clamped.

        BeamDyn turns each station's frame by minus its initial twist (it negates the
        twist on reading); the twist is interpolated linearly between key points.
        ``curved=False`` straightens the axis (keeps its length). Families are named
        flap (x, towards the suction side) and edge (y, towards the trailing edge).
        """
        kp = np.asarray(blade.key_points, dtype=float)
        L = blade.length
        twist = np.interp(blade.eta, blade.key_point_eta, blade.twist)
        beam = cls(blade.eta * L, blade.stiffness, blade.mass, frame_angle=-twist,
                   axis=kp if curved else None, **options)
        beam.family_names.update({"x-bending": "flap", "y-bending": "edge"})
        return beam.clamp(0.0)

    def _build_axis(self, pts: np.ndarray):
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            raise ValueError("axis needs (k, 3) points, k >= 2")
        from scipy.interpolate import CubicSpline

        pts = pts - pts[0]
        chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        spline = CubicSpline(chord, pts, axis=0, bc_type="natural") if len(pts) > 2 else None
        # arc length of the spline against its parameter, on a fine grid
        p = np.linspace(0.0, chord[-1], 4001)
        if spline is None:
            arc = p.copy()
            pos = lambda q: np.outer(np.atleast_1d(q) / chord[-1], pts[-1])  # noqa: E731
            tan = lambda q: np.broadcast_to(pts[-1] / chord[-1], (len(np.atleast_1d(q)), 3))  # noqa: E731
        else:
            d = np.linalg.norm(spline(p, 1), axis=1)
            arc = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(p))])
            pos = lambda q: spline(np.atleast_1d(q))  # noqa: E731
            tan = lambda q: spline(np.atleast_1d(q), 1)  # noqa: E731
        if arc[-1] < self.length * (1.0 - 1e-6):
            raise ValueError(f"the axis ({arc[-1]:.6g} long) is shorter than the stations' span ({self.length:.6g})")

        def at(s):
            q = np.interp(np.atleast_1d(s) - self.s[0], arc, p)
            T = tan(q)
            return pos(q), T / np.linalg.norm(T, axis=1)[:, None]

        return at

    # ------------------------------------------------------------ supports, masses
    def fix(self, s: float, dofs: Sequence[int] = (0, 1, 2, 3, 4, 5)) -> "ModalBeam":
        """Fix DOFs (0-2 translations, 3-5 rotations, in the beam's axes) at ``s``."""
        self._fixed.setdefault(float(s), set()).update(int(d) for d in dofs)
        return self

    def clamp(self, s: float = 0.0) -> "ModalBeam":
        return self.fix(s)

    def pin(self, s: float, axial: bool = True, twist: bool = True) -> "ModalBeam":
        """Simple support: lateral translations fixed (and axial, twist unless told otherwise)."""
        return self.fix(s, [0, 1] + ([2] if axial else []) + ([5] if twist else []))

    def add_mass(self, s: float, mass) -> "ModalBeam":
        """A point mass at ``s``: a scalar (translational) or a 6 x 6 mass matrix."""
        M = np.asarray(mass, dtype=float)
        if M.ndim == 0:
            M = np.diag([float(M)] * 3 + [0.0] * 3)
        self._point_mass.append((float(s), 0.5 * (M + M.T)))
        return self

    def add_spring(self, s: float, stiffness) -> "ModalBeam":
        """A grounded spring at ``s``: six diagonal values or a 6 x 6."""
        K = np.asarray(stiffness, dtype=float)
        if K.ndim == 1:
            K = np.diag(K)
        self._springs.append((float(s), 0.5 * (K + K.T)))
        return self

    # ------------------------------------------------------------ mesh
    def element_ends(self) -> np.ndarray:
        pts = [*self.s, *self._fixed, *(s for s, _ in self._point_mass), *(s for s, _ in self._springs)]
        pts = np.unique(np.round(np.asarray(pts, float), 12))
        if pts[0] < self.s[0] - 1e-9 or pts[-1] > self.s[-1] + 1e-9:
            raise ValueError("supports and point masses must lie on the span")
        ends = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            k = max(1, int(np.ceil((b - a) / self.max_element_length - 1e-9)))
            ends.extend(a + (b - a) * np.arange(1, k + 1) / k)
        return np.asarray(ends)

    def _properties(self, s: np.ndarray):
        """Local stiffness, mass and frame angle at ``s`` (linear between stations)."""
        i = np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.s) - 2)
        w = ((s - self.s[i]) / (self.s[i + 1] - self.s[i]))[:, None, None]
        K = (1.0 - w) * self.stiffness[i] + w * self.stiffness[i + 1]
        M = (1.0 - w) * self.mass[i] + w * self.mass[i + 1]
        ang = np.interp(s, self.s, self.frame_angle)
        return K, M, ang

    def _frames(self, s: np.ndarray):
        """Axis positions, unit tangents and local frames Lambda0 (columns: local x, y, z)."""
        if self._axis is None:
            pos = np.column_stack([np.zeros_like(s), np.zeros_like(s), s - self.s[0]])
            tan = np.broadcast_to([0.0, 0.0, 1.0], (len(s), 3)).copy()
        else:
            pos, tan = self._axis(s)
        _, _, ang = self._properties(s)
        lam = np.array([_align_z(t) @ _rot_z(a) for t, a in zip(tan, ang)])
        return pos, tan, lam

    def _dof(self, s: float, ends: np.ndarray) -> int:
        """Global node number of the element end at s."""
        e = int(np.argmin(np.abs(ends - s)))
        if abs(ends[e] - s) > 1e-9 * max(self.length, 1.0):
            raise ValueError(f"s = {s} is not an element end")
        return e * self.order

    def assemble(self):
        """Global stiffness and mass (dense, all DOFs) and the mesh.

        Returns ``(K, M, mesh)``; ``mesh`` holds the element ends, the nodes' arc
        lengths ``s`` and positions ``x``, and what ``ModalResult`` needs to
        interpolate the shapes.
        """
        p = self.order
        ends = self.element_ends()
        ne = len(ends) - 1
        nn = ne * p + 1
        xi_n = gll_points(p)
        xi_q, w_q = np.polynomial.legendre.leggauss(p + 1)
        Nq, dNq = _lagrange(xi_n, xi_q)
        K = np.zeros((6 * nn, 6 * nn))
        M = np.zeros((6 * nn, 6 * nn))
        s_nodes = np.zeros(nn)
        for e in range(ne):
            a, b = ends[e], ends[e + 1]
            h = b - a
            sq = a + 0.5 * h * (xi_q + 1.0)
            s_nodes[e * p:e * p + p + 1] = a + 0.5 * h * (xi_n + 1.0)
            Kl, Ml, _ = self._properties(sq)
            _, tan, lam = self._frames(sq)
            idx = np.arange(6 * e * p, 6 * (e * p + p + 1))
            Ke = np.zeros((len(idx), len(idx)))
            Me = np.zeros_like(Ke)
            for q in range(len(xi_q)):
                R = _block(lam[q])
                Sg = R @ Kl[q] @ R.T
                Mg = R @ Ml[q] @ R.T
                jac = 0.5 * h
                B = np.zeros((6, len(idx)))
                Nm = np.zeros((6, len(idx)))
                T = _skew(tan[q])
                for k in range(p + 1):
                    c = 6 * k
                    dn = dNq[q, k] / jac
                    B[0:3, c:c + 3] = dn * np.eye(3)
                    B[0:3, c + 3:c + 6] = Nq[q, k] * T
                    B[3:6, c + 3:c + 6] = dn * np.eye(3)
                    Nm[:, c:c + 6] = Nq[q, k] * np.eye(6)
                wq = w_q[q] * jac
                Ke += wq * B.T @ Sg @ B
                Me += wq * Nm.T @ Mg @ Nm
            K[np.ix_(idx, idx)] += Ke
            M[np.ix_(idx, idx)] += Me
        for s, Mp in self._point_mass:
            j = self._dof(s, ends)
            M[6 * j:6 * j + 6, 6 * j:6 * j + 6] += Mp
        for s, Ks in self._springs:
            j = self._dof(s, ends)
            K[6 * j:6 * j + 6, 6 * j:6 * j + 6] += Ks
        pos, _, _ = self._frames(s_nodes)
        mesh = {"ends": ends, "s": s_nodes, "x": pos, "xi": xi_n}
        return 0.5 * (K + K.T), 0.5 * (M + M.T), mesh

    def fixed_dofs(self, ends: np.ndarray) -> np.ndarray:
        return np.array(sorted(6 * self._dof(s, ends) + d for s, ds in self._fixed.items() for d in ds), dtype=int)

    # ------------------------------------------------------------ solve
    def modes(self, n_modes: int = 10) -> "ModalResult":
        """The lowest ``n_modes`` natural frequencies and mode shapes.

        Without supports the first six modes are the rigid-body motions (frequency
        about zero).

        Solved as M phi = mu (K + a M) phi for the largest mu = 1 / (omega^2 + a): the low
        modes are then the well-conditioned end of the spectrum, which keeps them
        accurate when the shear stiffness is many orders above the bending stiffness.
        The shift a is zero for a supported beam and about the first flexible omega^2 for
        a free one. M may be singular (sections without rotary inertia).
        """
        from scipy.linalg import LinAlgError, cholesky, eigh

        K, M, mesh = self.assemble()
        n = K.shape[0]
        fixed = self.fixed_dofs(mesh["ends"])
        free = np.setdiff1d(np.arange(n), fixed)
        Kf, Mf = K[np.ix_(free, free)], M[np.ix_(free, free)]
        k = min(int(n_modes), len(free))
        shift = 0.0
        try:
            cholesky(Kf)
        except LinAlgError:  # rigid-body motions are free: shift by a flexible eigenvalue
            lam = eigh(Kf, Mf, eigvals_only=True, subset_by_index=[0, min(len(free), 12) - 1])
            flexible = lam[lam > 1e-8 * max(lam[-1], 1e-300)]
            shift = float(flexible[0]) if len(flexible) else 1.0
        try:
            mu, vec = eigh(Mf, Kf + shift * Mf, subset_by_index=[len(free) - k, len(free) - 1])
        except LinAlgError as exc:
            raise ValueError("the beam's stiffness is singular even with the shift; check the supports "
                             "and that every section has stiffness") from exc
        mu, vec = mu[::-1], vec[:, ::-1]
        lam = 1.0 / mu - shift
        vec = vec / np.sqrt(np.einsum("ik,ij,jk->k", vec, Mf, vec))  # unit modal mass
        phi = np.zeros((k, n))
        phi[:, free] = vec.T
        omega = np.sqrt(np.clip(lam, 0.0, None))
        return ModalResult(self, omega, phi.reshape(k, -1, 6), mesh, K, M)


# ----------------------------------------------------------------- results
@dataclass
class ModalResult:
    """Natural frequencies and mass-normalised mode shapes of a ``ModalBeam``."""

    beam: ModalBeam
    omega: np.ndarray                   # rad/s
    shapes: np.ndarray                  # (n_modes, n_nodes, 6): [ux, uy, uz, theta_x, theta_y, theta_z]
    mesh: dict
    K: np.ndarray = field(repr=False)
    M: np.ndarray = field(repr=False)

    @property
    def frequencies(self) -> np.ndarray:
        """Natural frequencies (Hz)."""
        return self.omega / (2.0 * np.pi)

    @property
    def s(self) -> np.ndarray:
        """Arc length of the nodes."""
        return self.mesh["s"]

    def __len__(self) -> int:
        return len(self.omega)

    def shape(self, mode: int, s=None) -> np.ndarray:
        """Mode shape [ux, uy, uz, theta_x, theta_y, theta_z] at ``s`` (default: the nodes),
        interpolated with the element functions."""
        phi = self.shapes[mode]
        if s is None:
            return phi.copy()
        ends, p, xi = self.mesh["ends"], self.beam.order, self.mesh["xi"]
        out = []
        for ss in np.atleast_1d(s):
            e = int(np.clip(np.searchsorted(ends, ss, side="right") - 1, 0, len(ends) - 2))
            x = 2.0 * (ss - ends[e]) / (ends[e + 1] - ends[e]) - 1.0
            N, _ = _lagrange(xi, np.array([x]))
            out.append(N[0] @ phi[e * p:e * p + p + 1])
        return np.array(out)

    def energy_fractions(self) -> np.ndarray:
        """(n_modes, 4) share of each mode's kinetic energy in the families ``FAMILIES``.

        From the diagonal of the mass matrix: x-bending is u_x with theta_y, y-bending
        u_y with theta_x, axial u_z, torsion theta_z (in the beam's axes). Rows sum to 1.
        """
        d = np.diag(self.M).reshape(-1, 6)
        e = (self.shapes ** 2 * d[None]).sum(axis=1)  # (n_modes, 6)
        fam = np.column_stack([e[:, 0] + e[:, 4], e[:, 1] + e[:, 3], e[:, 2], e[:, 5]])
        return fam / np.maximum(fam.sum(axis=1, keepdims=True), 1e-300)

    def labels(self) -> List[str]:
        """'1st flap', '2nd edge', '1st torsion', ...: each mode's dominant family, counted.

        Rigid-body modes of a free beam are counted too; check ``frequencies``.
        """
        count: Dict[str, int] = {}
        out = []
        names = self.beam.family_names
        for row in self.energy_fractions():
            fam = names[FAMILIES[int(np.argmax(row))]]
            count[fam] = count.get(fam, 0) + 1
            out.append(f"{_ordinal(count[fam])} {fam}")
        return out

    def find(self, label: str) -> int:
        """Index of the mode with this label (e.g. '1st edge')."""
        labels = self.labels()
        if label not in labels:
            raise KeyError(f"no mode {label!r} among {labels}")
        return labels.index(label)

    def table(self) -> List[dict]:
        names = self.beam.family_names
        fr = self.energy_fractions()
        return [{"mode": k + 1, "label": lab, "frequency": float(f),
                 **{names[n]: float(v) for n, v in zip(FAMILIES, fr[k])}}
                for k, (lab, f) in enumerate(zip(self.labels(), self.frequencies))]


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def mac(a, b) -> float:
    """Modal assurance criterion |a.b|^2 / (|a|^2 |b|^2) of two shape vectors (any shape)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float((a @ b) ** 2 / ((a @ a) * (b @ b)))


# ----------------------------------------------------------------- exact prismatic beam
def _state_matrix(stiffness, mass, omega: float) -> np.ndarray:
    """A(omega) of y' = A y, y = [u, theta, F, M]: the prismatic beam's 12 equations.

    u' = (C f)_F - e_z x theta, theta' = (C f)_M, F' = -omega^2 (M q)_u,
    M' = -e_z x F - omega^2 (M q)_theta, with f = [F, M], q = [u, theta], C = S^-1.
    """
    C = np.linalg.inv(0.5 * (np.asarray(stiffness, float) + np.asarray(stiffness, float).T))
    Mm = 0.5 * (np.asarray(mass, float) + np.asarray(mass, float).T)
    E = _skew(np.array([0.0, 0.0, 1.0]))
    A = np.zeros((12, 12))
    A[0:6, 6:12] = C
    A[0:3, 3:6] -= E
    A[6:12, 0:6] = -omega ** 2 * Mm
    A[9:12, 6:9] -= E
    return A


def _end_rows(end: str) -> np.ndarray:
    """Indices of the state [u, theta, F, M] that vanish at an end."""
    if end == "clamped":
        return np.arange(6)
    if end == "free":
        return np.arange(6, 12)
    if end == "pinned":  # translations and twist held, bending moments free
        return np.array([0, 1, 2, 5, 9, 10])
    raise ValueError(f"an end is one of {_ENDS}")


def prismatic_frequencies(stiffness, mass, length: float, f_max: float, ends=("clamped", "free"),
                          n_grid: int = 2000) -> np.ndarray:
    """Exact natural frequencies (Hz) below ``f_max`` of a prismatic beam, fully coupled.

    The 12 first-order equations of the beam (``_state_matrix``) are integrated across
    the span exactly, y(L) = expm(A L) y(0); a frequency is one where some y(0) meeting
    the root condition also meets the tip condition, a zero of the 6 x 6 end matrix.
    ``ends`` are the root and tip conditions: "clamped", "free" or "pinned" (lateral
    translations, axial and twist held). Repeated frequencies (a symmetric section) are
    returned once per independent mode. Accurate while expm stays well conditioned:
    the first ten or so modes of a slender beam.
    """
    from scipy.linalg import expm
    from scipy.optimize import minimize_scalar

    root, tip = _end_rows(ends[0]), _end_rows(ends[1])
    free0 = np.setdiff1d(np.arange(12), root)
    S = np.asarray(stiffness, dtype=float)
    # scale forces and moments so the end matrix is balanced
    fs = np.ones(12)
    fs[6:9] = np.mean(np.abs(np.diag(S)[3:5])) / length ** 2
    fs[9:12] = np.mean(np.abs(np.diag(S)[3:5])) / length
    fs[0:3] = 1.0 / length
    D_scale = np.diag(1.0 / fs)

    def sv(f):
        Phi = expm(_state_matrix(S, mass, 2.0 * np.pi * f) * length)
        Phi = D_scale @ Phi @ np.diag(fs)
        # no balancing that depends on omega: at a root of an uncoupled motion (torsion
        # of an orthotropic box) a whole row of the end matrix vanishes
        s = np.linalg.svd(Phi[np.ix_(tip, free0)], compute_uv=False)
        return s / s[0]

    grid = np.linspace(f_max / n_grid, 1.02 * f_max, n_grid)  # a little past f_max: a root at the end is a dip
    smin = np.array([sv(f)[-1] for f in grid])
    out = []
    for i in range(1, n_grid - 1):
        if smin[i] <= smin[i - 1] and smin[i] <= smin[i + 1]:
            r = minimize_scalar(lambda f: sv(f)[-1], bracket=(grid[i - 1], grid[i], grid[i + 1]),
                                tol=1e-13)
            f = float(r.x)
            s = sv(f)
            # a root is a deep dip (elsewhere the minima stay near the neighbours' level);
            # a repeated root has as many singular values that small
            if s[-1] < 1e-2 * max(smin[i - 1], smin[i + 1]):
                out.extend([f] * int(np.sum(s <= 1e3 * s[-1] + 1e-300)))
    return np.array(sorted(f for f in out if f <= f_max))


__all__ = ["ModalBeam", "ModalResult", "prismatic_frequencies", "mac", "gll_points", "FAMILIES", "DOF_NAMES"]
