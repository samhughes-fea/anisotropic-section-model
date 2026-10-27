"""Timoshenko and Euler-Bernoulli beams from a section stiffness matrix.

``TimoshenkoBeam`` is a straight, prismatic beam along z whose section is described by
a 6 x 6 Timoshenko stiffness (BeamDyn / windIO order: [Fx, Fy, N, Mx, My, Mz] against
[gamma_x, gamma_y, eps, kappa_x, kappa_y, kappa_z], with gamma_x = u_x' - theta_y and
gamma_y = u_y' + theta_x), about the beam's reference axis. All couplings are kept.
A 4 x 4 classical stiffness [N, Mx, My, Mz], or ``shear=False``, gives the
Euler-Bernoulli beam (no shear strain).

Each element is exact for any 6 x 6: its flexibility is the integral of the section
compliance along a cantilever, carried to the element end by equilibrium, so a mesh
of one element per support or load point already gives the exact nodal response for
point loads and uniform distributed loads. Displacements and section forces inside an
element are recovered exactly too.

It needs only a section stiffness, from any source: ``beam_section_properties`` (the
shell model) or ``ThinWalledSection.timoshenko_stiffness`` (thin-wall theory). Linear
static analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DOF_NAMES = ("ux", "uy", "uz", "theta_x", "theta_y", "theta_z")
FORCE_NAMES = ("Fx", "Fy", "N", "Mx", "My", "Mz")


def _transport(a: float) -> np.ndarray:
    """T(a): section forces at a point a behind the end where forces f act, f(z) = T f.

    M(z) = M + (r_end - r_z) x F with r_end - r_z = (0, 0, a).
    """
    T = np.eye(6)
    T[3, 1] = -a
    T[4, 0] = a
    return T


def _load_resultant(q: np.ndarray, a: float) -> np.ndarray:
    """Section forces from a uniform load q (6 per length) over the length a outboard of the section."""
    f = q * a
    f[3] += -q[1] * a * a / 2.0
    f[4] += q[0] * a * a / 2.0
    return f


def _simpson(fun, h: float, n: int = 8):
    """Composite Simpson integral of fun(a) over [0, h] (exact for the cubic integrands here)."""
    a = np.linspace(0.0, h, 2 * n + 1)
    w = np.ones(2 * n + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    w *= h / (6.0 * n)
    return sum(wi * fun(ai) for wi, ai in zip(w, a))


class TimoshenkoBeam:
    """A prismatic beam from z = z_nodes[0] to z_nodes[-1] with a 6 x 6 section stiffness.

    Parameters
    ----------
    stiffness:
        6 x 6 Timoshenko stiffness (BeamDyn order) or 4 x 4 classical stiffness
        [N, Mx, My, Mz] (an Euler-Bernoulli beam).
    length:
        Span, when ``z_nodes`` is not given.
    n_elements:
        Uniform elements along the span (the elements are exact, so they only set where
        loads and supports can go and where results are sampled exactly).
    z_nodes:
        Element ends instead of a uniform mesh.
    shear:
        False drops the shear compliance of a 6 x 6 (Euler-Bernoulli with the same
        extension, bending and torsion, and the same 4 x 4).
    """

    def __init__(self, stiffness, length: Optional[float] = None, n_elements: int = 20,
                 z_nodes: Optional[Sequence[float]] = None, shear: bool = True):
        K = np.asarray(stiffness, dtype=float)
        if K.shape == (6, 6):
            C = np.linalg.inv(0.5 * (K + K.T))
            if not shear:
                C4 = C[2:, 2:]
                C = np.zeros((6, 6))
                C[2:, 2:] = C4
        elif K.shape == (4, 4):
            C = np.zeros((6, 6))
            C[2:, 2:] = np.linalg.inv(0.5 * (K + K.T))
        else:
            raise ValueError("stiffness must be 6 x 6 (Timoshenko) or 4 x 4 (classical)")
        self.compliance = C
        self.shear = bool(shear) and K.shape == (6, 6)
        if z_nodes is None:
            if length is None or length <= 0:
                raise ValueError("give a positive length or z_nodes")
            z_nodes = np.linspace(0.0, float(length), int(n_elements) + 1)
        self.z_nodes = np.asarray(z_nodes, dtype=float)
        if len(self.z_nodes) < 2 or np.any(np.diff(self.z_nodes) <= 0):
            raise ValueError("z_nodes must increase")
        self.length = float(self.z_nodes[-1] - self.z_nodes[0])
        self.n_nodes = len(self.z_nodes)
        self._fixed: Dict[int, set] = {}
        self._prescribed: Dict[Tuple[int, int], float] = {}
        self._nodal = np.zeros(6 * self.n_nodes)
        self._q = np.zeros((self.n_nodes - 1, 6))

    @classmethod
    def with_nodes_at(cls, stiffness, length: float, points: Sequence[float] = (), n_elements: int = 20,
                      **options) -> "TimoshenkoBeam":
        """Uniform mesh with extra element ends at ``points`` (supports, point loads)."""
        z = np.unique(np.concatenate([np.linspace(0.0, length, n_elements + 1), np.asarray(points, float)]))
        return cls(stiffness, z_nodes=z, **options)

    # ------------------------------------------------------------ mesh helpers
    def _node(self, z: float) -> int:
        j = int(np.argmin(np.abs(self.z_nodes - z)))
        if abs(self.z_nodes[j] - z) > 1e-9 * max(self.length, 1.0):
            raise ValueError(f"z = {z} is not a node; build the beam with TimoshenkoBeam.with_nodes_at")
        return j

    def _element_flexibility(self, h: float) -> np.ndarray:
        C = self.compliance
        return _simpson(lambda a: _transport(a).T @ C @ _transport(a), h)

    def _element_load_flexibility(self, h: float, q: np.ndarray) -> np.ndarray:
        """Tip displacement of a cantilever element (clamped at its start) under the uniform load q."""
        C = self.compliance
        return _simpson(lambda a: _transport(a).T @ C @ _load_resultant(q, a), h)

    # ------------------------------------------------------------ supports
    def fix(self, z: float, dofs: Sequence[int] = (0, 1, 2, 3, 4, 5), values: Optional[Sequence[float]] = None
            ) -> "TimoshenkoBeam":
        """Fix DOFs (0-2 translations, 3-5 rotations) at node z, to zero or to ``values``."""
        j = self._node(z)
        for k, d in enumerate(dofs):
            self._fixed.setdefault(j, set()).add(int(d))
            if values is not None:
                self._prescribed[(j, int(d))] = float(values[k])
        return self

    def clamp(self, z: float = 0.0) -> "TimoshenkoBeam":
        return self.fix(z)

    def pin(self, z: float, axial: bool = True, twist: bool = True) -> "TimoshenkoBeam":
        """Simple support: lateral translations fixed (and axial, twist unless told otherwise)."""
        dofs = [0, 1] + ([2] if axial else []) + ([5] if twist else [])
        return self.fix(z, dofs)

    # ------------------------------------------------------------ loads
    def add_force(self, z: float, force: Sequence[float], point: Optional[Sequence[float]] = None) -> "TimoshenkoBeam":
        """Force [Fx, Fy, Fz] or force and moment [Fx, Fy, N, Mx, My, Mz] at node z.

        ``point`` = (x, y) puts the force there instead of on the reference axis
        (adds its moment about the axis).
        """
        f = np.zeros(6)
        f[:len(force)] = force
        if point is not None:
            x, y = point
            f[3] += y * f[2]
            f[4] += -x * f[2]
            f[5] += x * f[1] - y * f[0]
        j = self._node(z)
        self._nodal[6 * j:6 * j + 6] += f
        return self

    def add_distributed(self, load: Sequence[float], z_range: Optional[Tuple[float, float]] = None,
                        point: Optional[Sequence[float]] = None) -> "TimoshenkoBeam":
        """A uniform load per length (3 forces, or 3 forces and 3 moments) over ``z_range``
        (whole elements; the whole span by default), optionally off the axis at ``point``."""
        q = np.zeros(6)
        q[:len(load)] = load
        if point is not None:
            x, y = point
            q[3] += y * q[2]
            q[4] += -x * q[2]
            q[5] += x * q[1] - y * q[0]
        z0, z1 = (self.z_nodes[0], self.z_nodes[-1]) if z_range is None else z_range
        mids = 0.5 * (self.z_nodes[:-1] + self.z_nodes[1:])
        sel = (mids >= z0) & (mids <= z1)
        self._q[sel] += q
        return self

    # ------------------------------------------------------------ solve
    def _element(self, e: int):
        h = self.z_nodes[e + 1] - self.z_nodes[e]
        F = self._element_flexibility(h)
        Kbb = np.linalg.pinv(0.5 * (F + F.T), rcond=1e-14)
        T = _transport(h)
        D = np.vstack([-T, np.eye(6)])  # nodal forces on the element = D f_b
        K = D @ Kbb @ D.T
        q = self._q[e]
        if np.any(q):
            dq = self._element_load_flexibility(h, q)
            fb = -Kbb @ dq  # end force that holds end b in place
            fa = -T @ fb - _load_resultant(q, h)
            P = -np.concatenate([fa, fb])
        else:
            P = np.zeros(12)
        return K, P, Kbb, T

    def solve(self) -> "BeamSolution":
        n = 6 * self.n_nodes
        K = np.zeros((n, n))
        F = self._nodal.copy()
        self._cache = []
        for e in range(self.n_nodes - 1):
            Ke, Pe, Kbb, T = self._element(e)
            idx = np.arange(6 * e, 6 * e + 12)
            K[np.ix_(idx, idx)] += Ke
            F[idx] += Pe
            self._cache.append((Kbb, T))
        fixed = sorted(6 * j + d for j, ds in self._fixed.items() for d in ds)
        if not fixed:
            raise ValueError("the beam needs supports")
        U = np.zeros(n)
        for (j, d), v in self._prescribed.items():
            U[6 * j + d] = v
        free = np.setdiff1d(np.arange(n), fixed)
        rhs = F[free] - K[np.ix_(free, fixed)] @ U[fixed]
        Kff = K[np.ix_(free, free)]
        try:
            U[free] = np.linalg.solve(Kff, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError("the beam is not supported against every rigid-body motion") from exc
        reactions = K @ U - F
        return BeamSolution(self, U, reactions)


@dataclass
class BeamSolution:
    """Displacements [ux, uy, uz, theta_x, theta_y, theta_z] and section forces along the beam."""

    beam: TimoshenkoBeam
    U: np.ndarray
    reactions: np.ndarray

    def node_displacement(self, z: float) -> np.ndarray:
        j = self.beam._node(z)
        return self.U[6 * j:6 * j + 6].copy()

    def reaction(self, z: float) -> np.ndarray:
        j = self.beam._node(z)
        return self.reactions[6 * j:6 * j + 6].copy()

    def _element_of(self, z: float) -> int:
        zn = self.beam.z_nodes
        return int(np.clip(np.searchsorted(zn, z, side="right") - 1, 0, len(zn) - 2))

    def _end_force(self, e: int) -> np.ndarray:
        """Section force at the end b of element e (outboard face)."""
        b = self.beam
        Kbb, T = b._cache[e]
        da, db = self.U[6 * e:6 * e + 6], self.U[6 * e + 6:6 * e + 12]
        fb = Kbb @ (db - T.T @ da)
        q = b._q[e]
        if np.any(q):
            h = b.z_nodes[e + 1] - b.z_nodes[e]
            fb -= Kbb @ b._element_load_flexibility(h, q)
        return fb

    def section_forces(self, z) -> np.ndarray:
        """[Fx, Fy, N, Mx, My, Mz] at each z (exact inside the elements)."""
        b = self.beam
        out = []
        for zz in np.atleast_1d(z):
            e = self._element_of(float(zz))
            a = b.z_nodes[e + 1] - float(zz)
            out.append(_transport(a) @ self._end_force(e) + _load_resultant(b._q[e], a))
        return np.array(out)

    def strains(self, z) -> np.ndarray:
        """[gamma_x, gamma_y, eps, kappa_x, kappa_y, kappa_z] at each z."""
        return self.section_forces(z) @ self.beam.compliance.T

    def displacement(self, z) -> np.ndarray:
        """[ux, uy, uz, theta_x, theta_y, theta_z] at each z (exact inside the elements)."""
        b = self.beam
        C = b.compliance
        out = []
        for zz in np.atleast_1d(z):
            e = self._element_of(float(zz))
            z0 = b.z_nodes[e]
            s = float(zz) - z0
            da = self.U[6 * e:6 * e + 6]
            d = _transport(s).T @ da
            if s > 0:
                fb = self._end_force(e)
                h = b.z_nodes[e + 1] - z0
                q = b._q[e]

                def integrand(t):  # strain at z0 + t, carried to the point at s
                    f = _transport(h - t) @ fb + _load_resultant(q, h - t)
                    return _transport(s - t).T @ (C @ f)

                d = d + _simpson(integrand, s)
            out.append(d)
        return np.array(out)


def euler_bernoulli_beam(stiffness, length: Optional[float] = None, **options) -> TimoshenkoBeam:
    """The Euler-Bernoulli beam of a 6 x 6 (or 4 x 4) section stiffness: no shear strain."""
    return TimoshenkoBeam(stiffness, length, shear=False, **options)


__all__ = ["TimoshenkoBeam", "BeamSolution", "euler_bernoulli_beam", "DOF_NAMES", "FORCE_NAMES"]
