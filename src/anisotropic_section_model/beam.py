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

Tapered beams
-------------
The stiffness can vary along the span: give one 6 x 6 per node (interpolated linearly in
between) or a function of z. Each element's flexibility is then the integral of the
compliance that varies along it, not that of one cut, so a coarse mesh of a tapered beam
gives the response of the tapered beam, where a stepped (piecewise prismatic) one needs
many elements to converge. The taper's other effect, the Resal coupling of shear with
bending and extension through the inclined walls, is in the section stiffness
(``thin_walled.taper``, ``solid.taper``); ``from_section_order`` puts the right-handed
section stiffness of ``thin_walled`` and ``solid`` (beam along x) into this order.

``BeamSolution.displacement_derivative`` differentiates the nodal displacements exactly
for a change of the stiffness along the span (a layup sensitivity).
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


#: ``from_section_order``: this order's entry ``i`` is the section order's ``_SECTION_ORDER[i]``
#: ([gamma_x, gamma_y, eps, kappa_x, kappa_y, kappa_z] from [eps, kappa_y, kappa_z, gamma_xy,
#: gamma_xz, phi]): the section's (x, y, z) are this beam's (z, x, y), a rotation, so no signs change
_SECTION_ORDER = (3, 4, 0, 1, 2, 5)


def from_section_order(matrix) -> np.ndarray:
    """A right-handed section 6 x 6 of ``thin_walled`` / ``solid`` (beam along x, order
    ``[eps, kappa_y, kappa_z, gamma_xy, gamma_xz, phi]``) in this module's order (beam along
    z, section axes ``(x, y)`` = the section's ``(y, z)``). Works on (..., 6, 6) stacks."""
    K = np.asarray(matrix, dtype=float)
    idx = np.array(_SECTION_ORDER)
    return K[..., idx[:, None], idx[None, :]]


def _compliance(K: np.ndarray, shear: bool, dK: Optional[np.ndarray] = None) -> np.ndarray:
    """6 x 6 compliance of a 6 x 6 or 4 x 4 stiffness (shear rows zero without shear), or its
    derivative ``-C dK C`` for the stiffness changing by ``dK``."""
    K = np.asarray(K, dtype=float)
    if K.shape not in ((6, 6), (4, 4)):
        raise ValueError("stiffness must be 6 x 6 (Timoshenko) or 4 x 4 (classical)")
    Ci = np.linalg.inv(0.5 * (K + K.T))
    if dK is not None:
        Ci = -Ci @ (0.5 * (dK + dK.T)) @ Ci
    if K.shape == (6, 6) and shear:
        return Ci
    C = np.zeros((6, 6))
    C[2:, 2:] = Ci[-4:, -4:]
    return C


def _stiffness_field(stiffness, z_nodes: np.ndarray):
    """``z -> stiffness``: a constant, one matrix per node interpolated linearly, or a function."""
    if callable(stiffness):
        return lambda z: np.asarray(stiffness(z), dtype=float)
    K = np.asarray(stiffness, dtype=float)
    if K.ndim == 2:
        if K.shape not in ((6, 6), (4, 4)):
            raise ValueError("stiffness must be 6 x 6 (Timoshenko) or 4 x 4 (classical)")
        return lambda z: K
    if K.ndim != 3 or K.shape[0] != len(z_nodes) or K.shape[1:] not in ((6, 6), (4, 4)):
        raise ValueError(f"a stiffness per node needs shape ({len(z_nodes)}, 6, 6) or ({len(z_nodes)}, 4, 4)")

    def at(z):
        j = int(np.clip(np.searchsorted(z_nodes, z, side="right") - 1, 0, len(z_nodes) - 2))
        t = (z - z_nodes[j]) / (z_nodes[j + 1] - z_nodes[j])
        return (1.0 - t) * K[j] + t * K[j + 1]

    return at


_GL_X, _GL_W = np.polynomial.legendre.leggauss(4)


def _gauss(fun, h: float, n: int = 8):
    """Composite 4-point Gauss-Legendre integral of fun(a) over [0, h] in n pieces: the
    integrands of a tapered element are smooth but not polynomial (a compliance)."""
    edges = np.linspace(0.0, h, n + 1)
    total = 0.0
    for a0, a1 in zip(edges[:-1], edges[1:]):
        m, r = 0.5 * (a0 + a1), 0.5 * (a1 - a0)
        total = total + sum(w * r * fun(m + r * x) for x, w in zip(_GL_X, _GL_W))
    return total


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
        [N, Mx, My, Mz] (an Euler-Bernoulli beam). For a beam whose section changes along
        the span (tapered): one per node, shape ``(n_nodes, 6, 6)`` (or 4 x 4), linear in
        between, or a function ``z -> 6 x 6``.
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
        if z_nodes is None:
            if length is None or length <= 0:
                raise ValueError("give a positive length or z_nodes")
            z_nodes = np.linspace(0.0, float(length), int(n_elements) + 1)
        self.z_nodes = np.asarray(z_nodes, dtype=float)
        if len(self.z_nodes) < 2 or np.any(np.diff(self.z_nodes) <= 0):
            raise ValueError("z_nodes must increase")
        self._stiffness_at = _stiffness_field(stiffness, self.z_nodes)
        K0 = self._stiffness_at(self.z_nodes[0])
        self.shear = bool(shear) and K0.shape == (6, 6)
        self._shear_option = bool(shear)
        prismatic = not callable(stiffness) and np.asarray(stiffness).ndim == 2
        #: the compliance of a prismatic beam (``None`` for a tapered one: ``compliance_at``)
        self.compliance = _compliance(K0, shear) if prismatic else None
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

    @property
    def tapered(self) -> bool:
        return self.compliance is None

    def stiffness_at(self, z: float) -> np.ndarray:
        """The section stiffness at ``z`` (as given: 6 x 6 or 4 x 4)."""
        return self._stiffness_at(float(z))

    def compliance_at(self, z: float) -> np.ndarray:
        """The 6 x 6 section compliance at ``z`` (zero shear rows without shear)."""
        if self.compliance is not None:
            return self.compliance
        return _compliance(self._stiffness_at(float(z)), self._shear_option)

    def _compliance_rate(self, dstiffness, z: float) -> np.ndarray:
        """``d compliance_at(z)`` for the stiffness changing by ``dstiffness(z)``: ``-C dK C``."""
        return _compliance(self._stiffness_at(float(z)), self._shear_option, np.asarray(dstiffness(float(z)), float))

    def _element_flexibility(self, e: int, compliance=None) -> np.ndarray:
        """Flexibility of element ``e`` as a cantilever clamped at its start: ``int T^T C T``
        with ``C`` at ``z_b - a`` (``compliance(z)`` replaces ``compliance_at``)."""
        zb = self.z_nodes[e + 1]
        h = zb - self.z_nodes[e]
        Cz = compliance or self.compliance_at
        rule = _simpson if (compliance is None and not self.tapered) else _gauss
        return rule(lambda a: _transport(a).T @ Cz(zb - a) @ _transport(a), h)

    def _element_load_flexibility(self, e: int, q: np.ndarray, compliance=None) -> np.ndarray:
        """Tip displacement of a cantilever element (clamped at its start) under the uniform load q."""
        zb = self.z_nodes[e + 1]
        h = zb - self.z_nodes[e]
        Cz = compliance or self.compliance_at
        rule = _simpson if (compliance is None and not self.tapered) else _gauss
        return rule(lambda a: _transport(a).T @ Cz(zb - a) @ _load_resultant(q, a), h)

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
        F = self._element_flexibility(e)
        Kbb = np.linalg.pinv(0.5 * (F + F.T), rcond=1e-14)
        T = _transport(h)
        D = np.vstack([-T, np.eye(6)])  # nodal forces on the element = D f_b
        K = D @ Kbb @ D.T
        q = self._q[e]
        if np.any(q):
            dq = self._element_load_flexibility(e, q)
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
        self._system = (K, free)
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

    def displacement_derivative(self, dstiffness) -> np.ndarray:
        """Exact ``d U`` (n_nodes, 6) of the nodal displacements for a change ``dstiffness`` of
        the section stiffness along the span, given as the stiffness is (one matrix, one per
        node, or a function of z), the loads and supports held.

        Each element's flexibility ``int T^T C T`` changes by ``int T^T dC T`` with
        ``dC = -C dK C``, its stiffness ``K_bb = F^-1`` by ``-K_bb dF K_bb`` (and the
        distributed-load terms likewise), and ``K dU = dP - dK U`` on the free DOFs.
        """
        b = self.beam
        if getattr(b, "_system", None) is None:
            raise ValueError("solve the beam first")
        K, free = b._system
        dfield = _stiffness_field(dstiffness, b.z_nodes)

        def dC(z):
            return b._compliance_rate(dfield, z)

        n = 6 * b.n_nodes
        dK = np.zeros((n, n))
        dP = np.zeros(n)
        for e in range(b.n_nodes - 1):
            Kbb, T = b._cache[e]
            dF = b._element_flexibility(e, dC)
            dKbb = -Kbb @ (0.5 * (dF + dF.T)) @ Kbb
            D = np.vstack([-T, np.eye(6)])
            idx = np.arange(6 * e, 6 * e + 12)
            dK[np.ix_(idx, idx)] += D @ dKbb @ D.T
            q = b._q[e]
            if np.any(q):
                dq, ddq = b._element_load_flexibility(e, q), b._element_load_flexibility(e, q, dC)
                dfb = -dKbb @ dq - Kbb @ ddq
                dP[idx] -= np.concatenate([-T @ dfb, dfb])
        dU = np.zeros(n)
        rhs = (dP - dK @ self.U)[free]
        dU[free] = np.linalg.solve(K[np.ix_(free, free)], rhs)
        return dU.reshape(-1, 6)

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
            fb -= Kbb @ b._element_load_flexibility(e, q)
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
        b = self.beam
        z = np.atleast_1d(z)
        return np.array([b.compliance_at(zz) @ f for zz, f in zip(z, self.section_forces(z))])

    def displacement(self, z) -> np.ndarray:
        """[ux, uy, uz, theta_x, theta_y, theta_z] at each z (exact inside the elements)."""
        b = self.beam
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
                    return _transport(s - t).T @ (b.compliance_at(z0 + t) @ f)

                d = d + (_gauss if b.tapered else _simpson)(integrand, s)
            out.append(d)
        return np.array(out)


def euler_bernoulli_beam(stiffness, length: Optional[float] = None, **options) -> TimoshenkoBeam:
    """The Euler-Bernoulli beam of a 6 x 6 (or 4 x 4) section stiffness: no shear strain."""
    return TimoshenkoBeam(stiffness, length, shear=False, **options)


__all__ = ["TimoshenkoBeam", "BeamSolution", "euler_bernoulli_beam", "from_section_order", "DOF_NAMES", "FORCE_NAMES"]
