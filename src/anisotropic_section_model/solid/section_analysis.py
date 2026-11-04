# pre_processing/section_integration/solid/section_analysis.py
"""Saint-Venant analysis of beam sections meshed in 2D: ``SolidSectionAnalysis``.

The Giavotto method (Giavotto, Borri et al. 1983), as implemented in BECAS (Blasques and
Stolpe 2012), on a 2D mesh of the section with 3D anisotropic materials. It gives the fully
coupled 6 x 6 Timoshenko stiffness, the shear and tension centres and the mass properties
of any section shape: solid, thick-walled, laminated, sandwich or a mix. The results are
the same quantities as ``thin_walled.ThinWalledSectionAnalysis`` gives from wall mid-lines,
in the same order and signs, so ``SectionAnalysis.from_solid`` fills the same element data.

Kinematics (right-handed, as ``thin_walled/section_analysis.py``)
------------------------------------------------------------------
The section moves rigidly and warps in all three directions, ``g = (g_x, g_y, g_z)``::

    u_x = u - y theta_z + z theta_y + g_x
    u_y = v - z theta_x + g_y,   u_z = w + y theta_x + g_z

so with ``psi = [eps, kappa_y, kappa_z, gamma_xy, gamma_xz, phi]`` (``STRAINS``) the 3D
engineering strains, Voigt order ``[xx, yy, zz, yz, xz, xy]``, are::

    eps_xx = eps + z kappa_y - y kappa_z          + g_x'
    eps_yy = dg_y/dy,   eps_zz = dg_z/dz,   gamma_yz = dg_y/dz + dg_z/dy
    gamma_xz = gamma_xz + y phi + dg_x/dz         + g_z'
    gamma_xy = gamma_xy - z phi + dg_x/dy         + g_y'

or ``eps = Z psi + B g_hat + N_x g_hat'`` in discrete form (``g_hat`` the nodal warping).
Unlike the thin-walled walls there is no ``Y`` (``psi'``) term: the section's own
through-thickness shear carries what the Kirchhoff walls' ``Q_x`` did.

Weak form
---------
With ``E = int B^T C B``, ``Cx = int B^T C N_x``, ``R = int B^T C Z``, ``L = int N_x^T C Z``,
``A = int Z^T C Z`` and ``D`` the warping fields equal to rigid motions (``g_x = 1``,
``g_y = 1``, ``g_z = 1``, the in-plane rotation ``(0, -z, y)`` and ``g_x = z``, ``g_x = -y``)::

    [E   R  D] [g1  ]   [0    ]          [E   R  D] [g0  ]   [-(Cx - Cx^T) g1 + L psi1]
    [R^T A  0] [psi1] = [T F0 ]          [R^T A  0] [psi0] = [F0 - L^T g1             ]
    [D^T 0  0] [lam1]   [0    ]          [D^T 0  0] [lam0]   [0                       ]

the thin-walled system with its ``Y`` terms zero. With ``eps = Phi F0`` at ``x = 0`` the
flexibility is ``int Phi^T C Phi dA``. Tied nodes (``SectionMesh.ties``, where a wall's end
joins another wall's strip) are eliminated before the solve.

Stresses and restrained warping
-------------------------------
``point_stresses`` and ``stress_at`` give the 3D stresses anywhere in the section, in
section, material and wall axes; the wall-axes ``sigma_nn``, ``tau_sn``, ``tau_xn`` are the
interlaminar stresses of laminated walls and the material-axes stresses those of cores.
``benscoter_stiffness`` extends the 6 x 6 with the torsional warping amplitude ``psi``
(Benscoter / Vlasov), in the thin-walled order. Cross-section distortion is left to GBT.

Public API
----------
SolidSectionAnalysis, PointStress, BENSCOTER_STRAINS.

See Also
--------
docs/element_library/solid_section_analysis.md
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pre_processing.section_integration.solid.elements import shape
from pre_processing.section_integration.solid.material import bond_matrix, material_axes
from pre_processing.section_integration.solid.mesh import SectionMesh
from pre_processing.section_integration.thin_walled.section_analysis import CLASSICAL, GEOMETRIC_TERMS, T_RATE

__all__ = ["SolidSectionAnalysis", "PointStress", "BENSCOTER_STRAINS"]

#: generalised strains of the restrained-warping (Benscoter) section, as the thin-walled one
BENSCOTER_STRAINS: Tuple[str, ...] = ("eps", "kappa_y", "kappa_z", "gamma_xy", "gamma_xz", "phi", "d", "psi'")

_XX, _YY, _ZZ, _YZ, _XZ, _XY = range(6)


def _operators(y: float, z: float, N: npt.NDArray[np.float64], dN: npt.NDArray[np.float64]):
    """``Z`` (6, 6), ``B`` (6, 3n) and ``N_x`` (6, 3n) at a point; element DOFs node-major ``(g_x, g_y, g_z)``."""
    n = N.size
    Z = np.zeros((6, 6))
    Z[_XX, :3] = (1.0, z, -y)
    Z[_XZ, 4], Z[_XZ, 5] = 1.0, y
    Z[_XY, 3], Z[_XY, 5] = 1.0, -z
    B = np.zeros((6, 3 * n))
    gx, gy, gz = slice(0, None, 3), slice(1, None, 3), slice(2, None, 3)
    B[_YY, gy] = dN[0]
    B[_ZZ, gz] = dN[1]
    B[_YZ, gy] = dN[1]
    B[_YZ, gz] = dN[0]
    B[_XZ, gx] = dN[1]
    B[_XY, gx] = dN[0]
    Nx = np.zeros((6, 3 * n))
    Nx[_XX, gx] = N
    Nx[_XZ, gz] = N
    Nx[_XY, gy] = N
    return Z, B, Nx


class _Point:
    """A quadrature point with its strain operators and material stiffness."""

    __slots__ = ("element", "y", "z", "weight", "dofs", "Z", "B", "Nx", "C", "N")

    def __init__(self, q, dofs, C):
        self.element, self.y, self.z, self.weight = q.element, q.y, q.z, q.weight
        self.dofs = dofs
        self.N = q.N
        self.Z, self.B, self.Nx = _operators(q.y, q.z, q.N, q.dN)
        self.C = C


@dataclass(frozen=True)
class PointStress:
    """3D state at a point of the section.

    All vectors are Voigt-ordered with engineering shear strains:

    - ``strain``, ``stress``: section axes ``[xx, yy, zz, yz, xz, xy]``;
    - ``stress_material``: material axes ``[11, 22, 33, 23, 13, 12]`` (ply or core axes);
    - ``stress_wall``: wall axes ``[xx, ss, nn, sn, xn, xs]`` (``s`` the element's wall
      direction ``alpha``, ``n = x cross s``). ``sigma_nn``, ``tau_sn`` and ``tau_xn`` are
      the interlaminar (through-thickness) stresses of a laminated wall.
    """

    element: int
    y: float
    z: float
    strain: npt.NDArray[np.float64]
    stress: npt.NDArray[np.float64]
    stress_material: npt.NDArray[np.float64]
    stress_wall: npt.NDArray[np.float64]

    @property
    def interlaminar(self) -> Tuple[float, float, float]:
        """``(sigma_nn, tau_sn, tau_xn)`` in wall axes."""
        return float(self.stress_wall[2]), float(self.stress_wall[3]), float(self.stress_wall[4])


class _NaturalPoint:
    """``QuadraturePoint``-like data at natural coordinates ``(xi, eta)`` of element ``e`` (zero weight)."""

    def __init__(self, mesh: SectionMesh, e: int, xi: float, eta: float):
        conn = mesh.elements[e]
        X = mesh.nodes[conn]
        N, dNn = shape(mesh.kinds[e], xi, eta)
        J = dNn @ X
        if np.linalg.det(J) <= 0.0:
            raise ValueError(f"({xi}, {eta}) is outside element {e} or the element is degenerate there")
        self.element = e
        self.y, self.z = (float(v) for v in N @ X)
        self.weight = 0.0
        self.N, self.dN, self.nodes = N, np.linalg.solve(J, dNn), conn


class SolidSectionAnalysis:
    """Saint-Venant stiffness and properties of a section meshed in 2D.

    Parameters
    ----------
    mesh : SectionMesh
        The section (``solid/mesh.py``). Results are about its coordinate origin.

    Attributes
    ----------
    n_dof : int
        Warping unknowns after tied nodes are eliminated (three per free node).

    Notes
    -----
    Results are in the order ``STRAINS`` / ``FORCES`` of ``thin_walled.section_analysis``
    (right-handed). See the module docstring for the weak form.
    """

    def __init__(self, mesh: SectionMesh) -> None:
        self.mesh = mesh
        self.section_id = mesh.section_id
        n = mesh.nodes.shape[0]
        used = np.zeros(n, dtype=bool)
        for conn in mesh.elements:
            used[conn] = True
        for row in mesh.ties.values():
            for m, _ in row:
                if m in mesh.ties:
                    raise ValueError(f"tied node {m} is itself a master of another tie")
                used[m] = True
        free = [i for i in range(n) if used[i] and i not in mesh.ties]
        self._free_index = {node: k for k, node in enumerate(free)}
        self._free_nodes = np.array(free, dtype=np.int64)
        self.n_dof = 3 * len(free)
        self.n_full = 3 * n
        rows, cols, vals = [], [], []
        for node in range(n):
            if node in mesh.ties:
                pairs = [(self._free_index[m], w) for m, w in mesh.ties[node]]
            elif used[node]:
                pairs = [(self._free_index[node], 1.0)]
            else:
                continue
            for k, w in pairs:
                for c in range(3):
                    rows.append(3 * node + c)
                    cols.append(3 * k + c)
                    vals.append(w)
        #: full nodal warping = T @ free warping
        self.T = sp.csr_matrix((vals, (rows, cols)), shape=(self.n_full, self.n_dof))

    # ------------------------------------------------------------------ #
    # Discretisation
    # ------------------------------------------------------------------ #
    @cached_property
    def _points(self) -> List[_Point]:
        mesh = self.mesh
        C_el = [mesh.element_stiffness(e) for e in range(len(mesh.elements))]
        out = []
        for q in mesh.quadrature_points():
            dofs = (3 * q.nodes[:, None] + np.arange(3)[None, :]).ravel()
            out.append(_Point(q, dofs, C_el[q.element]))
        return out

    @cached_property
    def _full_modes(self) -> npt.NDArray[np.float64]:
        """Nodal warping (n_full, 6) of the rigid motions (see the module docstring)."""
        y, z = self.mesh.nodes[:, 0], self.mesh.nodes[:, 1]
        M = np.zeros((self.n_full, 6))
        M[0::3, 0] = 1.0
        M[1::3, 1] = 1.0
        M[2::3, 2] = 1.0
        M[1::3, 3], M[2::3, 3] = -z, y
        M[0::3, 4] = z
        M[0::3, 5] = -y
        return M

    @cached_property
    def _matrices(self) -> Dict[str, object]:
        nf = self.n_full
        e_r, e_c, e_v = [], [], []
        c_v = []
        Mv = []
        R = np.zeros((nf, 6))
        L = np.zeros((nf, 6))
        A = np.zeros((6, 6))
        by_element: Dict[int, list] = {}
        for p in self._points:
            by_element.setdefault(p.element, []).append(p)
        for pts in by_element.values():
            d = pts[0].dofs
            nd = d.size
            Ee = np.zeros((nd, nd))
            Ce = np.zeros((nd, nd))
            Me = np.zeros((nd, nd))
            for p in pts:
                CB = p.C @ p.B
                CZ = p.C @ p.Z
                w = p.weight
                Ee += w * p.B.T @ CB
                Ce += w * CB.T @ p.Nx
                R[d] += w * p.B.T @ CZ
                L[d] += w * p.Nx.T @ CZ
                A += w * p.Z.T @ CZ
                Me += w * np.kron(np.outer(p.N, p.N), np.eye(3))
            rr, cc = np.meshgrid(d, d, indexing="ij")
            e_r.append(rr.ravel())
            e_c.append(cc.ravel())
            e_v.append(Ee.ravel())
            c_v.append(Ce.ravel())
            Mv.append(Me.ravel())
        r, c = np.concatenate(e_r), np.concatenate(e_c)
        T = self.T
        E = T.T @ sp.csr_matrix((np.concatenate(e_v), (r, c)), shape=(nf, nf)) @ T
        Cx = T.T @ sp.csr_matrix((np.concatenate(c_v), (r, c)), shape=(nf, nf)) @ T
        M = sp.csr_matrix((np.concatenate(Mv), (r, c)), shape=(nf, nf))
        D_full = T.T @ (M @ self._full_modes)
        return {"E": E.tocsr(), "Cx": Cx.tocsr(), "R": T.T @ R, "L": T.T @ L, "A": A,
                "D_modes": D_full}

    @staticmethod
    def _independent(D: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """An orthonormal basis of the columns of ``D`` (a strip on ``z = 0`` has no ``g_x = z`` mode)."""
        U, sv, _ = np.linalg.svd(D, full_matrices=False)
        return U[:, sv > 1e-12 * sv[0]]

    def _saddle(self, D: npt.NDArray[np.float64], with_psi: bool = True):
        m = self._matrices
        blocks = [[m["E"], sp.csr_matrix(m["R"]) if with_psi else None, sp.csr_matrix(D)]]
        if with_psi:
            blocks.append([sp.csr_matrix(m["R"].T), sp.csr_matrix(m["A"]), None])
            blocks.append([sp.csr_matrix(D.T), None, None])
        else:
            blocks = [[m["E"], sp.csr_matrix(D)], [sp.csr_matrix(D.T), None]]
        return spla.splu(sp.bmat(blocks, format="csc"))

    @cached_property
    def _solution(self) -> Dict[str, npt.NDArray[np.float64]]:
        m = self._matrices
        nd = self.n_dof
        D = self._independent(m["D_modes"])
        lu = self._saddle(D)
        n_tot = nd + 6 + D.shape[1]
        Cx, L = m["Cx"], m["L"]

        rhs1 = np.zeros((n_tot, 6))
        rhs1[nd:nd + 6] = T_RATE
        X1 = lu.solve(rhs1)
        W1, P1 = X1[:nd], X1[nd:nd + 6]

        rhs0 = np.zeros((n_tot, 6))
        rhs0[:nd] = -(Cx - Cx.T) @ W1 + L @ P1
        rhs0[nd:nd + 6] = np.eye(6) - L.T @ W1
        X0 = lu.solve(rhs0)
        W0, P0 = X0[:nd], X0[nd:nd + 6]
        W0f, W1f = self.T @ W0, self.T @ W1

        flex = np.zeros((6, 6))
        for p in self._points:
            Phi = self._phi(p, P0, W0f, W1f)
            flex += p.weight * Phi.T @ p.C @ Phi
        flex = 0.5 * (flex + flex.T)
        return {"W0": W0f, "W1": W1f, "P0": P0, "P1": P1, "flexibility": flex}

    @staticmethod
    def _phi(p: _Point, P0, W0f, W1f) -> npt.NDArray[np.float64]:
        return p.Z @ P0 + p.B @ W0f[p.dofs] + p.Nx @ W1f[p.dofs]

    # ------------------------------------------------------------------ #
    # Stiffness
    # ------------------------------------------------------------------ #
    @property
    def flexibility(self) -> npt.NDArray[np.float64]:
        """Section compliance (6, 6): ``psi = flexibility @ F`` about the origin."""
        return self._solution["flexibility"].copy()

    @cached_property
    def _stiffness(self) -> npt.NDArray[np.float64]:
        K = np.linalg.inv(self._solution["flexibility"])
        return 0.5 * (K + K.T)

    @property
    def stiffness(self) -> npt.NDArray[np.float64]:
        """Timoshenko section stiffness (6, 6): ``F = stiffness @ psi``, order ``STRAINS``."""
        return self._stiffness.copy()

    @property
    def classical_stiffness(self) -> npt.NDArray[np.float64]:
        """Shear-rigid stiffness (4, 4) for ``[eps, kappa_y, kappa_z, phi]`` (inverse of the
        ``[N, M_y, M_z, T]`` flexibility block)."""
        K = np.linalg.inv(self._solution["flexibility"][np.ix_(CLASSICAL, CLASSICAL)])
        return 0.5 * (K + K.T)

    def strain_response(self, forces: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Generalised strains (6,) for section forces (6,) in order ``FORCES``."""
        return self._solution["flexibility"] @ np.asarray(forces, dtype=np.float64)

    def point_strains(self, forces: npt.ArrayLike) -> List[Tuple[int, float, float, npt.NDArray[np.float64]]]:
        """3D strains at the quadrature points for section forces ``forces`` (constant shear).

        Returns ``(element, y, z, strain (6,))`` with the strain in Voigt order
        ``[xx, yy, zz, yz, xz, xy]`` (engineering shear strains), section axes.
        """
        sol = self._solution
        f = np.asarray(forces, dtype=np.float64)
        P0, W0, W1 = sol["P0"] @ f, sol["W0"] @ f, sol["W1"] @ f
        return [(p.element, p.y, p.z, p.Z @ P0 + p.B @ W0[p.dofs] + p.Nx @ W1[p.dofs]) for p in self._points]

    def nodal_warping(self, forces: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Warping ``(g_x, g_y, g_z)`` at every node (n_nodes, 3) at ``x = 0`` for ``forces``."""
        return (self._solution["W0"] @ np.asarray(forces, dtype=np.float64)).reshape(-1, 3)

    # ------------------------------------------------------------------ #
    # Centres
    # ------------------------------------------------------------------ #
    @property
    def shear_centre(self) -> Tuple[float, float]:
        """Point ``(y, z)`` where a transverse force causes no twist."""
        F = self._solution["flexibility"]
        return float(-F[5, 4] / F[5, 5]), float(F[5, 3] / F[5, 5])

    @property
    def tension_centre(self) -> Tuple[float, float]:
        """Point ``(y, z)`` where an axial force causes no bending curvature."""
        F = self._solution["flexibility"]
        M = np.array([[-F[1, 2], F[1, 1]], [-F[2, 2], F[2, 1]]])
        y_t, z_t = np.linalg.solve(M, -np.array([F[1, 0], F[2, 0]]))
        return float(y_t), float(z_t)

    # ------------------------------------------------------------------ #
    # Torsional warping and E Gamma
    # ------------------------------------------------------------------ #
    @cached_property
    def _axial_modulus(self) -> List[float]:
        """Axial modulus ``E_x`` per element: ``C_xx`` with the in-plane stresses
        ``sigma_yy, sigma_zz, tau_yz`` free and the shear strains ``gamma_xz, gamma_xy`` held,
        as the axial ``K_xx`` of the thin-walled wall law (``N_ss`` free, ``gamma_xs`` held)."""
        out = []
        for e in range(len(self.mesh.elements)):
            C = self.mesh.element_stiffness(e)
            ip = [_YY, _ZZ, _YZ]
            out.append(float(C[0, 0] - C[0, ip] @ np.linalg.solve(C[np.ix_(ip, ip)], C[ip, 0])))
        return out

    @cached_property
    def _torsion_warping(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Warping (n_full,) per unit twist rate and the coefficients ``c`` of its extension and
        bending parts ``c0 + c1 z - c2 y`` in the axial stiffness (removed: pole at the shear centre)."""
        m = self._matrices
        nd = self.n_dof
        # only the warping fields with no strain are fixed; g_x = y, z shear the section
        D = self._independent(m["D_modes"][:, :4])
        lu = self._saddle(D, with_psi=False)
        rhs = np.zeros(nd + D.shape[1])
        rhs[:nd] = -m["R"][:, 5]
        omega = self.T @ lu.solve(rhs)[:nd]
        G = np.zeros((3, 3))
        b = np.zeros(3)
        for p in self._points:
            Ex = self._axial_modulus[p.element]
            r = np.array([1.0, p.z, -p.y])
            om = p.N @ omega[p.dofs[0::3]]
            G += p.weight * Ex * np.outer(r, r)
            b += p.weight * Ex * r * om
        return omega, np.linalg.solve(G, b)

    def warping_function(self) -> npt.NDArray[np.float64]:
        """Torsional (axial) warping ``omega`` per unit twist rate at every node, about the shear
        centre and orthogonal to extension and bending in the axial stiffness."""
        omega, c = self._torsion_warping
        y, z = self.mesh.nodes[:, 0], self.mesh.nodes[:, 1]
        return omega[0::3] - (c[0] + c[1] * z - c[2] * y)

    @property
    def warping_constant(self) -> float:
        """Vlasov warping stiffness ``E Gamma``: the axial energy of a unit warping gradient,
        the section relaxing in its own plane (``benscoter_stiffness[7, 7]``). It includes the
        warping through the wall thickness (secondary warping)."""
        return float(self._benscoter[0][7, 7])

    # ------------------------------------------------------------------ #
    # Restrained warping (Benscoter): the 7th beam DOF psi, warping amplitude
    # ------------------------------------------------------------------ #
    @cached_property
    def _omega_tilde(self) -> npt.NDArray[np.float64]:
        """Torsional warping (n_full,) about the shear centre, its extension and bending parts removed."""
        omega, c = self._torsion_warping
        return omega - self._full_modes[:, [0, 4, 5]] @ c

    @cached_property
    def _relaxed_gradient(self) -> npt.NDArray[np.float64]:
        """Warping (n_full,) that lets the section relax in its own plane (and in shear) under the
        axial strain ``N_x omega`` of a unit warping gradient ``psi'``: the Poisson contraction of
        restrained warping. Without it ``K[7, 7]`` would use the laterally constrained ``C_xx``."""
        m = self._matrices
        nd = self.n_dof
        free_dofs = (3 * self._free_nodes[:, None] + np.arange(3)[None, :]).ravel()
        om_free = self._omega_tilde[free_dofs]
        # only the strain-free fields are fixed (g_x = y, z would hold the shear strains)
        D = self._independent(m["D_modes"][:, :4])
        lu = self._saddle(D, with_psi=False)
        rhs = np.zeros(nd + D.shape[1])
        rhs[:nd] = -(m["Cx"] @ om_free)
        return self.T @ lu.solve(rhs)[:nd]

    def _benscoter_operator(self, p) -> npt.NDArray[np.float64]:
        """3D strains (6, 8) at a point per unit Benscoter strain ``[psi6, d, psi']`` (``BENSCOTER_STRAINS``)."""
        sol = self._solution
        om = self._omega_tilde[p.dofs]
        G = np.zeros((6, 8))
        G[:, :6] = self._phi(p, sol["P0"], sol["W0"], sol["W1"]) @ self._stiffness
        G[:, 6] = p.B @ om        # warping deviation d = psi - phi strains the section in shear
        # warping amplitude gradient psi' stretches it; the section relaxes in its own plane
        G[:, 7] = p.Nx @ om + p.B @ self._relaxed_gradient[p.dofs]
        return G

    @cached_property
    def _benscoter(self) -> Tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64]]:
        K8 = np.zeros((8, 8))
        X = np.zeros((len(GEOMETRIC_TERMS), 2))
        inertia = 0.0
        mesh = self.mesh
        for p in self._points:
            G = self._benscoter_operator(p)
            K8 += p.weight * G.T @ p.C @ G
            X += self._geometric_rows(p, p.C @ G[:, 6:])
            om = p.N @ self._omega_tilde[p.dofs[0::3]]
            inertia += p.weight * mesh.materials[mesh.material_ids[p.element]].rho * om * om
        K8 = 0.5 * (K8 + K8.T)
        K8[:6, :6] = self._stiffness
        return K8, float(inertia), X

    @property
    def benscoter_stiffness(self) -> npt.NDArray[np.float64]:
        """Restrained-warping section stiffness (8, 8), right-handed, about the origin.

        Strains ``BENSCOTER_STRAINS``: the Saint-Venant section of ``psi6`` plus the torsional
        warping ``omega`` (all three components, about the shear centre) times
        ``d = psi - phi``, with ``psi`` the warping amplitude. Then ``eps += N_x omega psi'``
        and ``eps += B omega d``. The order and meaning are those of
        ``ThinWalledSectionAnalysis.benscoter_stiffness``; here the warping also varies
        through the wall thickness (secondary warping). The leading 6 x 6 block is
        ``stiffness``; ``K[7, 7]`` is ``warping_constant``.
        """
        return self._benscoter[0].copy()

    @property
    def warping_inertia(self) -> float:
        """Mass moment of the warping amplitude, ``int rho omega_x^2 dA`` per length."""
        return self._benscoter[1]

    @property
    def benscoter_geometric_terms(self) -> npt.NDArray[np.float64]:
        """Second-order terms (7, 2) per unit Benscoter strain ``[d, psi']``."""
        return self._benscoter[2].copy()

    # ------------------------------------------------------------------ #
    # 3D stresses over the section
    # ------------------------------------------------------------------ #
    def _state(self, p, forces: Optional[npt.ArrayLike], benscoter: Optional[npt.ArrayLike]) -> PointStress:
        if (forces is None) == (benscoter is None):
            raise ValueError("give either section forces (6,) or Benscoter strains (8,)")
        if benscoter is not None:
            eps = self._benscoter_operator(p) @ np.asarray(benscoter, dtype=np.float64)
        else:
            sol = self._solution
            eps = self._phi(p, sol["P0"], sol["W0"], sol["W1"]) @ np.asarray(forces, dtype=np.float64)
        e = p.element
        C = self.mesh.element_stiffness(e)
        sig = C @ eps
        th, al = self.mesh.theta[e], self.mesh.alpha[e]
        s_mat = np.linalg.solve(bond_matrix(material_axes(th, al)), sig)
        s_wall = np.linalg.solve(bond_matrix(material_axes(0.0, al)), sig)
        return PointStress(e, float(p.y), float(p.z), eps, sig, s_mat, s_wall)

    def point_stresses(self, forces: Optional[npt.ArrayLike] = None,
                       benscoter: Optional[npt.ArrayLike] = None) -> List[PointStress]:
        """3D state at every quadrature point (``PointStress``).

        Give the right-handed section forces ``forces`` (6,), order ``FORCES``, for the
        Saint-Venant state (forces varying along the beam as with no distributed load, so
        the transverse shear and interlaminar stresses of the bending gradient are in), or
        the right-handed Benscoter strains ``benscoter`` (8,) for restrained warping.
        """
        return [self._state(p, forces, benscoter) for p in self._points]

    def stress_at(self, element: int, xi: float, eta: float, forces: Optional[npt.ArrayLike] = None,
                  benscoter: Optional[npt.ArrayLike] = None) -> PointStress:
        """3D state at natural coordinates ``(xi, eta)`` of a mesh element (as ``point_stresses``).

        On a ``SectionMesh.from_thin_walled`` mesh ``eta`` runs through the thickness, so a
        ply interface is at ``eta = +1`` of one row and ``eta = -1`` of the next.
        """
        q = _NaturalPoint(self.mesh, element, xi, eta)
        dofs = (3 * q.nodes[:, None] + np.arange(3)[None, :]).ravel()
        return self._state(_Point(q, dofs, self.mesh.element_stiffness(element)), forces, benscoter)

    # ------------------------------------------------------------------ #
    # Interlaminar stresses on ply interfaces
    # ------------------------------------------------------------------ #
    @cached_property
    def _row_neighbours(self) -> Dict[Tuple[int, int], int]:
        """Layered ``Q8`` rows: ``(e, +1)`` is the element on top of ``e`` (its edge ``eta = +1``,
        nodes 3-2), ``(e, -1)`` the one below (edge ``eta = -1``, nodes 0-1)."""
        mesh = self.mesh
        bottom, top = {}, {}
        for e, (conn, kind) in enumerate(zip(mesh.elements, mesh.kinds)):
            if kind != "Q8":
                continue
            bottom[frozenset((int(conn[0]), int(conn[1])))] = e
            top[frozenset((int(conn[3]), int(conn[2])))] = e
        out = {}
        for key, e in top.items():
            f = bottom.get(key)
            if f is not None:
                out[(e, +1)] = f
                out[(f, -1)] = e
        return out

    def _same_ply(self, e: int, f: int) -> bool:
        m = self.mesh
        return m.material_ids[e] == m.material_ids[f] and m.theta[e] == m.theta[f] and m.alpha[e] == m.alpha[f]

    def ply_interfaces(self) -> List[Tuple[int, int]]:
        """``(below, above)`` element pairs of a layered ``Q8`` mesh whose shared edge is a ply
        interface (a change of material, ply angle or wall direction)."""
        nb = self._row_neighbours
        return sorted((e, f) for (e, side), f in nb.items() if side == +1 and not self._same_ply(e, f))

    def _side_fit(self, e: int, eta_edge: float, xi: float, forces, benscoter) -> Tuple[np.ndarray, np.ndarray]:
        """Interlaminar stresses (3,) of the ply below or above an interface, at the interface:
        a quadratic in the distance from it through the values at the Gauss lines of the two
        rows of that ply nearest the interface (a linear fit with one row), and the point."""
        rows = [e]
        f = self._row_neighbours.get((e, -int(eta_edge)))
        if f is not None and self._same_ply(e, f):
            rows.append(f)
        p0 = self.stress_at(e, xi, eta_edge, forces=forces, benscoter=benscoter)
        x0 = np.array([p0.y, p0.z])
        g = 1.0 / np.sqrt(3.0)
        d, v = [], []
        for r in rows:
            for eta in (-g, g):
                q = self.stress_at(r, xi, eta, forces=forces, benscoter=benscoter)
                d.append(float(np.hypot(q.y - x0[0], q.z - x0[1])))
                v.append(q.interlaminar)
        d, v = np.asarray(d), np.asarray(v)
        coef = np.polyfit(d, v, min(d.size - 1, 2))
        return coef[-1], x0

    def interface_stress(self, below: int, above: int, xi: float = 0.0, forces: Optional[npt.ArrayLike] = None,
                         benscoter: Optional[npt.ArrayLike] = None) -> Tuple[float, float, np.ndarray]:
        """``(y, z, [sigma_nn, tau_sn, tau_xn])`` on the interface between elements ``below`` and
        ``above`` (``ply_interfaces``) at ``xi`` along it, in the wall axes of ``below``.

        Each ply's stresses are fitted through its two rows nearest the interface and taken to
        the interface; the two plies' values are averaged. With two rows per ply this is exact
        for the quadratic shear of a layered beam, where the raw edge values of the elements
        need six rows per ply for 2 %.
        """
        a, x0 = self._side_fit(below, 1.0, xi, forces, benscoter)
        same = int(self.mesh.elements[below][3]) == int(self.mesh.elements[above][0])
        b, _ = self._side_fit(above, -1.0, xi if same else -xi, forces, benscoter)
        return float(x0[0]), float(x0[1]), 0.5 * (a + b)

    def interface_stresses(self, forces: Optional[npt.ArrayLike] = None, benscoter: Optional[npt.ArrayLike] = None,
                           xi: Sequence[float] = (0.0,)) -> List[Tuple[int, int, float, float, np.ndarray]]:
        """``interface_stress`` on every ply interface: ``(below, above, y, z, stresses)``."""
        out = []
        for e, f in self.ply_interfaces():
            for x in xi:
                y, z, s = self.interface_stress(e, f, x, forces=forces, benscoter=benscoter)
                out.append((e, f, y, z, s))
        return out

    # ------------------------------------------------------------------ #
    # Second-order terms, axial quadrature and mass
    # ------------------------------------------------------------------ #
    @cached_property
    def geometric_terms(self) -> npt.NDArray[np.float64]:
        """Second-order section terms (7, 6) per unit section force, order ``GEOMETRIC_TERMS``
        (as ``ThinWalledSectionAnalysis.geometric_terms``), from the 3D stresses at ``x = 0``."""
        sol = self._solution
        H = np.zeros((len(GEOMETRIC_TERMS), 6))
        for p in self._points:
            H += self._geometric_rows(p, p.C @ self._phi(p, sol["P0"], sol["W0"], sol["W1"]))
        return H

    @staticmethod
    def _geometric_rows(p, s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Weighted ``GEOMETRIC_TERMS`` rows (7, n) at a point for stresses ``s`` (6, n)."""
        sxx, txz, txy = s[_XX], s[_XZ], s[_XY]
        y, z = p.y, p.z
        return p.weight * np.vstack([sxx, sxx * y, sxx * z, sxx * (y * y + z * z),
                                     txy, txz, txy * y + txz * z])

    @property
    def axial_points(self) -> npt.NDArray[np.float64]:
        """Quadrature ``(m, 3)`` of ``[y, z, k]`` with ``sum k f = int E_x f dA``."""
        return np.array([[p.y, p.z, p.weight * self._axial_modulus[p.element]] for p in self._points])

    @cached_property
    def mass_properties(self) -> Dict[str, float]:
        """Mass per length, mass centre and mass moments per length about the origin
        (keys as ``ThinWalledSectionAnalysis.mass_properties``)."""
        mesh = self.mesh
        t = dict(m=0.0, my=0.0, mz=0.0, yy=0.0, zz=0.0, yz=0.0)
        for p in self._points:
            w = p.weight * mesh.materials[mesh.material_ids[p.element]].rho
            t["m"] += w
            t["my"] += w * p.y
            t["mz"] += w * p.z
            t["yy"] += w * p.z * p.z
            t["zz"] += w * p.y * p.y
            t["yz"] += w * p.y * p.z
        m = t["m"]
        return {"m": m, "y_m": t["my"] / m if m > 0 else 0.0, "z_m": t["mz"] / m if m > 0 else 0.0,
                "I_yy": t["yy"], "I_zz": t["zz"], "I_yz": t["yz"], "I_p": t["yy"] + t["zz"]}
