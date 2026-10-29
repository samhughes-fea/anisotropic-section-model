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

Public API
----------
SolidSectionAnalysis.

See Also
--------
docs/element_library/solid_section_analysis.md
"""

from __future__ import annotations

from functools import cached_property
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pre_processing.section_integration.solid.mesh import SectionMesh
from pre_processing.section_integration.thin_walled.section_analysis import CLASSICAL, GEOMETRIC_TERMS, T_RATE

__all__ = ["SolidSectionAnalysis"]

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
        """Vlasov warping stiffness ``E Gamma = int E_x omega^2 dA`` (``E_x`` as ``axial_points``)."""
        omega, c = self._torsion_warping
        total = 0.0
        for p in self._points:
            om = p.N @ omega[p.dofs[0::3]] - (c[0] + c[1] * p.z - c[2] * p.y)
            total += p.weight * self._axial_modulus[p.element] * om * om
        return float(total)

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
            s = p.C @ self._phi(p, sol["P0"], sol["W0"], sol["W1"])
            sxx, txz, txy = s[_XX], s[_XZ], s[_XY]
            y, z = p.y, p.z
            H += p.weight * np.vstack([sxx, sxx * y, sxx * z, sxx * (y * y + z * z),
                                       txy, txz, txy * y + txz * z])
        return H

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
