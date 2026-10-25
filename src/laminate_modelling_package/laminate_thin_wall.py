"""Thin-walled section theory for blade cross-sections: Saint-Venant torsion and warping.

A blade or spar cross-section is a network of thin laminate walls: skins, spar caps,
shear webs and a trailing-edge web. ``ThinWalledSection`` describes it as straight
walls between section points (curved skins are split into short walls) and gives the
section properties of classical thin-walled beam theory, for open, closed and
multi-cell sections:

- axial and bending stiffness about the modulus-weighted centroid
- Saint-Venant torsion: the shear flow in every wall per unit twist rate (Bredt-Batho
  for any number of cells, solved as one network problem) and GJ, including the plate
  twisting stiffness of each wall
- the warping function omega(s) of Saint-Venant torsion (the sectorial coordinate on
  open branches, corrected by the shear flows in closed cells, as in Benscoter's
  theory), the shear centre (the pole that makes omega orthogonal to bending) and the
  warping constant EΓ
- the warping shear stiffness S = Ip - J that measures how far a closed section is from
  the Vlasov limit of no shear deformation in warping

These feed ``RestrainedTorsionBeam`` (non-uniform torsion with a bimoment) and
``GeneralizedBeam`` (GBT, which adds the cross-section's own distortion).

Axes: z along the beam, (x, y) in the section. A wall runs from ``start`` to ``end``;
its contour coordinate s points that way and its laminate normal is z x s (the left of
the wall seen from +z, looking along s). Ply angles are measured from the beam axis
about that normal, so a wall cut from a shell model with the beam axis as the material
axis keeps its plies. The laminate ABD of a wall is taken about its own mid-plane (any
offset is used only to place the wall).

Wall stiffness: membrane resultants act with no hoop stress (N_ss = 0) and the wall
free to bend, which gives the effective axial stiffness Et = 1 / a11 and membrane
shear stiffness Gt = 1 / a66 used here (a = inverse ABD), and the plate stiffnesses
D11 = 1 / d11 (secondary warping) and D66 = 1 / d66 (wall twisting). Couplings
between extension, shear and bending of an unbalanced wall are carried in full by
``GeneralizedBeam``; the closed-form properties here use the uncoupled values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .laminate_kinematics import integrate_abd
from .laminate_mechanics import Laminate


@dataclass(frozen=True)
class Wall:
    """A straight wall between section points ``start`` and ``end`` (indices) with a laminate."""

    start: int
    end: int
    laminate: Laminate
    name: str = ""


def condensed_wall_stiffness(laminate: Laminate) -> np.ndarray:
    """5 x 5 wall stiffness with no hoop stress (N_ss = 0).

    Order [eps_zz, gamma_zs, kappa_zz, kappa_ss, kappa_zs] against
    [N_zz, N_zs, M_zz, M_ss, M_zs], about the laminate mid-plane, with x of the
    laminate along the beam axis and y along the contour.
    """
    K = integrate_abd(Laminate(laminate.plies, laminate.name))
    keep = [0, 2, 3, 4, 5]
    return K[np.ix_(keep, keep)] - np.outer(K[keep, 1], K[1, keep]) / K[1, 1]


def _line_integral(fi, fj, gi, gj, b):
    """Integral of the product of two functions linear along a wall of length b."""
    return b / 6.0 * (2.0 * fi * gi + fi * gj + fj * gi + 2.0 * fj * gj)


class ThinWalledSection:
    """A thin-walled cross-section: section points and the laminate walls joining them.

    Parameters
    ----------
    points:
        (n, 2) wall-midline coordinates (x, y).
    walls:
        ``Wall`` objects, or (start, end, laminate[, name]) tuples.
    """

    def __init__(self, points, walls: Sequence):
        self.points = np.asarray(points, dtype=float).reshape(-1, 2)
        ws = []
        for w in walls:
            ws.append(w if isinstance(w, Wall) else Wall(int(w[0]), int(w[1]), w[2], w[3] if len(w) > 3 else ""))
        if not ws:
            raise ValueError("a section needs at least one wall")
        for w in ws:
            if not (0 <= w.start < len(self.points) and 0 <= w.end < len(self.points)) or w.start == w.end:
                raise ValueError(f"wall {w} needs two different section points")
        self.walls: Tuple[Wall, ...] = tuple(ws)
        n = len(self.points)
        used = sorted({w.start for w in ws} | {w.end for w in ws})
        if len(used) != n:
            raise ValueError("every section point must belong to a wall")
        # connectivity check (a single connected network)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for w in ws:
            parent[find(w.start)] = find(w.end)
        if len({find(i) for i in range(n)}) != 1:
            raise ValueError("the walls must form one connected section")
        self.n_cells = len(ws) - n + 1  # independent closed loops (cycle rank)

        P = self.points
        self.start = np.array([w.start for w in ws])
        self.end = np.array([w.end for w in ws])
        d = P[self.end] - P[self.start]
        self.length = np.linalg.norm(d, axis=1)
        if np.any(self.length <= 0.0):
            raise ValueError("walls need a positive length")
        self.tangent = d / self.length[:, None]
        self.normal = np.column_stack([-self.tangent[:, 1], self.tangent[:, 0]])  # z x s
        self.stiffness = np.array([condensed_wall_stiffness(w.laminate) for w in ws])  # (n_walls, 5, 5)
        compliances = []
        for w in ws:
            compliances.append(np.linalg.inv(integrate_abd(Laminate(w.laminate.plies))))
        F = np.array(compliances)
        self.Et = 1.0 / F[:, 0, 0]
        self.Gt = 1.0 / F[:, 2, 2]
        self.D11 = 1.0 / F[:, 3, 3]
        self.D66 = 1.0 / F[:, 5, 5]
        self.thickness = np.array([w.laminate.thickness for w in ws])
        self.areal_density = np.array([sum(p.thickness * p.material.density for p in w.laminate.plies) for w in ws])
        self._solve()

    # ------------------------------------------------------------ construction
    @classmethod
    def from_segments(cls, segments: Sequence, tol: float = 1e-9) -> "ThinWalledSection":
        """Build from (p0, p1, laminate[, name]) segments, merging coincident end points."""
        pts: List[np.ndarray] = []
        walls = []

        def index(p):
            p = np.asarray(p, dtype=float)
            for k, q in enumerate(pts):
                if np.linalg.norm(p - q) <= tol:
                    return k
            pts.append(p)
            return len(pts) - 1

        for seg in segments:
            i, j = index(seg[0]), index(seg[1])
            walls.append(Wall(i, j, seg[2], seg[3] if len(seg) > 3 else ""))
        return cls(np.array(pts), walls)

    @classmethod
    def from_blade_mesh(cls, blade, z: Optional[float] = None, tol: float = 1e-9) -> "ThinWalledSection":
        """The section of a ``BladeMesh`` (or any shell ``FEAMesh`` along z) at mesh section ``z``.

        Every shell element edge that lies on the section plane becomes one wall per
        element edge segment (two for quadratic elements), placed on the laminate
        mid-surface (the mesh node moved by the laminate offset along the element
        normal, averaged over the elements at each node) and oriented so that its
        laminate normal is the element normal. Wall names are the element set names
        of ``BladeMesh`` (``spar_cap_ss``, ``web_fwd``, ...) when available.
        """
        mesh = getattr(blade, "mesh", blade)
        if z is None:
            zs = np.asarray(getattr(blade, "span_positions", []), dtype=float)
            if zs.size == 0:
                raise ValueError("give z for a plain mesh")
            z = float(zs[int(np.argmin(np.abs(zs - 0.5 * (zs[0] + zs[-1]))))])
        names: Dict[int, str] = {}
        sets = getattr(blade, "element_sets", {})
        generic = {"all", "skin", "webs"}
        for name, ids in sets.items():
            if name in generic:
                continue
            for e in ids:
                # the most specific set name wins (region_ss over region)
                if e not in names or len(name) > len(names[e]):
                    names[e] = name
        scale = max(1.0, abs(z))
        raw = []  # (node_a, node_b, laminate, name, normal)
        shifts: Dict[int, List[np.ndarray]] = {}
        for e in mesh.shell_elements:
            ids = list(e.node_ids)
            X = np.array([mesh.node_coordinates(n) for n in ids])
            on = np.abs(X[:, 2] - z) <= tol * scale
            if on.sum() < 2:
                continue
            frame = np.asarray(e.frame)
            if abs(frame[0, 2]) < 0.99:
                raise ValueError(f"element {e.element_id}: its material axis is not the beam axis (z)")
            n3 = frame[2]
            nc = 4 if len(ids) in (4, 8) else 3
            for k in range(nc):
                a, b = k, (k + 1) % nc
                if not (on[a] and on[b]):
                    continue
                chain = [ids[a]]
                if len(ids) == 2 * nc:
                    chain.append(ids[nc + k])
                chain.append(ids[b])
                for p, q in zip(chain[:-1], chain[1:]):
                    raw.append((p, q, e.laminate, names.get(e.element_id, ""), n3))
                for n in chain:
                    shifts.setdefault(n, []).append(e.laminate.mid_plane * n3)
        if not raw:
            raise ValueError(f"no shell element edges lie on the section z = {z}")
        order = sorted(shifts)
        index = {n: i for i, n in enumerate(order)}
        P = np.array([mesh.node_coordinates(n)[:2] + np.mean(shifts[n], axis=0)[:2] for n in order])
        walls = []
        seen = set()
        for p, q, lam, name, n3 in raw:
            key = (min(p, q), max(p, q))
            if key in seen:
                continue
            seen.add(key)
            i, j = index[p], index[q]
            t = P[j] - P[i]
            if t[0] * n3[1] - t[1] * n3[0] < 0.0:  # (z x t) . n3 < 0: run the wall the other way
                i, j = j, i
            walls.append(Wall(i, j, lam, name))
        section = cls(P, walls)
        section.mesh_nodes = order  # mesh node id of each section point
        section.z = float(z)
        return section

    # ------------------------------------------------------------ analysis
    def _integrals(self, fi, fj, gi=None, gj=None, weight=None):
        w = np.ones(len(self.walls)) if weight is None else weight
        if gi is None:
            return float(np.sum(w * 0.5 * (fi + fj) * self.length))
        return float(np.sum(w * _line_integral(fi, fj, gi, gj, self.length)))

    def _solve(self) -> None:
        P, i, j, b = self.points, self.start, self.end, self.length
        x, y = P[:, 0], P[:, 1]
        Et, Gt = self.Et, self.Gt
        self.EA = float(np.sum(Et * b))
        xc = self._integrals(x[i], x[j], weight=Et) / self.EA
        yc = self._integrals(y[i], y[j], weight=Et) / self.EA
        self.centroid = (xc, yc)
        X, Y = x - xc, y - yc
        self.EI_xx = self._integrals(Y[i], Y[j], Y[i], Y[j], Et)  # about the x axis: integral Et y^2
        self.EI_yy = self._integrals(X[i], X[j], X[i], X[j], Et)
        self.EI_xy = self._integrals(X[i], X[j], Y[i], Y[j], Et)
        self.mass_per_length = float(np.sum(self.areal_density * b))

        # Saint-Venant warping about the origin: minimise sum Gt b (r_n - d omega/ds)^2
        r0 = x[i] * self.tangent[:, 1] - y[i] * self.tangent[:, 0]
        n = len(P)
        c = Gt / b
        L = np.zeros((n, n))
        g = np.zeros(n)
        for k in range(len(b)):
            a, e = i[k], j[k]
            L[a, a] += c[k]
            L[e, e] += c[k]
            L[a, e] -= c[k]
            L[e, a] -= c[k]
            g[e] += Gt[k] * r0[k]
            g[a] -= Gt[k] * r0[k]
        L[0, :] = 0.0
        L[0, 0] = 1.0
        g[0] = 0.0
        omega0 = np.linalg.solve(L, g)
        q = Gt * (r0 - (omega0[j] - omega0[i]) / b)
        q[np.abs(q) < 1e-12 * max(np.max(np.abs(Gt * r0)), 1e-300)] = 0.0
        self.shear_flow = q  # per unit twist rate (N/m per rad/m); independent of the pole
        self.J_membrane = float(np.sum(q**2 * b / Gt))
        self.GJ_walls = float(np.sum(4.0 * self.D66 * b))
        self.GJ = self.J_membrane + self.GJ_walls

        # shear centre: the pole that makes omega orthogonal to x and y
        # omega_P = omega_0 - xP y + yP x + const
        Iwx = self._integrals(omega0[i], omega0[j], X[i], X[j], Et)
        Iwy = self._integrals(omega0[i], omega0[j], Y[i], Y[j], Et)
        A = np.array([[-self.EI_xy, self.EI_yy], [-self.EI_xx, self.EI_xy]])
        xs, ys = np.linalg.solve(A, [-Iwx, -Iwy])
        self.shear_centre = (float(xs), float(ys))
        omega = omega0 - xs * y + ys * x
        omega -= self._integrals(omega[i], omega[j], weight=Et) / self.EA
        self.omega = omega
        self.EGamma_primary = self._integrals(omega[i], omega[j], omega[i], omega[j], Et)
        rt_i = (x[i] - xs) * self.tangent[:, 0] + (y[i] - ys) * self.tangent[:, 1]
        rt_j = rt_i + b
        self.EGamma_secondary = self._integrals(rt_i, rt_j, rt_i, rt_j, self.D11)
        self.EGamma = self.EGamma_primary + self.EGamma_secondary
        self.r_n = (x[i] - xs) * self.tangent[:, 1] - (y[i] - ys) * self.tangent[:, 0]  # about the shear centre
        self.Ip = float(np.sum(Gt * b * self.r_n**2))
        self.S_warping = max(self.Ip - self.J_membrane, 0.0)

    # ------------------------------------------------------------ derived values
    @property
    def benscoter_mu(self) -> float:
        """Benscoter's warping shear parameter mu = 1 - J / Ip (0: no warping, 1: Vlasov)."""
        return float(self.S_warping / self.Ip) if self.Ip > 0 else 1.0

    @property
    def decay_length(self) -> float:
        """1 / k: the length over which restrained warping dies out, k^2 = mu GJ / EΓ (Benscoter)."""
        k2 = self.GJ * self.S_warping / (self.EGamma * (self.GJ + self.S_warping)) if self.S_warping > 0 else \
            self.GJ / self.EGamma
        return float(1.0 / np.sqrt(k2))

    def warping_stress_resultant(self, bimoment: float) -> np.ndarray:
        """N_zz at each section point for a bimoment (warping normal stress resultant, N/m).

        The value at a point uses the Et of the walls meeting there (averaged); see
        ``wall_warping_resultant`` for per-wall values.
        """
        Et_node = np.zeros(len(self.points))
        cnt = np.zeros(len(self.points))
        for k in range(len(self.walls)):
            for p in (self.start[k], self.end[k]):
                Et_node[p] += self.Et[k]
                cnt[p] += 1
        return bimoment * Et_node / cnt * self.omega / self.EGamma

    def wall_warping_resultant(self, bimoment: float) -> Tuple[np.ndarray, np.ndarray]:
        """N_zz at the start and end of every wall for a bimoment."""
        f = bimoment / self.EGamma * self.Et
        return f * self.omega[self.start], f * self.omega[self.end]

    # ------------------------------------------------------------ beam stiffness matrices
    def _axial_bending_stiffness(self, reference) -> np.ndarray:
        """3 x 3 stiffness of [N, Mx, My] against [eps, kappa_x, kappa_y] about ``reference``.

        Plane sections: eps_zz = eps + kappa_x y - kappa_y x (coordinates from the reference).
        """
        P, i, j = self.points, self.start, self.end
        xr, yr = reference
        f = [np.ones(len(P)), P[:, 1] - yr, -(P[:, 0] - xr)]
        K = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                K[a, b] = self._integrals(f[a][i], f[a][j], f[b][i], f[b][j], self.Et)
        return K

    def shear_flow_basis(self, reference=(0.0, 0.0)) -> Dict[str, np.ndarray]:
        """Shear flows of transverse shear, as values at three Gauss points on every wall.

        Returns ``open_x``, ``open_y`` (n_walls, 3): a shear flow in equilibrium with the
        axial stress gradient of a unit shear force Fx or Fy (bending varying along z,
        plane sections, about ``reference``), with no flow out of free edges; ``loops``
        (n_cells, n_walls): the closed-cell flows that can be added to it (constant along
        each wall, positive along the wall direction); and the Gauss data ``s`` (n_walls,
        3) and ``w`` (n_walls, 3) for integrating along the walls.
        """
        P, i, j, b = self.points, self.start, self.end, self.length
        n_pts, n_w = len(P), len(b)
        Kab = self._axial_bending_stiffness(reference)
        Cab = np.linalg.inv(Kab)
        xr, yr = reference
        X, Y = P[:, 0] - xr, P[:, 1] - yr
        gp, gw = np.polynomial.legendre.leggauss(3)
        s = 0.5 * (gp[None, :] + 1.0) * b[:, None]
        w = 0.5 * gw[None, :] * b[:, None]
        A = np.zeros((n_pts, n_w))  # flow balance: in at the wall end, out at its start
        A[j, np.arange(n_w)] += 1.0
        A[i, np.arange(n_w)] -= 1.0
        out = {"s": s, "w": w}
        # d/dz of [N, Mx, My] for unit shear: Mx' = Fy, My' = -Fx
        for key, dM in (("open_x", np.array([0.0, 0.0, -1.0])), ("open_y", np.array([0.0, 1.0, 0.0]))):
            de = Cab @ dM
            f_node_i = de[0] + de[1] * Y[i] - de[2] * X[i]
            f_node_j = de[0] + de[1] * Y[j] - de[2] * X[j]
            fi, fj = self.Et * f_node_i, self.Et * f_node_j  # dN_zz/dz at the wall ends
            G = 0.5 * b * (fi + fj)  # integral of dN_zz/dz along each wall
            rhs = np.zeros(n_pts)
            np.add.at(rhs, j, G)
            q0 = np.linalg.lstsq(A, rhs, rcond=None)[0]
            out[key] = q0[:, None] - (fi[:, None] * s + (fj - fi)[:, None] * s**2 / (2.0 * b[:, None]))
        # closed-cell flows: the null space of the balance matrix
        _, sv, Vt = np.linalg.svd(A)
        rank = int(np.sum(sv > 1e-10 * sv[0]))
        out["loops"] = Vt[rank:]
        return out

    def shear_torsion_compliance(self, reference=(0.0, 0.0)) -> np.ndarray:
        """3 x 3 compliance of [Fx, Fy, Mz] to [gamma_x, gamma_y, kappa_z] about ``reference``.

        Minimum complementary energy of the wall shear flows (sum of the integral of
        q^2 / Gt) plus the walls' own twisting (Mz carried by the plate twisting
        stiffness GJ_walls), with the closed-cell flows free and the torque in
        equilibrium. The section warps freely (Saint-Venant torsion, Timoshenko shear).
        """
        basis = self.shear_flow_basis(reference)
        w = basis["w"]
        loops = basis["loops"]
        n_c = loops.shape[0]
        xr, yr = reference
        P, i = self.points, self.start
        r_n = (P[i, 0] - xr) * self.tangent[:, 1] - (P[i, 1] - yr) * self.tangent[:, 0]
        # columns of q (n_walls, 3 Gauss points): Fx, Fy, Mz (no flow), loops
        cols = [basis["open_x"], basis["open_y"], np.zeros_like(w)] + [np.repeat(l[:, None], 3, axis=1) for l in loops]
        Q = np.stack([c.ravel() for c in cols], axis=1)
        Fq = (w / self.Gt[:, None]).ravel()
        H = Q.T @ (Fq[:, None] * Q)
        # torque carried by the walls' twisting: T_w = Mz - torque of the shear flows
        torque = np.array([float(np.sum(c * w * r_n[:, None])) for c in cols])
        tw = -torque
        tw[2] = 1.0
        H += np.outer(tw, tw) / self.GJ_walls
        if n_c:
            Hff, Hfc, Hcc = H[:3, :3], H[:3, 3:], H[3:, 3:]
            return Hff - Hfc @ np.linalg.solve(Hcc, Hfc.T)
        return H

    def timoshenko_stiffness(self, reference=(0.0, 0.0)) -> np.ndarray:
        """6 x 6 Timoshenko stiffness about ``reference`` (in the section axes).

        Order (as BeamDyn and the windIO files): [Fx, Fy, N, Mx, My, Mz] against
        [gamma_x, gamma_y, eps, kappa_x, kappa_y, kappa_z], with gamma_x = u_x' - theta_y
        and gamma_y = u_y' + theta_x. Thin-wall theory: the axial and bending block comes
        from plane sections with each wall's Et, the shear and torsion block from the
        shear flows (``shear_torsion_compliance``). Couplings between the two blocks
        from unbalanced walls are not included (``beam_section_properties`` measures
        them on the shell model).
        """
        K = np.zeros((6, 6))
        K[2:5, 2:5] = self._axial_bending_stiffness(reference)
        st = np.linalg.inv(self.shear_torsion_compliance(reference))
        idx = [0, 1, 5]
        K[np.ix_(idx, idx)] = 0.5 * (st + st.T)
        return K

    def vlasov_stiffness(self, reference=(0.0, 0.0)) -> np.ndarray:
        """7 x 7 Timoshenko-Vlasov stiffness: ``timoshenko_stiffness`` plus the warping DOF.

        The seventh strain is the warping amplitude's gradient, kappa_z' in Vlasov's
        theory (or phi' in Benscoter's), and its force is the bimoment B. The warping
        function is taken about the shear centre and orthogonal to extension and bending,
        so in thin-wall theory B is uncoupled from the other forces and K[6, 6] = EΓ.
        ``S_warping`` is the extra shear stiffness Benscoter's theory needs for a closed
        section.
        """
        K = np.zeros((7, 7))
        K[:6, :6] = self.timoshenko_stiffness(reference)
        K[6, 6] = self.EGamma
        return K

    def contour(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """(start, end) point pairs of the walls, for plotting."""
        return [(self.points[a], self.points[b]) for a, b in zip(self.start, self.end)]

    def summary(self) -> Dict[str, float]:
        return {
            "EA": self.EA, "EI_xx": self.EI_xx, "EI_yy": self.EI_yy, "GJ": self.GJ,
            "GJ_walls": self.GJ_walls, "EGamma": self.EGamma, "EGamma_secondary": self.EGamma_secondary,
            "Ip": self.Ip, "S_warping": self.S_warping, "mu": self.benscoter_mu,
            "shear_centre_x": self.shear_centre[0], "shear_centre_y": self.shear_centre[1],
            "decay_length": self.decay_length, "cells": self.n_cells,
        }

    def __repr__(self) -> str:
        return (f"ThinWalledSection({len(self.points)} points, {len(self.walls)} walls, {self.n_cells} cells, "
                f"GJ={self.GJ:.4g}, EGamma={self.EGamma:.4g})")


__all__ = ["Wall", "ThinWalledSection", "condensed_wall_stiffness"]
