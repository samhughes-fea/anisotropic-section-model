"""Non-uniform torsion of thin-walled beams: Saint-Venant twist plus restrained warping.

Saint-Venant torsion assumes the section warps freely: the twist rate is T / GJ
everywhere. Where warping is stopped (a clamped root, a stiff rib, a change of
section) axial warping stresses build up, described by the bimoment, and the torque
is shared between the Saint-Venant shear flows and a warping torque.

``RestrainedTorsionBeam`` solves Benscoter's theory, in which the warping amplitude
phi(z) is a field of its own (w(s, z) = -omega(s) phi(z)). The beam stores

    U = 1/2 integral [ GJ theta'^2 + S (theta' - phi)^2 + EΓ phi'^2 ] dz

with the section's Saint-Venant stiffness GJ, warping constant EΓ and warping shear
stiffness S (from ``ThinWalledSection``). Free warping makes phi = theta' and gives
back Saint-Venant torsion; S -> infinity is Vlasov's theory for open sections, where
the warping follows the twist rate exactly. The stress resultants are

    Saint-Venant torque  T_sv = GJ theta'
    warping torque       T_w  = S (theta' - phi) = dB/dz
    bimoment             B    = -EΓ phi'        (N_zz(s) = B Et(s) omega(s) / EΓ)

Closed-form cantilever solutions (``cantilever_tip_torque``) and a finite element
solution for any supports and torque distribution (``solve``) are provided; both use
the same sign conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TorsionResponse:
    """Twist, warping and the torque split along the beam."""

    z: np.ndarray
    twist: np.ndarray            # theta [rad]
    twist_rate: np.ndarray       # theta' [rad/m]
    warping: np.ndarray          # phi, the warping amplitude [rad/m]
    bimoment: np.ndarray         # B [N m^2]
    saint_venant_torque: np.ndarray
    warping_torque: np.ndarray

    @property
    def torque(self) -> np.ndarray:
        return self.saint_venant_torque + self.warping_torque


def _lagrange(nodes: np.ndarray, xi: float) -> Tuple[np.ndarray, np.ndarray]:
    """Lagrange shape functions and their derivatives at xi for the given nodes."""
    n = len(nodes)
    N = np.ones(n)
    dN = np.zeros(n)
    for a in range(n):
        for b in range(n):
            if b != a:
                N[a] *= (xi - nodes[b]) / (nodes[a] - nodes[b])
        s = 0.0
        for c in range(n):
            if c == a:
                continue
            term = 1.0 / (nodes[a] - nodes[c])
            for b in range(n):
                if b != a and b != c:
                    term *= (xi - nodes[b]) / (nodes[a] - nodes[b])
            s += term
        dN[a] = s
    return N, dN


class RestrainedTorsionBeam:
    """Benscoter (or Vlasov) torsion beam between z = 0 and ``length``.

    Parameters
    ----------
    GJ, EGamma:
        Saint-Venant torsional stiffness and warping constant.
    S:
        Warping shear stiffness (``ThinWalledSection.S_warping``); ``None`` gives
        Vlasov's theory (no shear deformation in warping).
    n_elements:
        Elements for ``solve`` (cubic twist, quadratic warping amplitude), graded
        towards both ends where the warping boundary layers sit.
    """

    def __init__(self, GJ: float, EGamma: float, length: float, S: Optional[float] = None,
                 n_elements: int = 80):
        if GJ <= 0 or EGamma <= 0 or length <= 0:
            raise ValueError("GJ, EGamma and length must be positive")
        self.GJ, self.EGamma, self.length = float(GJ), float(EGamma), float(length)
        self.S = None if S is None or not np.isfinite(S) else float(S)
        self.n_elements = int(n_elements)

    @classmethod
    def from_section(cls, section, length: float, vlasov: bool = False, n_elements: int = 80):
        """From a ``ThinWalledSection``; ``vlasov=True`` drops the warping shear flexibility."""
        return cls(section.GJ, section.EGamma, length, None if vlasov else section.S_warping, n_elements)

    # ------------------------------------------------------------ closed form
    @property
    def k(self) -> float:
        """Decay rate of restrained warping, k^2 = GJ S / (EΓ (GJ + S)) (GJ / EΓ for Vlasov)."""
        if self.S is None:
            return float(np.sqrt(self.GJ / self.EGamma))
        return float(np.sqrt(self.GJ * self.S / (self.EGamma * (self.GJ + self.S))))

    @property
    def rho(self) -> float:
        """S / (GJ + S): the share of the twist rate the warping restraint can remove (1 for Vlasov)."""
        return 1.0 if self.S is None else self.S / (self.GJ + self.S)

    def cantilever_tip_torque(self, torque: float, z=None, tip_warping: str = "free") -> TorsionResponse:
        """Closed form: root (z = 0) clamped (no twist, no warping), torque at the tip.

        ``tip_warping`` is ``"free"`` (no bimoment at the tip) or ``"restrained"``
        (a tip rib or end plate that stops warping, phi = 0).
        """
        L, k, rho, GJ = self.length, self.k, self.rho, self.GJ
        z = np.linspace(0.0, L, 201) if z is None else np.asarray(z, dtype=float)
        # phi GJ / T = f = 1 - cosh(k (z - c)) / cosh(k c): c = L for a free tip (phi' = 0
        # there), c = L / 2 for a restrained one (phi = 0 at both ends)
        if tip_warping == "free":
            c = L
        elif tip_warping == "restrained":
            c = 0.5 * L
        else:
            raise ValueError("tip_warping must be 'free' or 'restrained'")
        T = float(torque)
        a, b = np.abs(k * (z - c)), k * c
        ch = np.exp(a - b) * (1.0 + np.exp(-2.0 * a)) / (1.0 + np.exp(-2.0 * b))  # cosh(k(z-c)) / cosh(kc)
        sh = np.sign(z - c) * np.exp(a - b) * (1.0 - np.exp(-2.0 * a)) / (1.0 + np.exp(-2.0 * b))
        tb = (1.0 - np.exp(-2.0 * b)) / (1.0 + np.exp(-2.0 * b))  # tanh(kc)
        f = 1.0 - ch
        df = -k * sh
        ddf = -k * k * ch
        phi = T / GJ * f
        # theta' = (T + S phi) / (GJ + S) = T/GJ (1 - rho + rho f)
        rate = T / GJ * (1.0 - rho + rho * f)
        F = z - (sh + tb) / k  # integral of f from 0 to z
        twist = T / GJ * ((1.0 - rho) * z + rho * F)
        B = -self.EGamma * T / GJ * df
        Tw = -self.EGamma * T / GJ * ddf
        return TorsionResponse(z, twist, rate, phi, B, GJ * rate, Tw)

    def saint_venant(self, torque: float, z=None) -> TorsionResponse:
        """Free warping everywhere: theta = T z / GJ."""
        z = np.linspace(0.0, self.length, 201) if z is None else np.asarray(z, dtype=float)
        rate = np.full_like(z, torque / self.GJ)
        zero = np.zeros_like(z)
        return TorsionResponse(z, rate * z, rate, rate.copy(), zero, np.full_like(z, float(torque)), zero)

    # ------------------------------------------------------------ finite elements
    def _mesh(self) -> np.ndarray:
        n = self.n_elements
        u = np.linspace(0.0, 1.0, n + 1)
        return self.length * 0.5 * (1.0 - np.cos(np.pi * u))  # graded towards both ends

    def solve(self, tip_torque: float = 0.0, distributed_torque: float = 0.0,
              point_torques: Sequence[Tuple[float, float]] = (),
              supports: Optional[Dict[float, Tuple[bool, bool]]] = None, n_out: int = 201) -> TorsionResponse:
        """Finite element solution.

        ``supports`` maps a z position to (twist fixed, warping fixed); the default
        clamps the root, {0: (True, True)}. Supports and point torques snap to the
        nearest element end. ``distributed_torque`` is a uniform torque per length.
        """
        supports = {0.0: (True, True)} if supports is None else supports
        z_end = self._mesh()
        ne = len(z_end) - 1
        # twist: cubic (4 nodes per element), warping: quadratic (3 nodes per element)
        n_t = 3 * ne + 1
        n_p = 2 * ne + 1
        ndof = n_t + n_p
        K = np.zeros((ndof, ndof))
        F = np.zeros(ndof)
        xt = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])
        xp = np.array([-1.0, 0.0, 1.0])
        gp, gw = np.polynomial.legendre.leggauss(4)
        S = self.S if self.S is not None else 1e6 * self.GJ * max(1.0, self.EGamma / (self.GJ * self.length**2))
        for e in range(ne):
            h = z_end[e + 1] - z_end[e]
            it = np.arange(3 * e, 3 * e + 4)
            ip = n_t + np.arange(2 * e, 2 * e + 3)
            idx = np.concatenate([it, ip])
            ke = np.zeros((7, 7))
            fe = np.zeros(7)
            for xi, w in zip(gp, gw):
                Nt, dNt = _lagrange(xt, xi)
                Np, dNp = _lagrange(xp, xi)
                jac = 0.5 * h
                rate = np.concatenate([dNt / jac, np.zeros(3)])
                phi = np.concatenate([np.zeros(4), Np])
                dphi = np.concatenate([np.zeros(4), dNp / jac])
                ke += w * jac * (self.GJ * np.outer(rate, rate) + S * np.outer(rate - phi, rate - phi)
                                 + self.EGamma * np.outer(dphi, dphi))
                fe[:4] += w * jac * distributed_torque * Nt
            K[np.ix_(idx, idx)] += ke
            F[idx] += fe
        F[3 * ne] += tip_torque
        for zt, T in point_torques:
            F[3 * int(np.argmin(np.abs(z_end - zt)))] += T
        fixed = []
        for zs, (twist_fixed, warping_fixed) in supports.items():
            j = int(np.argmin(np.abs(z_end - zs)))
            if twist_fixed:
                fixed.append(3 * j)
            if warping_fixed:
                fixed.append(n_t + 2 * j)
        free = np.setdiff1d(np.arange(ndof), fixed)
        U = np.zeros(ndof)
        U[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

        z = np.linspace(0.0, self.length, n_out)
        out = {k: np.zeros(n_out) for k in ("t", "r", "p", "dp")}
        for m, zz in enumerate(z):
            e = int(np.clip(np.searchsorted(z_end, zz, side="right") - 1, 0, ne - 1))
            h = z_end[e + 1] - z_end[e]
            xi = 2.0 * (zz - z_end[e]) / h - 1.0
            Nt, dNt = _lagrange(xt, xi)
            Np, dNp = _lagrange(xp, xi)
            ut = U[3 * e: 3 * e + 4]
            upp = U[n_t + 2 * e: n_t + 2 * e + 3]
            out["t"][m] = Nt @ ut
            out["r"][m] = dNt @ ut * 2.0 / h
            out["p"][m] = Np @ upp
            out["dp"][m] = dNp @ upp * 2.0 / h
        B = -self.EGamma * out["dp"]
        Tsv = self.GJ * out["r"]
        Tw = S * (out["r"] - out["p"])
        return TorsionResponse(z, out["t"], out["r"], out["p"], B, Tsv, Tw)


__all__ = ["RestrainedTorsionBeam", "TorsionResponse"]
