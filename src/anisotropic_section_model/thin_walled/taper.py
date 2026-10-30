# pre_processing/section_integration/thin_walled/taper.py
"""Taper coupling of a thin-walled section: the Resal effect.

In a tapered beam the walls are not parallel to the beam axis. A wall point at ``r(x) =
(y(x), z(x))`` in the section frame runs along a fibre inclined at ``r' = dr/dx``. When the
section translates and rotates rigidly, that fibre stretches by the shear strains too::

    eps_fibre = eps_x + z kappa_y - y kappa_z + y' gamma_xy + z' gamma_xz

(right-handed section strains, ``thin_walled.section_analysis``). With the wall's axial
membrane stiffness ``A`` (hoop free, ``beam_wall_stiffness``) the energy ``1/2 int A
eps_fibre^2 ds`` adds to the prismatic 6 x 6 section stiffness::

    K[a, g] += C[a, g] = int A f_a r'_g ds,     f = (1, z, -y)   for a in (eps_x, kappa_y, kappa_z)
    K[g, h] += Q[g, h] = int A r'_g r'_h ds                      for g, h in (gamma_xy, gamma_xz)

and ``C`` symmetric. The coupling is first order in the taper. It is the Resal effect: the
inclined caps of a tapered beam carry part of the shear force, so the webs carry less and the
beam deflects less in shear than a prismatic beam of the same local section. Shear-rigid
(Euler-Bernoulli) elements do not see it.

For a box whose depth ``h`` changes at ``h'`` (walls at ``z = -+h/2``, so ``z' = (h'/h) z``)
the coupling is ``C[kappa_y, gamma_xz] = (h'/h) EI_y`` and ``Q = (h'/h)^2 EI_y``, and the webs'
shear strain is the classical tapered-beam result ``gamma_xz = (V_z - M_y h'/h) / GA``.

``section_taper`` takes two sections of the same topology (point ids and walls) a distance
apart, such as cuts close either side of an element's mid-point; ``section_taper_from_geometry``
cuts them from an implicit-geometry-kernel ``BladeGeometry``. The rates are those of the
material fibres in fixed axes, so a section that turns along the beam (pretwist ``k``) adds
``r' = k (-z, y)``: the helical fibres' share of the same coupling. Their stretch under a twist
rate is ``SectionAnalysis.with_pretwist``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import numpy.typing as npt

from pre_processing.section_integration.thin_walled.laminate import WallLaminate
from pre_processing.section_integration.thin_walled.thin_walled_geometry import ThinWalledSection
from pre_processing.section_integration.thin_walled.wall_stiffness import beam_wall_stiffness

__all__ = ["SectionTaper", "section_taper", "section_taper_from_geometry"]


@dataclass(frozen=True)
class SectionTaper:
    """Right-handed taper coupling of one section.

    Attributes
    ----------
    coupling : ndarray (3, 2)
        ``C``: rows ``(eps_x, kappa_y, kappa_z)``, columns ``(gamma_xy, gamma_xz)``.
    shear : ndarray (2, 2)
        ``Q`` added to the shear block.
    """

    coupling: npt.NDArray[np.float64]
    shear: npt.NDArray[np.float64]

    @staticmethod
    def zero() -> "SectionTaper":
        return SectionTaper(np.zeros((3, 2)), np.zeros((2, 2)))

    def matrix(self) -> npt.NDArray[np.float64]:
        """The 6 x 6 right-handed stiffness to add (zero outside the coupling and shear blocks)."""
        K = np.zeros((6, 6))
        K[0:3, 3:5] = self.coupling
        K[3:5, 0:3] = self.coupling.T
        K[3:5, 3:5] = self.shear
        return K

    def as_row(self) -> npt.NDArray[np.float64]:
        """``[C (row major, 6), Q_yy, Q_yz, Q_zz]``: the columns of ``section_taper.txt``."""
        return np.concatenate([self.coupling.ravel(), [self.shear[0, 0], self.shear[0, 1], self.shear[1, 1]]])

    @staticmethod
    def from_row(row) -> "SectionTaper":
        v = np.asarray(row, dtype=np.float64).ravel()
        if v.size != 9:
            raise ValueError(f"a taper row has 9 values, got {v.size}")
        Q = np.array([[v[6], v[7]], [v[7], v[8]]])
        return SectionTaper(v[:6].reshape(3, 2), Q)


def section_taper(
    start: ThinWalledSection,
    end: ThinWalledSection,
    laminates: Mapping[int, WallLaminate],
    distance: float,
    n_gauss: int = 3,
) -> SectionTaper:
    """Taper coupling of the section midway between ``start`` and ``end``.

    Parameters
    ----------
    start, end : ThinWalledSection
        The section at two positions along the beam, ``distance`` apart (``end`` further
        along ``x``), with the same point ids and walls. Each point is the same material
        point at the two positions.
    laminates : mapping int -> WallLaminate
        Wall laminates by id (those of ``start``).
    distance : float
    n_gauss : int
        Gauss points per wall (the integrand is cubic in the wall coordinate).

    Raises
    ------
    ValueError
        The two sections differ in topology.
    """
    if distance <= 0.0:
        raise ValueError("distance must be positive")
    if set(start.points) != set(end.points) or len(start.walls) != len(end.walls):
        raise ValueError("the two sections must have the same points and walls")
    xg, wg = np.polynomial.legendre.leggauss(n_gauss)
    C = np.zeros((3, 2))
    Q = np.zeros((2, 2))
    for wa, wb in zip(start.walls, end.walls):
        if (wa.start, wa.end) != (wb.start, wb.end):
            raise ValueError(f"wall {wa.wall_id} joins different points in the two sections")
        A = float(beam_wall_stiffness(laminates[wa.laminate_id]).matrix[0, 0])
        a0, a1 = start.points[wa.start], start.points[wa.end]
        b0, b1 = end.points[wb.start], end.points[wb.end]
        r0, r1 = 0.5 * (a0 + b0), 0.5 * (a1 + b1)
        d0, d1 = (b0 - a0) / distance, (b1 - a1) / distance
        half = 0.5 * float(np.linalg.norm(r1 - r0))
        for x, w in zip(xg, wg):
            t = 0.5 * (1.0 + x)
            y, z = (1.0 - t) * r0 + t * r1
            rp = (1.0 - t) * d0 + t * d1
            C += A * w * half * np.outer((1.0, z, -y), rp)
            Q += A * w * half * np.outer(rp, rp)
    return SectionTaper(C, 0.5 * (Q + Q.T))


def section_taper_from_geometry(
    geometry,
    z: float,
    laminates,
    step: float,
    library: Optional[Mapping[int, WallLaminate]] = None,
    thickness_tol: float = 0.01,
) -> SectionTaper:
    """Taper coupling at span ``z`` of an implicit-geometry-kernel blade, from cuts at ``z -+ step``.

    ``laminates`` and ``library`` are as for ``thin_walled_from_cut``. The two cuts must have
    the same wall topology, which a small ``step`` gives; if they do not the step is halved,
    up to four times.
    """
    from pre_processing.section_integration.thin_walled.from_geometry import thin_walled_from_cut

    h = float(step)
    for _ in range(5):
        a = thin_walled_from_cut(geometry.section_at(z - h), laminates, library, thickness_tol=thickness_tol)
        b = thin_walled_from_cut(geometry.section_at(z + h), laminates, library, thickness_tol=thickness_tol)
        try:
            return section_taper(a.section, b.section, a.laminates, 2.0 * h)
        except ValueError:
            h *= 0.5
    raise ValueError(f"cuts either side of z = {z} differ in topology; no taper coupling")
