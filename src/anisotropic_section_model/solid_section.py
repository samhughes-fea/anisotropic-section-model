"""``SolidSection``: a 2D finite-element section in the package's public frame.

``solid.SolidSectionAnalysis`` (the Giavotto / BECAS method on a 2D mesh) works in the
engine frame (beam axis x, section axes (y, z), ``axes.ENGINE_STRAINS``).
``SolidSection`` gives the same results in the public frame (``axes``): beam axis z,
section axes (x, y), BeamDyn's order ``[Fx, Fy, N, Mx, My, Mz]`` against
``[gamma_x, gamma_y, eps, kappa_x, kappa_y, kappa_z]``, as ``ThinWalledSection`` does
for wall mid-lines. Section coordinates are the same numbers in both frames, so a mesh
is built once and read either way; the engine analysis is kept in ``analysis``.

Public API
----------
SolidSection.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np

from .axes import from_engine, mass_matrix, stiffness_about, to_engine


class SolidSection:
    """A meshed beam section with its results in the public frame.

    ``mesh`` is a ``solid.SectionMesh``. Build one from a kernel cut with ``from_cut``,
    from a thin-walled geometry with ``SectionMesh.from_thin_walled``, or from Gmsh with
    ``SectionMesh.from_gmsh``.
    """

    def __init__(self, mesh) -> None:
        from .solid.section_analysis import SolidSectionAnalysis

        self.mesh = mesh
        self.analysis = SolidSectionAnalysis(mesh)

    @classmethod
    def from_cut(cls, cut, laminates: Mapping[str, object], **options) -> "SolidSection":
        """The layered mesh of an implicit-geometry-kernel ``SectionCut``
        (``solid.from_geometry.section_mesh_from_cut``; keywords go to it)."""
        from .solid.from_geometry import section_mesh_from_cut

        return cls(section_mesh_from_cut(cut, laminates, **options))

    # ------------------------------------------------------------ stiffness
    def timoshenko_stiffness(self, reference=(0.0, 0.0)) -> np.ndarray:
        """6 x 6 Timoshenko stiffness about ``reference`` (x, y), fully coupled, in
        BeamDyn's order (``axes.STRAINS``), as ``ThinWalledSection.timoshenko_stiffness``."""
        return stiffness_about(self.analysis.stiffness, reference)

    def mass_matrix(self, reference=(0.0, 0.0)) -> np.ndarray:
        """6 x 6 sectional mass matrix about ``reference``, as
        ``ThinWalledSection.mass_matrix`` (BeamDyn's order and form)."""
        return mass_matrix(self.analysis.mass_properties, reference)

    def strain_response(self, forces) -> np.ndarray:
        """Public strains for public ``forces`` ``[Fx, Fy, N, Mx, My, Mz]`` about the origin."""
        return from_engine(self.analysis.strain_response(to_engine(np.asarray(forces, dtype=float))))

    # ------------------------------------------------------------ centres and torsion
    @property
    def shear_centre(self) -> Tuple[float, float]:
        """Point (x, y) where a transverse force causes no twist."""
        return self.analysis.shear_centre

    @property
    def tension_centre(self) -> Tuple[float, float]:
        """Point (x, y) where an axial force causes no bending curvature."""
        return self.analysis.tension_centre

    @property
    def GJ_free(self) -> float:
        """Torsional stiffness with the other forces free, 1 / F[kappa_z, kappa_z], as
        ``ThinWalledSection.GJ_free``."""
        return float(1.0 / np.linalg.inv(self.timoshenko_stiffness())[5, 5])

    @property
    def EGamma(self) -> float:
        """Warping constant EΓ (``SolidSectionAnalysis.warping_constant``)."""
        return float(self.analysis.warping_constant)

    def point_stresses(self, forces) -> list:
        """3D states at the quadrature points for public ``forces`` ``[Fx, Fy, N, Mx, My, Mz]``.

        Each is a ``solid.section_analysis.PointStress``, whose section-axes Voigt
        stresses are in the engine's 3D axes (``xx`` along the beam); its material and
        wall axes are the same in both frames.
        """
        return self.analysis.point_stresses(to_engine(np.asarray(forces, dtype=float)))


__all__ = ["SolidSection"]
