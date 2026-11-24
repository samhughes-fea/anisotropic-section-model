"""Tapered sections: the Resal coupling of thin-walled and 2D-meshed sections, and the layup
derivatives of both. (Tapered beams, solved along the span, are 1d-fem's.)"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from anisotropic_section_model.section_analysis import SectionAnalysis
from anisotropic_section_model.solid.layup_sensitivity import solid_section_derivative
from anisotropic_section_model.solid.mesh import SectionMesh
from anisotropic_section_model.solid.section_analysis import SolidSectionAnalysis
from anisotropic_section_model.solid.taper import solid_section_taper
from anisotropic_section_model.thin_walled import ThinWalledSection, Wall, WallLaminate
from anisotropic_section_model.thin_walled.laminate import Ply, PlyMaterial
from anisotropic_section_model.thin_walled.layup_sensitivity import (LayupParameter, section_analysis_derivative,
                                                                     section_derivative)
from anisotropic_section_model.thin_walled.section_analysis import ThinWalledSectionAnalysis
from anisotropic_section_model.thin_walled.taper import section_taper, taper_derivative

CFRP = PlyMaterial(E1=135e9, E2=10e9, G12=5e9, nu12=0.3, G13=5e9, G23=3.8e9, rho=1550.0)
WIDTH, H0, H1, LENGTH = 0.2, 0.12, 0.06, 2.0


def _box(h, b=WIDTH):
    pts = {1: (-b / 2, -h / 2), 2: (b / 2, -h / 2), 3: (b / 2, h / 2), 4: (-b / 2, h / 2)}
    return ThinWalledSection(pts, [Wall(1, 1, 2, 1), Wall(2, 2, 3, 2), Wall(3, 3, 4, 1), Wall(4, 4, 1, 2)])


def _lams(d_ply=None, d_angle=0.0, d_thickness=0.0):
    # odd ply counts: no ply interface on the mid-line, where walls' nodes would merge
    angles = {1: (0, 45, -45), 2: (45, -45, 0)}
    out = {}
    for lid, a in angles.items():
        t = 1e-3 if lid == 1 else 1.5e-3          # different ply thicknesses: butt-jointed corners
        out[lid] = WallLaminate([Ply(CFRP, ang + (d_angle if (lid, j) == d_ply else 0.0),
                                     t + (d_thickness if (lid, j) == d_ply else 0.0)) for j, ang in enumerate(a)])
    return out


def _depth(x):
    return H0 + (H1 - H0) * x / LENGTH


# ---------------------------------------------------------------------- Resal coupling
def _cuts(x, dx, lams, mesh=False, elements_along=3):
    a, b = _box(_depth(x - dx / 2)), _box(_depth(x + dx / 2))
    if not mesh:
        return a, b
    return (SectionMesh.from_thin_walled(a, lams, elements_along=elements_along),
            SectionMesh.from_thin_walled(b, lams, elements_along=elements_along))


def test_solid_taper_matches_the_thin_walled_one():
    lams, x, dx = _lams(), 0.8, 1e-3
    tw = section_taper(*_cuts(x, dx, lams), lams, dx)
    so = solid_section_taper(*_cuts(x, dx, lams, mesh=True, elements_along=6), dx)
    # the box tapers in depth: the caps' z' couples kappa_y with gamma_xz, and Q_zz is its square
    i, j = 1, 1
    assert so.coupling[i, j] == pytest.approx(tw.coupling[i, j], rel=0.03)
    assert so.shear[1, 1] == pytest.approx(tw.shear[1, 1], rel=0.03)
    hp = (H1 - H0) / LENGTH
    # closed form: z' = (h'/h) z on every wall, so C[kappa_y, gamma_xz] = (h'/h) int A z^2 ds
    an = ThinWalledSectionAnalysis(_box(_depth(x)), lams)
    EIy = an.axial_points[:, 2] @ an.axial_points[:, 1] ** 2
    assert tw.coupling[1, 1] == pytest.approx(hp / _depth(x) * EIy, rel=1e-6)


@pytest.mark.parametrize("kind, ply", [("thickness", (1, 0)), ("angle", (1, 1)), ("thickness", (2, 1))])
def test_thin_walled_taper_derivative(kind, ply):
    x, dx = 0.8, 1e-3
    lams = _lams()
    p = LayupParameter(*ply, kind)
    tw = section_taper(*_cuts(x, dx, lams), lams, dx)
    d = section_derivative(ThinWalledSectionAnalysis(_box(_depth(x)), lams), p, taper=tw)
    h = 1e-2 if kind == "angle" else 1e-6
    key = "d_angle" if kind == "angle" else "d_thickness"
    T = [section_taper(*_cuts(x, dx, _lams(ply, **{key: s})), _lams(ply, **{key: s}), dx).matrix() for s in (h, -h)]
    fd = (T[0] - T[1]) / (2 * h)
    np.testing.assert_allclose(d.taper.matrix(), fd, rtol=1e-6, atol=1e-8 * np.abs(fd).max())


@pytest.mark.parametrize("kind, ply", [("thickness", (1, 0)), ("angle", (1, 1)), ("thickness", (2, 1))])
def test_solid_taper_derivative(kind, ply):
    """The two meshes move with the ply at their own velocities; the mean section and the
    rates move with them. Checked against the meshes rebuilt with the changed angle, or
    with their nodes moved along their velocities (a rebuilt mesh's butt joints merge
    different nodes once the plies' interfaces move)."""
    x, dx = 0.8, 1e-3
    lams = _lams()
    p = LayupParameter(*ply, kind)
    a, b = _cuts(x, dx, lams, mesh=True)
    taper = solid_section_taper(a, b, dx)
    mid = SectionMesh.from_thin_walled(_box(_depth(x)), lams, elements_along=3)
    d = solid_section_derivative(SolidSectionAnalysis(mid), p, taper=taper)
    if kind == "angle":
        h = 1e-2
        T = [solid_section_taper(*_cuts(x, dx, _lams(ply, d_angle=s), mesh=True), dx).matrix() for s in (h, -h)]
    else:
        h = 1e-7

        def moved(m, s):
            return SectionMesh(m.nodes + s * m.velocities[ply], m.elements, m.kinds, m.material_ids, m.theta,
                               m.alpha, m.materials, m.section_id, m.ties, m.plies)

        T = [solid_section_taper(moved(a, s), moved(b, s), dx).matrix() for s in (h, -h)]
    fd = (T[0] - T[1]) / (2 * h)
    np.testing.assert_allclose(d.taper.matrix(), fd, rtol=1e-5, atol=1e-6 * np.abs(fd).max())


def test_element_derivative_of_a_tapered_section_adds_the_taper_derivative():
    x, dx = 0.8, 1e-3
    p = LayupParameter(1, 0, "thickness")

    def element(lams):
        a, b = _cuts(x, dx, lams)
        return SectionAnalysis.from_thin_walled(ThinWalledSectionAnalysis(_box(_depth(x)), lams)) \
            .with_taper(section_taper(a, b, lams, dx))

    lams = _lams()
    sa = element(lams)
    d = section_derivative(ThinWalledSectionAnalysis(_box(_depth(x)), lams), p,
                           taper=section_taper(*_cuts(x, dx, lams), lams, dx))
    exact = section_analysis_derivative(sa, d).stiffness
    h = 1e-6
    fd = (element(_lams((1, 0), d_thickness=h)).stiffness - element(_lams((1, 0), d_thickness=-h)).stiffness) / (2 * h)
    np.testing.assert_allclose(exact, fd, rtol=1e-6, atol=1e-7 * np.abs(fd).max())


@pytest.mark.parametrize("ply", [(1, 0), (2, 2)])
def test_thin_walled_thickness_derivative_of_stiffness_and_mass(ply):
    lams = _lams()
    sec = _box(_depth(0.5))
    d = section_derivative(ThinWalledSectionAnalysis(sec, lams), LayupParameter(*ply, "thickness"))
    h = 1e-7
    a, b = (ThinWalledSectionAnalysis(sec, _lams(ply, d_thickness=s)) for s in (h, -h))
    np.testing.assert_allclose(d.stiffness, (a.stiffness - b.stiffness) / (2 * h), rtol=1e-5,
                               atol=1e-6 * np.abs(d.stiffness).max())
    assert d.mass["m"] == pytest.approx((a.mass_properties["m"] - b.mass_properties["m"]) / (2 * h), rel=1e-7)
