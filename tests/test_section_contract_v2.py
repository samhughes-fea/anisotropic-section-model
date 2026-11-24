"""Section-provider contract v2: engine-frame sections, plain derivatives, torsion and polar constants."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import anisotropic_section_model as asm
from anisotropic_section_model.section_analysis import SECTION_CONSTANTS, SectionAnalysis
from anisotropic_section_model.solid.mesh import SectionMesh
from anisotropic_section_model.solid.section_analysis import SolidSectionAnalysis
from anisotropic_section_model.thin_walled import ThinWalledSection, ThinWalledSectionAnalysis, Wall, WallLaminate
from anisotropic_section_model.thin_walled.laminate import Ply, PlyMaterial
from anisotropic_section_model.thin_walled.layup_sensitivity import (LayupParameter, SectionAnalysisDerivative,
                                                                     section_analysis_derivative, section_derivative)

CFRP = PlyMaterial(E1=135e9, E2=10e9, G12=5e9, nu12=0.3, G13=5e9, G23=3.8e9, rho=1550.0)


def _box(d_angle=0.0, d_thickness=0.0):
    """An off-centre, unbalanced CFRP box: every coupling is non-zero, kappa_y included."""
    pts = {1: (-0.08, -0.03), 2: (0.12, -0.03), 3: (0.12, 0.07), 4: (-0.08, 0.07)}
    walls = [Wall(1, 1, 2, 1), Wall(2, 2, 3, 2), Wall(3, 3, 4, 1), Wall(4, 4, 1, 2)]
    lams = {1: WallLaminate([Ply(CFRP, 30.0 + d_angle, 1e-3 + d_thickness), Ply(CFRP, 0.0, 1e-3)]),
            2: WallLaminate([Ply(CFRP, 45.0, 1.5e-3), Ply(CFRP, -45.0, 1.5e-3)])}
    return ThinWalledSectionAnalysis(ThinWalledSection(pts, walls), lams)


def test_package_declares_contract_2_x():
    assert asm.SECTION_API_VERSION == (2, 1)


def test_thin_walled_and_solid_sections_keep_the_right_handed_matrices():
    tw = _box()
    sa = SectionAnalysis.from_thin_walled(tw)
    assert sa.frame == "engine"
    np.testing.assert_array_equal(sa.stiffness, tw.stiffness)
    np.testing.assert_array_equal(sa.classical_stiffness, tw.classical_stiffness)
    np.testing.assert_array_equal(sa.benscoter_stiffness, tw.benscoter_stiffness)
    np.testing.assert_array_equal(sa.geometric_terms, tw.geometric_terms)
    assert abs(sa.stiffness[0, 1]) > 0.0                       # extension-bending is there to flip
    mesh = SectionMesh.from_thin_walled(tw.section, tw.laminates, elements_along=2)
    so = SolidSectionAnalysis(mesh)
    ss = SectionAnalysis.from_solid(so)
    np.testing.assert_array_equal(ss.stiffness, so.stiffness)
    np.testing.assert_array_equal(ss.benscoter_stiffness, so.benscoter_stiffness)


def test_right_handed_bending_stiffness_couples_z_with_kappa_y():
    """M_y = int sigma z dA with the fibre strain eps + z kappa_y - y kappa_z: a section whose
    tension centre sits at z_t > 0 has D[0, 1] = EA z_t > 0 about the origin."""
    sa = SectionAnalysis.from_thin_walled(_box())
    z_t = sa.tension_centre[1]
    assert z_t > 0.0
    C = sa.classical_stiffness
    assert C[0, 1] == pytest.approx(C[0, 0] * z_t, rel=1e-6)


def test_pretwist_uses_the_right_handed_fibre_strain():
    sa = SectionAnalysis.from_thin_walled(_box())
    k = 0.2
    y, z, wE = sa.axial_points.T
    r2 = y * y + z * z
    dD = sa.with_pretwist(k).stiffness - sa.stiffness
    np.testing.assert_allclose(dD[:3, 5], k * np.array([wE @ r2, wE @ (z * r2), -(wE @ (y * r2))]), rtol=1e-12)
    assert dD[5, 5] == pytest.approx(k * k * wE @ (r2 * r2), rel=1e-12)


def test_station_modifiers_refuse_a_re_signed_section():
    sa = dataclasses.replace(SectionAnalysis.from_thin_walled(_box()), frame="element")
    with pytest.raises(ValueError, match="engine frame"):
        sa.with_pretwist(0.1)


def test_isotropic_constants_and_the_polar_moment():
    kw = dict(E=70e9, G=26e9, A=0.01, I_y=2e-5, I_z=5e-6, J_t=1e-5, rho=2700.0, Gamma=3e-9)
    sa = SectionAnalysis.from_isotropic(**kw, I_p=3e-5)
    c = sa.section_constants
    assert set(c) == set(SECTION_CONSTANTS)
    assert c["I_p"] == 3e-5 and c["J_t"] == 1e-5
    assert sa.stiffness[5, 5] == pytest.approx(26e9 * 1e-5)          # G J_t: stiffness only
    assert sa.mass_matrix()[3, 3] == pytest.approx(2700.0 * 3e-5)     # rho I_p: mass only
    assert SectionAnalysis.from_isotropic(**kw).section_constants["I_p"] == pytest.approx(2.5e-5)
    with pytest.warns(DeprecationWarning, match="I_p"):
        old = SectionAnalysis.from_isotropic(**kw, I_x=3e-5)
    assert old.mass == sa.mass
    with pytest.raises(TypeError):
        SectionAnalysis.from_isotropic(**kw, I_x=3e-5, I_p=3e-5)
    assert SectionAnalysis.from_thin_walled(_box()).section_constants is None


@pytest.mark.parametrize("kind", ["angle", "thickness"])
def test_section_analysis_derivative_matches_central_differences(kind):
    p = LayupParameter(1, 0, kind)
    tw = _box()
    sa = SectionAnalysis.from_thin_walled(tw)
    d = section_analysis_derivative(sa, section_derivative(tw, p, benscoter=True))
    assert isinstance(d, SectionAnalysisDerivative)
    h = 1e-2 if kind == "angle" else 1e-7
    key = "d_angle" if kind == "angle" else "d_thickness"
    plus, minus = (SectionAnalysis.from_thin_walled(_box(**{key: s})) for s in (h, -h))
    for name in ("stiffness", "classical_stiffness", "benscoter_stiffness"):
        fd = (getattr(plus, name) - getattr(minus, name)) / (2 * h)
        np.testing.assert_allclose(getattr(d, name), fd, rtol=1e-4, atol=1e-6 * np.abs(fd).max(), err_msg=name)
    fd = (plus.mass_matrix() - minus.mass_matrix()) / (2 * h)
    np.testing.assert_allclose(d.mass_matrix, fd, rtol=1e-4, atol=1e-8 * max(np.abs(fd).max(), 1e-30))
