"""Sections given as 6 x 6 tables (``SectionAnalysis.from_matrices``) and the contract version."""

from __future__ import annotations

import numpy as np
import pytest

from anisotropic_section_model import axes
from anisotropic_section_model.section_analysis import SECTION_API_VERSION, SectionAnalysis
from anisotropic_section_model.thin_walled import ThinWalledSection, Wall, WallLaminate
from anisotropic_section_model.thin_walled.laminate import Ply, PlyMaterial
from anisotropic_section_model.thin_walled.section_analysis import ThinWalledSectionAnalysis

CFRP = PlyMaterial(E1=135e9, E2=10e9, G12=5e9, nu12=0.3, G13=5e9, G23=3.8e9, rho=1550.0)


@pytest.fixture(scope="module")
def coupled():
    """An off-centre box with unbalanced walls: every coupling, centres away from the origin."""
    y0, z0, b, h = 0.07, -0.03, 0.2, 0.1
    pts = {1: (y0 - b / 2, z0 - h / 2), 2: (y0 + b / 2, z0 - h / 2), 3: (y0 + b / 2, z0 + h / 2),
           4: (y0 - b / 2, z0 + h / 2)}
    sec = ThinWalledSection(pts, [Wall(1, 1, 2, 1), Wall(2, 2, 3, 2), Wall(3, 3, 4, 3), Wall(4, 4, 1, 2)])
    lams = {1: WallLaminate([Ply(CFRP, a, 1e-3) for a in (30, 30, 0)]),
            2: WallLaminate([Ply(CFRP, a, 1.5e-3) for a in (45, -45, 0)]),
            3: WallLaminate([Ply(CFRP, a, 2e-3) for a in (-20, 0, 0)])}
    an = ThinWalledSectionAnalysis(sec, lams)
    return an, SectionAnalysis.from_thin_walled(an)


def _same(a: SectionAnalysis, b: SectionAnalysis):
    np.testing.assert_allclose(a.stiffness, b.stiffness, rtol=1e-10, atol=1e-10 * np.abs(b.stiffness).max())
    np.testing.assert_allclose(a.classical_stiffness, b.classical_stiffness, rtol=1e-8,
                               atol=1e-8 * np.abs(b.classical_stiffness).max())
    np.testing.assert_allclose(a.mass_matrix(), b.mass_matrix(), rtol=1e-12, atol=1e-12)


def test_api_version_is_exported():
    import anisotropic_section_model as asm

    assert SECTION_API_VERSION == (1, 0)
    assert asm.SECTION_API_VERSION == SECTION_API_VERSION


def test_engine_tables_reproduce_the_thin_walled_section(coupled):
    an, sa = coupled
    t = SectionAnalysis.from_matrices(an.stiffness, sa.mass_matrix(), section_id=7)
    _same(t, sa)
    assert t.source == "matrix" and t.section_id == 7
    assert t.mass == pytest.approx(sa.mass)
    # the centres the stiffness implies are the section's own
    np.testing.assert_allclose(t.tension_centre, sa.tension_centre, atol=1e-9)
    np.testing.assert_allclose(t.shear_centre, sa.shear_centre, atol=1e-6)


def test_public_and_beamdyn_tables_give_the_same_section(coupled):
    an, sa = coupled
    K_pub = axes.from_engine(an.stiffness)
    M_pub = axes.mass_matrix(an.mass_properties)
    M_pub[5, 5] = an.mass_properties["I_p"]
    _same(SectionAnalysis.from_matrices(K_pub, M_pub, frame="public"), sa)
    K_bd = axes.to_beamdyn_frame(K_pub)
    M_bd = axes.to_beamdyn_frame(M_pub)
    _same(SectionAnalysis.from_matrices(K_bd, M_bd, frame="beamdyn"), sa)


def test_isotropic_tables_match_from_isotropic():
    E, G, A, Iy, Iz, J, rho = 70e9, 26e9, 0.01, 2e-5, 5e-6, 1e-5, 2700.0
    iso = SectionAnalysis.from_isotropic(E=E, G=G, A=A, I_y=Iy, I_z=Iz, J_t=J, rho=rho)
    t = SectionAnalysis.from_matrices(iso.stiffness, iso.mass_matrix())
    _same(t, iso)
    assert t.shear_centre == pytest.approx((0.0, 0.0)) and t.tension_centre == pytest.approx((0.0, 0.0))


def test_a_mass_matrix_that_is_not_rigid_is_refused(coupled):
    an, sa = coupled
    M = sa.mass_matrix()
    M[1, 2] = M[2, 1] = 0.1 * M[0, 0]
    with pytest.raises(ValueError, match="rigid-section"):
        SectionAnalysis.from_matrices(an.stiffness, M)
    with pytest.raises(ValueError, match="frame"):
        SectionAnalysis.from_matrices(an.stiffness, sa.mass_matrix(), frame="windio")


def test_massless_tables_are_allowed_for_statics(coupled):
    an, _ = coupled
    t = SectionAnalysis.from_matrices(an.stiffness)
    assert t.mass == {}
    with pytest.raises(ValueError, match="no mass"):
        t.mass_matrix()
