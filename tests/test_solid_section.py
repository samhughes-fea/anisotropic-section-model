"""``SolidSection``: the 2D finite-element section in the public frame (``axes``)."""

import numpy as np
import pytest

from anisotropic_section_model import SolidSection, axes
from anisotropic_section_model.solid.mesh import SectionMesh
from anisotropic_section_model.thin_walled import ThinWalledSection, Wall, WallLaminate

E, NU, RHO = 70e9, 0.3, 2700.0
XC, YC = 0.05, 0.02  # the box's centre in public (x, y)
B, H, T = 0.20, 0.10, 2e-3  # wider along x than along y


@pytest.fixture(scope="module")
def section():
    pts = {1: (XC - B / 2, YC - H / 2), 2: (XC + B / 2, YC - H / 2),
           3: (XC + B / 2, YC + H / 2), 4: (XC - B / 2, YC + H / 2)}
    walls = [Wall(1, 1, 2, 1), Wall(2, 2, 3, 1), Wall(3, 3, 4, 1), Wall(4, 4, 1, 1)]
    lams = {1: WallLaminate.isotropic(E, NU, T, rho=RHO)}
    return SolidSection(SectionMesh.from_thin_walled(ThinWalledSection(pts, walls), lams, elements_along=4))


def test_stiffness_is_the_engine_stiffness_reordered(section):
    np.testing.assert_allclose(section.timoshenko_stiffness(), axes.from_engine(section.analysis.stiffness))


def test_public_axes_of_an_offset_box(section):
    K = section.timoshenko_stiffness((XC, YC))
    scale = np.abs(K).max()
    # about its centre an isotropic box has no couplings (to discretisation precision)
    off = K - np.diag(np.diag(K))
    assert np.abs(off).max() < 1e-6 * scale
    EA = E * 2 * (B + H) * T
    assert K[2, 2] == pytest.approx(EA, rel=0.02)
    # kappa_x bends about x: int E y^2, smaller than int E x^2 for a box wider along x
    EIx = E * T * (2 * B * (H / 2) ** 2 + 2 * H ** 3 / 12)
    assert K[3, 3] == pytest.approx(EIx, rel=0.03)
    assert K[4, 4] > 2.0 * K[3, 3]
    assert section.tension_centre == pytest.approx((XC, YC), abs=1e-9)
    assert section.shear_centre == pytest.approx((XC, YC), abs=1e-9)


def test_mass_matrix_about_the_origin(section):
    M = section.mass_matrix()
    m = RHO * 2 * (B + H) * T
    assert M[0, 0] == pytest.approx(m, rel=0.02)
    assert M[2, 3] == pytest.approx(m * YC, rel=0.02)
    assert M[2, 4] == pytest.approx(-m * XC, rel=0.02)
    np.testing.assert_allclose(section.mass_matrix((XC, YC))[2, 3:5], 0.0, atol=1e-9 * m)


def test_strain_response_inverts_the_stiffness(section):
    f = np.array([1.0, -2.0, 3e3, 40.0, -50.0, 6.0])
    e = section.strain_response(f)
    np.testing.assert_allclose(section.timoshenko_stiffness() @ e, f, rtol=1e-8, atol=1e-8)
    assert section.GJ_free == pytest.approx(1.0 / np.linalg.inv(section.timoshenko_stiffness())[5, 5])
