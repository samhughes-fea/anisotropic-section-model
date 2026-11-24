"""Isotropic sections built from geometry agree with ``from_isotropic`` and the closed forms.

ASM is the single source of section properties for 1D-FEM, isotropic sections included.
Given constants (``EA``, ``EI``, ``GJ`` ...) go through ``SectionAnalysis.from_isotropic``;
isotropic geometry goes through the 2D (``solid``) or thin-walled builders. These tests
check that the two routes give the same section for a solid rectangle and a thin-walled
tube, and that an asymmetric (angle) section keeps its couplings and principal axes.
"""

from __future__ import annotations

import numpy as np
import pytest

from anisotropic_section_model.section_analysis import SectionAnalysis
from anisotropic_section_model.solid.material import SolidMaterial
from anisotropic_section_model.solid.mesh import SectionMesh
from anisotropic_section_model.solid.section_analysis import SolidSectionAnalysis
from anisotropic_section_model.thin_walled import ThinWalledSection, ThinWalledSectionAnalysis, Wall, WallLaminate

E, NU, RHO = 70e9, 0.3, 2700.0
G = E / (2 * (1 + NU))


def _thin_walled(points, walls, t):
    lam = {1: WallLaminate.isotropic(E, NU, t, rho=RHO)}
    return SectionAnalysis.from_thin_walled(ThinWalledSectionAnalysis(ThinWalledSection(points, walls), lam, hoop="free"))


def _assert_matches_isotropic(se, iso, rtol, shear=(3, 4)):
    """Same diagonal ``D`` (the ``shear`` rows too), no couplings, same mass per length."""
    D, Di = se.stiffness, iso.stiffness
    scale = np.sqrt(np.outer(np.diag(D), np.diag(D)))
    off = D - np.diag(np.diag(D))
    assert np.max(np.abs(off) / scale) < 1e-9
    for i in (0, 1, 2, 5):
        assert D[i, i] == pytest.approx(Di[i, i], rel=rtol)
    for i in shear:
        assert D[i, i] == pytest.approx(Di[i, i], rel=rtol)
    for k in ("m", "I_yy", "I_zz", "I_p"):
        assert se.mass[k] == pytest.approx(iso.mass[k], rel=rtol)


def test_solid_rectangle_matches_from_isotropic():
    b, h = 0.04, 0.1                     # b along y, h along z
    se = SectionAnalysis.from_solid(SolidSectionAnalysis(
        SectionMesh.rectangle(b, h, 8, 16, SolidMaterial.isotropic(E, NU, RHO))))
    A, I_y, I_z = b * h, b * h**3 / 12, h * b**3 / 12
    # Saint-Venant torsion constant of a rectangle (Roark, short side b): J = h b^3 / 3 (1 - 0.63 b/h + 0.052 (b/h)^5)
    r = b / h
    J = h * b**3 / 3 * (1 - 0.63 * r + 0.052 * r**5)
    iso = SectionAnalysis.from_isotropic(E=E, G=G, A=A, I_y=I_y, I_z=I_z, J_t=J, kappa=5 / 6, rho=RHO)
    # the 2D solution's shear coefficient is not exactly 5/6 (next test): shear is checked there
    _assert_matches_isotropic(se, iso, rtol=1e-3, shear=())
    assert se.stiffness[4, 4] == pytest.approx(iso.stiffness[4, 4], rel=1e-2)
    assert se.shear_centre == pytest.approx((0.0, 0.0), abs=1e-12)


def test_solid_rectangle_shear_is_only_approximately_five_sixths():
    """Across the short side the 2D shear coefficient drops well below 5/6 (here about 0.74)."""
    b, h = 0.04, 0.1
    se = SectionAnalysis.from_solid(SolidSectionAnalysis(
        SectionMesh.rectangle(b, h, 8, 16, SolidMaterial.isotropic(E, NU, RHO))))
    kappa_y = se.stiffness[3, 3] / (G * b * h)
    kappa_z = se.stiffness[4, 4] / (G * b * h)
    assert 0.70 < kappa_y < 0.80
    assert kappa_z == pytest.approx(5 / 6, rel=1e-2)


def test_thin_walled_tube_matches_from_isotropic():
    R, t, n = 0.1, 1e-3, 128
    pts = {i + 1: (R * np.cos(2 * np.pi * i / n), R * np.sin(2 * np.pi * i / n)) for i in range(n)}
    walls = [Wall(i + 1, i + 1, (i + 1) % n + 1, 1) for i in range(n)]
    se = _thin_walled(pts, walls, t)
    A, I, J = 2 * np.pi * R * t, np.pi * R**3 * t, 2 * np.pi * R**3 * t
    iso = SectionAnalysis.from_isotropic(E=E, G=G, A=A, I_y=I, I_z=I, J_t=J, I_p=J, kappa=0.5, rho=RHO)
    # a 128-sided polygon: perimeter and moments within 0.1 % of the circle
    _assert_matches_isotropic(se, iso, rtol=1e-3)


def test_thin_walled_angle_keeps_its_couplings_and_principal_axes():
    """An equal-leg angle about its centroid: ``I_y = I_z = 5 a^3 t / 24``, ``I_yz = -a^3 t / 8``,
    principal ``a^3 t / 3`` and ``a^3 t / 12`` at 45 degrees, shear centre at the corner."""
    a, t = 0.1, 1e-3
    c = a / 4                                             # centroid of the mid-lines
    pts = {1: (a - c, -c), 2: (-c, -c), 3: (-c, a - c)}
    se = _thin_walled(pts, [Wall(1, 1, 2, 1), Wall(2, 2, 3, 1)], t)
    C = se.classical_stiffness                            # [eps, kappa_y, kappa_z, phi]
    assert se.tension_centre == pytest.approx((0.0, 0.0), abs=1e-12)
    assert C[1, 1] == pytest.approx(E * 5 * a**3 * t / 24, rel=1e-3)
    assert C[2, 2] == pytest.approx(E * 5 * a**3 * t / 24, rel=1e-3)
    assert abs(C[1, 2]) == pytest.approx(E * a**3 * t / 8, rel=1e-3)
    assert np.linalg.eigvalsh(C[1:3, 1:3]) == pytest.approx([E * a**3 * t / 12, E * a**3 * t / 3], rel=1e-3)
    assert se.is_coupled
    assert se.shear_centre == pytest.approx((-c, -c), abs=1e-4 * a)   # thin-wall limit; off by O(t^2 / a)
    assert C[3, 3] == pytest.approx(G * 2 * a * t**3 / 3, rel=1e-3)
    # in its principal axes the angle is the uncoupled isotropic section of those moments
    angle, (I_1, I_2) = se.principal_axes()
    assert abs(np.degrees(angle)) == pytest.approx(45.0, abs=1e-6)
    iso = SectionAnalysis.from_isotropic(E=E, G=G, A=2 * a * t, I_y=a**3 * t / 3, I_z=a**3 * t / 12,
                                         J_t=2 * a * t**3 / 3, rho=RHO)
    assert sorted([I_1, I_2]) == pytest.approx(sorted([iso.mass["I_yy"], iso.mass["I_zz"]]), rel=1e-3)
