"""laminate-constitutive-model plugs into the section code: laminates, materials and failure."""

from pathlib import Path

import numpy as np
import pytest

from laminate_constitutive_model import (
    ElasticProperties3D,
    Laminate,
    MaterialCard,
    StrengthProperties3D,
    integrate_abd,
)
from laminate_constitutive_model.laminate_kinematics import integrate_shear

from anisotropic_section_model.ply_failure import PlyStrength, exposures
from anisotropic_section_model.solid.material import SolidMaterial
from anisotropic_section_model.thin_walled.laminate import PlyMaterial, WallLaminate

ELASTIC = ElasticProperties3D(E1=135e9, E2=9.5e9, E3=9.8e9, v12=0.30, v23=0.42, v13=0.29,
                              G12=5.2e9, G23=3.3e9, G13=5.0e9)
CARD = MaterialCard("cfrp", ELASTIC, StrengthProperties3D(Xt=1500e6, Xc=900e6, Yt=27e6, Yc=200e6, S12=80e6),
                    density=1550.0)


def test_wall_laminate_from_lcm_laminate_has_the_same_stiffness():
    lam = Laminate.from_angles(CARD, [0, 45, -45, 90, 30], 0.25e-3)
    wall = WallLaminate.from_laminate(lam)
    abd = integrate_abd(lam)
    assert np.allclose(wall.ABD(), abd, rtol=1e-13, atol=1e-12 * np.abs(abd).max())
    swap = np.array([[0, 1], [1, 0]])  # LCM [gamma_yz, gamma_xz] -> wall [gamma_xn, gamma_sn]
    assert np.allclose(wall.H(), wall.shear_correction * swap @ integrate_shear(lam) @ swap, rtol=1e-13)
    assert wall.mass_per_area == pytest.approx(1550.0 * lam.thickness)
    assert [p.angle_deg for p in wall.plies] == [0, 45, -45, 90, 30]


def test_offset_laminates_keep_their_eccentricity():
    """The wall law is about the laminate's reference surface, as the shell's ABD is."""
    lam = Laminate.from_angles(CARD, [0, 90], 1e-3, offset=1e-3)
    wall = WallLaminate.from_laminate(lam)
    assert wall.offset == 1e-3
    abd = integrate_abd(lam)
    assert np.allclose(wall.ABD(), abd, rtol=1e-12, atol=1e-12 * np.abs(abd).max())
    centred = WallLaminate.from_laminate(lam.with_offset(0.0))
    assert np.allclose(wall.A(), centred.A(), rtol=1e-13)
    assert np.allclose(wall.B(), centred.B() + 1e-3 * centred.A(), rtol=1e-12, atol=1e-12 * np.abs(wall.B()).max())


def test_materials_from_lcm():
    ply = PlyMaterial.from_material_card(CARD)
    assert np.allclose(ply.reduced_stiffness(), ELASTIC.reduced_stiffness(), rtol=1e-14)
    solid = SolidMaterial.from_material_card(CARD)
    assert solid.rho == 1550.0
    assert np.allclose(solid.stiffness(), ELASTIC.stiffness_matrix(), rtol=1e-12)


def test_strength_from_lcm():
    st = PlyStrength.from_strength_properties(CARD.strength, p_TT=0.3)
    assert (st.Xt, st.Yc, st.S12, st.s23, st.p_TT) == (1500e6, 200e6, 80e6, 80e6, 0.3)


def test_exposures_match_the_criteria_moved_from_1d_fem():
    """The criteria now come from laminate-constitutive-model. The reference is what 1d-fem's
    own implementation gave (``tests/data/ply_failure_1dfem.npz``, made before the swap)."""
    ref = np.load(Path(__file__).parent / "data" / "ply_failure_1dfem.npz")
    s = ref["stress"]
    for i, kw in enumerate(({}, {"S23": 60e6, "Zt": 40e6, "F_star": 0.2, "p_TL_t": 0.35, "p_TL_c": 0.3})):
        new = exposures(s, PlyStrength(Xt=1500e6, Xc=900e6, Yt=27e6, Yc=200e6, S12=80e6, **kw))
        for key in ("max_stress", "tsai_wu", "hashin", "puck", "puck_angle", "exposure"):
            assert np.allclose(new[key], ref[f"{i}_{key}"], rtol=1e-12, atol=0.0), key
        for key in ("max_stress_mode", "hashin_mode", "puck_mode"):
            assert list(new[key]) == list(ref[f"{i}_{key}"]), key
