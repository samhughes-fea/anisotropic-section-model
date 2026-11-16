"""Solid 2D 6 x 6 versus the thin-walled 6 x 6, and versus published matrices when held.

Published cells are ``absent`` unless the caller supplies a record that actually
holds them. VABS and BECAS are never filled. The kernel tests use a 2D-F cut at
the root and in the cap region; they do not invent a published 6 x 6.
"""

from __future__ import annotations

import numpy as np
import pytest

from anisotropic_section_model.solid.compare import (
    ABSENT,
    NAMED_STIFFNESS,
    compare_analyses,
    compare_cut,
    compare_thick_stations,
    named_stiffness,
    published_from_record,
)
from anisotropic_section_model.solid.mesh import SectionMesh
from anisotropic_section_model.solid.section_analysis import SolidSectionAnalysis
from anisotropic_section_model.thin_walled import ThinWalledSection, Wall, WallLaminate
from anisotropic_section_model.thin_walled.section_analysis import ThinWalledSectionAnalysis

E, NU = 70e9, 0.3


def iso(t, rho=2700.0):
    return WallLaminate.isotropic(E, NU, t, rho=rho)


def box(b=0.20, h=0.10):
    pts = {1: (-b / 2, -h / 2), 2: (b / 2, -h / 2), 3: (b / 2, h / 2), 4: (-b / 2, h / 2)}
    walls = [Wall(1, 1, 2, 1), Wall(2, 2, 3, 1), Wall(3, 3, 4, 1), Wall(4, 4, 1, 1)]
    return ThinWalledSection(pts, walls)


def box_pair(elements_along=4):
    sec = box()
    lams = {1: iso(2e-3)}
    thin = ThinWalledSectionAnalysis(sec, lams, hoop="free")
    solid = SolidSectionAnalysis(SectionMesh.from_thin_walled(sec, lams, elements_along=elements_along))
    return solid, thin


def test_published_from_record_drops_null_and_unordered_k6():
    record = {
        "EA": 1.0e9,
        "EI_flap": 2.0e6,
        "EI_edge": 3.0e6,
        "GJ": 4.0e5,
        "GA_flap": None,
        "GA_edge": None,
        "K6": np.eye(6).tolist(),
        "vabs": {"EA": 1.1e9},
        "becas": {"EA": 1.05e9},
    }
    held = published_from_record(record)
    assert held["EA"] == 1.0e9
    assert "GA_flap" not in held and "GA_edge" not in held
    assert "K6" not in held
    assert "vabs" not in held and "becas" not in held
    held_k6 = published_from_record({**record, "K6_order": "onedfem"})
    np.testing.assert_allclose(held_k6["K6"], np.eye(6))


def test_named_stiffness_of_a_symmetric_box():
    _, thin = box_pair()
    terms = named_stiffness(thin)
    assert terms["EA"] > 0 and terms["EI_flap"] > 0 and terms["GJ"] > 0
    assert terms["shear_centre_y"] == pytest.approx(0.0, abs=1e-12)
    assert terms["tension_centre_y"] == pytest.approx(0.0, abs=1e-12)


def test_compare_analyses_marks_unpublished_cells_absent():
    solid, thin = box_pair()
    row = compare_analyses(solid, thin, role="root", z=0.0, has_caps=False)
    assert row.role == "root" and row.z == 0.0
    assert row.vabs == ABSENT and row.becas == ABSENT
    assert row.published_stiffness == ABSENT
    assert row.relative_frobenius_to_published == ABSENT
    for name in NAMED_STIFFNESS:
        term = row.terms[name]
        assert term.published == ABSENT
        assert term.relative_to_published == ABSENT
        assert term.difference_from_published == ABSENT
        assert isinstance(term.relative_to_thin_walled, float)
    assert isinstance(row.relative_frobenius_to_thin_walled, float)
    # same mid-line walls: the 2D strip mesh is within a few percent of the thin-walled 6 x 6
    assert row.terms["EA"].relative_to_thin_walled == pytest.approx(0.0, abs=0.03)
    payload = row.as_dict()
    assert payload["vabs"] == ABSENT and payload["published_stiffness"] == ABSENT


def test_compare_analyses_fills_only_the_held_published_terms():
    solid, thin = box_pair()
    solid_terms = named_stiffness(solid)
    published = {
        "EA": solid_terms["EA"] * 1.10,
        "GJ": solid_terms["GJ"],
        "GA_flap": None,
        "K6": (1.05 * np.asarray(solid.stiffness)).tolist(),
        "K6_order": "strains",
    }
    row = compare_analyses(solid, thin, published, role="cap", z=3.0, has_caps=True)
    assert row.terms["EA"].published == pytest.approx(solid_terms["EA"] * 1.10)
    assert row.terms["EA"].relative_to_published == pytest.approx(-0.10 / 1.10)
    assert row.terms["GJ"].relative_to_published == pytest.approx(0.0, abs=1e-12)
    assert row.terms["EI_flap"].published == ABSENT
    assert row.terms["GA_flap"].published == ABSENT
    assert row.relative_frobenius_to_published == pytest.approx(0.05 / 1.05, rel=1e-12)
    assert row.vabs == ABSENT and row.becas == ABSENT
    assert row.has_caps is True


def test_unknown_role_is_refused():
    solid, thin = box_pair()
    with pytest.raises(ValueError, match="role"):
        compare_analyses(solid, thin, role="midspan")


@pytest.fixture(scope="module")
def kernel_cuts():
    pytest.importorskip("implicit_geometry_kernel")
    from implicit_geometry_kernel import Airfoil, BladeGeometry, Station

    from anisotropic_section_model.thin_walled import Ply, PlyMaterial, WallLaminate as WL

    glass = PlyMaterial(E1=38.6e9, E2=8.27e9, G12=4.14e9, nu12=0.26, G13=4.14e9, G23=3.063e9, rho=1800.0)
    carbon = PlyMaterial(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28, G13=7.17e9, G23=3.679e9, rho=1600.0)

    def lam(mat, angles, t=0.5e-3):
        return WL([Ply(mat, a, t) for a in angles])

    laminates = {
        "skin": lam(glass, [0, 45, -45, -45, 45, 0]),
        "web": lam(glass, [45, -45, -45, 45]),
        "cap": lam(carbon, [0] * 16),
    }
    af = Airfoil.naca4("0024")
    geo = BladeGeometry(
        [Station(0.0, 1.0, af, pitch_axis=0.3), Station(6.0, 1.0, af, pitch_axis=0.3)],
        "2D-F-LC-0R",
        0.003,
        0.008,
        0.002,
        web_positions=(0.18, 0.48),
        pivot_fraction=0.3,
    )
    return geo.section_at(0.0), geo.section_at(3.0), laminates


def test_root_and_cap_stations_report_thin_walled_difference_and_absent_published(kernel_cuts):
    root, cap, laminates = kernel_cuts
    rows = compare_thick_stations(
        (("root", root), ("cap", cap)),
        laminates,
        element_size=0.03,
    )
    assert [row.role for row in rows] == ["root", "cap"]
    assert rows[0].z == pytest.approx(0.0)
    assert rows[1].z == pytest.approx(3.0)
    assert rows[1].has_caps is True
    for row in rows:
        assert row.vabs == ABSENT and row.becas == ABSENT
        assert row.published_stiffness == ABSENT
        assert row.relative_frobenius_to_published == ABSENT
        for name in ("EA", "EI_flap", "EI_edge", "GJ"):
            term = row.terms[name]
            assert isinstance(term.solid, float) and term.solid > 0
            assert isinstance(term.relative_to_thin_walled, float)
            assert term.published == ABSENT
            assert term.relative_to_published == ABSENT
        # true 2D geometry versus the mid-line model: a few percent, as the existing
        # solid-from-cut tests already allow; not a published gate
        assert abs(row.terms["EA"].relative_to_thin_walled) < 0.05
        assert isinstance(row.relative_frobenius_to_thin_walled, float)


def test_root_cut_fills_held_named_terms_and_leaves_k6_absent(kernel_cuts):
    root, _, laminates = kernel_cuts
    # caller-supplied named terms only: this is not a catalogue number
    published = {"EA": 1.0, "EI_flap": 1.0, "EI_edge": 1.0, "GJ": 1.0, "GA_flap": None}
    row = compare_cut(root, laminates, published, role="root", element_size=0.03)
    assert row.terms["EA"].published == 1.0
    assert isinstance(row.terms["EA"].relative_to_published, float)
    assert row.terms["GA_flap"].published == ABSENT
    assert row.published_stiffness == ABSENT
    assert row.vabs == ABSENT and row.becas == ABSENT
