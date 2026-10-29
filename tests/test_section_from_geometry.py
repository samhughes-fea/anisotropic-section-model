"""ThinWalledSection.from_geometry on cuts of the implicit geometry, checked against the
shell-mesh route (BladeBuilder, region-based and from the geometry)."""

import numpy as np
import pytest

from anisotropic_section_model import ThinWalledSection
from composite_laminate_model import BladeBuilder, FSDTModel, Laminate, create_default_library
from implicit_geometry_kernel import Airfoil, BladeGeometry, ShearWeb, SkinRegion, Station
from implicit_geometry_kernel.section_geometry.layout.system_layout import SYSTEM_TYPE_KEYS


AIRFOIL = Airfoil.naca4("0024")


def _prismatic(twist=0.0, offset=(0.0, 0.0)):
    return [Station(0.0, 1.0, AIRFOIL, twist=twist, pitch_axis=0.3, offset=offset),
            Station(4.0, 1.0, AIRFOIL, twist=twist, pitch_axis=0.3, offset=offset)]


def _cut(stations=None, z=1.0, layout="2D-F", **kw):
    options = dict(spar_cap_thickness=0.008, shear_web_thickness=0.002, web_positions=(0.18, 0.48))
    options.update(kw)
    return BladeGeometry(stations or _prismatic(), layout, 0.003, **options).section_at(z)


@pytest.fixture(scope="module")
def cut():
    return _cut()


@pytest.fixture(scope="module")
def laminates():
    lib = create_default_library()
    glass, carbon = lib.get("E-glass/epoxy"), lib.get("T300/5208")
    return {
        "skin": Laminate.from_angles(glass, [0, 45, -45, -45, 45, 0], 0.5e-3),
        "web": Laminate.from_angles(glass, [45, -45, -45, 45], 0.5e-3),
        "cap": Laminate.from_angles(carbon, [0] * 16, 0.5e-3),
    }


@pytest.fixture(scope="module")
def sections(cut, laminates):
    from_geometry = ThinWalledSection.from_geometry(cut, laminates)
    skin, web, cap = laminates["skin"], laminates["web"], laminates["cap"]
    regions = [SkinRegion("le_panel", 0.0, 0.18, 8), SkinRegion("spar_cap", 0.18, 0.48, 12),
               SkinRegion("te_panel", 0.48, 1.0, 16)]
    webs = [ShearWeb("web_fwd", 0.18, 6), ShearWeb("web_aft", 0.48, 6)]
    # in a shell model the spar cap region's laminate is the cap and skin stacked
    lams = {"le_panel": skin, "te_panel": skin, "spar_cap": Laminate(cap.plies + skin.plies), "web_fwd": web,
            "web_aft": web}
    blade = BladeBuilder(_prismatic(), regions, webs, n_span=2, mesh_surface="mid").build(lams, FSDTModel())
    return from_geometry, ThinWalledSection.from_blade_mesh(blade, z=2.0)


def test_from_geometry_matches_the_shell_mesh_section(sections):
    geo, mesh = sections
    assert geo.n_cells == mesh.n_cells == 3
    for name in ("EA", "EI_xx", "EI_yy", "GJ"):
        assert getattr(geo, name) == pytest.approx(getattr(mesh, name), rel=0.015), name
    assert geo.EGamma == pytest.approx(mesh.EGamma, rel=0.05)
    assert np.allclose(geo.shear_centre, mesh.shear_centre, atol=0.003)


def test_from_geometry_merges_caps_into_the_skin(sections, laminates):
    geo, _ = sections
    names = {w.name for w in geo.walls}
    assert {"spar_cap_suction_surface", "spar_cap_pressure_surface", "shear_web_0", "shear_web_1"} <= names
    stack = laminates["cap"].thickness + laminates["skin"].thickness
    for w in geo.walls:
        if w.name.startswith("spar_cap"):
            assert w.laminate.thickness == pytest.approx(stack)
    # the box walls carry the stacked laminate: the spar cap length is the bonded length
    cap_length = sum(geo.length[k] for k, w in enumerate(geo.walls) if w.name == "spar_cap_suction_surface")
    assert cap_length == pytest.approx(0.30, rel=0.05)


def test_from_geometry_laminate_lookup_and_checks(cut, laminates):
    lib = create_default_library()
    thick = Laminate.from_angles(lib.get("E-glass/epoxy"), [0, 45, -45, -45, 45, 0, 0, 0], 0.5e-3)
    with pytest.raises(ValueError, match="mm thick"):
        ThinWalledSection.from_geometry(cut, dict(laminates, skin=thick))
    with pytest.raises(KeyError, match="shear_web_0"):
        ThinWalledSection.from_geometry(cut, {"skin": laminates["skin"], "cap": laminates["cap"]})
    # a full label beats the region, which beats the kind
    stiff_web = Laminate.from_angles(lib.get("E-glass/epoxy"), [45, -45, 0, 0, -45, 45], 0.5e-3)
    base = ThinWalledSection.from_geometry(cut, laminates)
    one = ThinWalledSection.from_geometry(cut, dict(laminates, shear_web_1=stiff_web))
    assert any(w.laminate is stiff_web for w in one.walls if w.name == "shear_web_1")
    assert all(w.laminate is not stiff_web for w in one.walls if w.name == "shear_web_0")
    assert one.EA > base.EA


def _geometry(stations=None, **kw):
    options = dict(spar_cap_thickness=0.008, shear_web_thickness=0.002, web_positions=(0.18, 0.48))
    options.update(kw)
    return BladeGeometry(stations or _prismatic(), "2D-F", 0.003, **options)


@pytest.fixture(scope="module")
def geometry_blade(laminates):
    return BladeBuilder.from_geometry(_geometry(), n_span=4).build(laminates, FSDTModel())


def test_geometry_mesh_section_matches_the_thin_wall_section(geometry_blade, laminates):
    mesh_section = ThinWalledSection.from_blade_mesh(geometry_blade, z=2.0)
    direct = ThinWalledSection.from_geometry(_geometry().section_at(2.0), laminates)
    for name in ("EA", "EI_xx", "EI_yy", "GJ"):
        assert getattr(mesh_section, name) == pytest.approx(getattr(direct, name), rel=0.005), name
    assert np.allclose(mesh_section.shear_centre, direct.shear_centre, atol=1e-3)


def test_geometry_mesh_matches_the_region_mesh(sections, laminates):
    """The geometry's shell model against BladeBuilder's on the example 20 section."""
    _, region_section = sections
    blade = BladeBuilder.from_geometry(_geometry(), n_span=2).build(laminates, FSDTModel())
    geometry_section = ThinWalledSection.from_blade_mesh(blade, z=2.0)
    for name in ("EA", "EI_xx", "EI_yy", "GJ"):
        assert getattr(geometry_section, name) == pytest.approx(getattr(region_section, name), rel=0.015), name


@pytest.mark.parametrize("layout", SYSTEM_TYPE_KEYS)
def test_every_layout_gives_a_section(layout, laminates):
    geo = BladeGeometry(_prismatic(), layout, 0.003, spar_cap_thickness=0.008, shear_web_thickness=0.002)
    cut = geo.section_at(1.0)
    section = ThinWalledSection.from_geometry(cut, laminates)
    assert section.n_cells == len(cut.cells) >= 1
    assert section.GJ > 0.0 and section.EA > 0.0


@pytest.mark.parametrize("layout", ["0A", "1C-F", "3B-CN", "5D-F"])
def test_layouts_mesh(layout, laminates):
    geo = BladeGeometry(_prismatic(), layout, 0.003, spar_cap_thickness=0.008, shear_web_thickness=0.002)
    blade = BladeBuilder.from_geometry(geo, n_span=1, element="quad4").build(laminates, FSDTModel())
    from_mesh = ThinWalledSection.from_blade_mesh(blade, z=0.0)
    direct = ThinWalledSection.from_geometry(geo.section_at(0.0), laminates)
    assert from_mesh.n_cells == direct.n_cells
    assert from_mesh.GJ == pytest.approx(direct.GJ, rel=0.01)
    assert from_mesh.EA == pytest.approx(direct.EA, rel=0.01)
