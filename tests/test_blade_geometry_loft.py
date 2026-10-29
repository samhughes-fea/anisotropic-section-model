"""BladeGeometry (the implicit section geometry lofted between stations) and
ThinWalledSection.from_geometry, checked against the shell-mesh route."""

import numpy as np
import pytest

from anisotropic_section_model import ThinWalledSection
from blade_geometry_model import Airfoil, BladeGeometry, ShearWeb, SkinRegion, Station, section_at
from composite_laminate_model import BladeBuilder, FSDTModel, Laminate, create_default_library

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


def test_walls_labels_and_thicknesses(cut):
    kinds = {w.kind for w in cut.walls}
    assert kinds == {"skin", "web", "cap"}
    assert {w.label for w in cut.walls_of("web")} == {"shear_web_0", "shear_web_1"}
    assert {w.label for w in cut.walls_of("cap")} == {"spar_cap_suction_surface", "spar_cap_pressure_surface"}
    # thin walls are below section_geometry's manufacturing floors and still come back as given
    t = {w.kind: w.thickness for w in cut.walls}
    assert t == pytest.approx({"skin": 0.003, "web": 0.002, "cap": 0.008})
    # webs at their chord fractions, measured from the pitch axis
    xs = sorted(float(np.mean(w.xy[:, 0])) for w in cut.walls_of("web"))
    assert xs == pytest.approx([0.18 - 0.3, 0.48 - 0.3], abs=1e-9)
    # caps bonded inside the skin between the webs
    for cap in cut.walls_of("cap"):
        assert cap.bonds and all(host.startswith("skin_") for host in cap.bonds)
        assert cap.xy[:, 0].min() == pytest.approx(-0.12) and cap.xy[:, 0].max() == pytest.approx(0.18)
    assert cut.layout == "2D-F" and cut.dropped_webs == ()


def test_cell_areas(cut):
    assert len(cut.cells) == 3
    # the wall midline encloses a little less than the NACA 0024 outline (0.685 t c^2)
    total = sum(c.area for c in cut.cells)
    assert 0.95 * 0.685 * 0.24 < total < 0.685 * 0.24
    for c in cut.cells:
        x, y = c.points[:, 0], c.points[:, 1]
        assert abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) == pytest.approx(c.area, rel=1e-6)


def test_twist_and_offset_place_the_untwisted_section():
    cut = _cut(layout="2D-CN")
    placed = _cut(_prismatic(twist=15.0, offset=(0.1, -0.05)), layout="2D-CN")
    for wall in cut.walls:
        assert np.allclose(placed.to_blade(wall.xy[:, 0], wall.xy[:, 1]), placed.wall(wall.label).xy, atol=1e-9)
    # nose up: the leading edge moves to +y
    le = placed.to_blade([-0.3], [0.0])[0]
    assert le[1] > -0.05


@pytest.mark.parametrize("layout, tilt", [("2D-F", 0.0), ("2D-CN", 12.0)])
def test_flapwise_webs_stay_normal_to_the_rotor_plane(layout, tilt):
    twisted = _cut(_prismatic(twist=12.0), layout=layout)
    for web in twisted.walls_of("web"):
        d = web.xy[-1] - web.xy[0]
        assert abs(np.degrees(np.arctan2(abs(d[0]), abs(d[1])))) == pytest.approx(tilt, abs=1e-6)
    assert len(twisted.cells) == 3


def test_taper_interpolates_chord_twist_and_offset():
    stations = [Station(0.0, 2.0, AIRFOIL, pitch_axis=0.3),
                Station(10.0, 1.0, AIRFOIL, twist=10.0, pitch_axis=0.3, offset=(0.0, 0.2))]
    geo = BladeGeometry(stations, "2D-F", 0.004, spar_cap_thickness=0.008, shear_web_thickness=0.004)
    mid = geo.section_at(5.0)
    assert (mid.chord, mid.twist, *mid.offset) == pytest.approx((1.5, 5.0, 0.0, 0.1))
    root, tip = geo.section_at(0.0), geo.section_at(10.0)
    total = [sum(c.area for c in s.cells) for s in (root, mid, tip)]
    assert total[0] > total[1] > total[2]
    assert total[0] / total[2] == pytest.approx(4.0, rel=0.05)  # area scales with chord^2
    with pytest.raises(ValueError):
        geo.section_at(11.0)


def test_fixed_web_offsets_drop_a_web_that_no_longer_fits():
    stations = [Station(0.0, 2.0, AIRFOIL, pitch_axis=0.3), Station(10.0, 1.0, AIRFOIL, pitch_axis=0.3)]
    geo = BladeGeometry(stations, "2D-F", 0.004, spar_cap_thickness=0.008, shear_web_thickness=0.004,
                        web_offsets=(0.4, 1.2))
    root, tip = geo.section_at(0.0), geo.section_at(10.0)
    assert len(root.walls_of("web")) == 2 and root.dropped_webs == ()
    assert len(tip.walls_of("web")) == 1 and len(tip.dropped_webs) == 1
    assert "1.2 m" in tip.dropped_webs[0]
    xs = [float(np.mean(w.xy[:, 0])) for w in root.walls_of("web")]
    assert sorted(xs) == pytest.approx([0.4 - 0.6, 1.2 - 0.6], abs=1e-9)  # pitch axis 0.3 x 2 m


def test_thickness_as_a_function_of_span():
    stations = _prismatic()
    cut = section_at(stations, 3.0, "2D-F", lambda z: 0.002 + 0.001 * z, spar_cap_thickness=0.008,
                     shear_web_thickness=0.004)
    assert [w.thickness for w in cut.walls_of("skin")] == pytest.approx([0.005] * 6)


def test_bad_options():
    with pytest.raises(ValueError):
        BladeGeometry(_prismatic(), "2D-F", 0.003, web_positions=(0.2, 0.5), web_offsets=(0.2, 0.5))
    with pytest.raises(ValueError):
        BladeGeometry(_prismatic(), "2D-F", 0.003, web_offsets=(0.2,))


# ------------------------------------------------------------------ from_geometry


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


# ------------------------------------------------------------------ shell mesh from the geometry


def _geometry(stations=None, **kw):
    options = dict(spar_cap_thickness=0.008, shear_web_thickness=0.002, web_positions=(0.18, 0.48))
    options.update(kw)
    return BladeGeometry(stations or _prismatic(), "2D-F", 0.003, **options)


@pytest.fixture(scope="module")
def geometry_blade(laminates):
    return BladeBuilder.from_geometry(_geometry(), n_span=4).build(laminates, FSDTModel())


def test_mid_surface_graph(cut):
    mid = cut.mid_surface()
    names = {s.name for s in mid.segments}
    assert {"spar_cap_suction_surface", "shear_web_0", "skin_pressure_surface:trailing_edge"} <= names
    capped = [s for s in mid.segments if s.name == "spar_cap_suction_surface"]
    # the cap's sub-millimetre reach past the webs onto the next skins is left to the skins
    assert {s.layers for s in capped} == {("spar_cap_suction_surface", "skin_suction_surface:bay_0")}
    # oriented runs: every point is left by at most one segment of each wall
    starts = [(s.start, s.name) for s in mid.segments]
    assert len(starts) == len(set(starts))


def test_geometry_mesh_sets_and_normals(geometry_blade):
    blade = geometry_blade
    assert blade.surface == "mid" and blade.geometry is not None
    for name in ("skin", "caps", "webs", "all", "spar_cap_suction_surface", "shear_web_1",
                 "skin_suction_surface", "skin_pressure_surface:leading_edge"):
        assert blade.elements(name), name
    assert set(blade.elements("caps")) <= set(blade.elements("skin"))
    assert set(blade.elements("caps")) <= set(blade.elements("skin_suction_surface")) | set(
        blade.elements("skin_pressure_surface"))
    mesh = blade.mesh
    by_id = {e.element_id: e for e in mesh.shell_elements}
    centre = np.array([0.0, 0.0])
    for eid in blade.elements("skin"):
        e = by_id[eid]
        x = np.mean([mesh.node_coordinates(n)[:2] for n in e.node_ids], axis=0)
        n3 = np.asarray(e.frame)[2]
        assert abs(n3[2]) < 1e-9 and n3[:2] @ (x - centre) > 0.0  # out of the (convex) section
    for eid in blade.elements("webs"):
        assert np.asarray(by_id[eid].frame)[2][0] > 0.99  # towards the trailing edge
    z = [mesh.node_coordinates(n)[2] for n in blade.nodes("tip")]
    assert np.allclose(z, 4.0) and blade.span_positions[-1] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="mid-surface already"):
        blade.to_mid_surface()


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


def test_geometry_mesh_follows_taper_and_twist(laminates):
    stations = [Station(0.0, 1.2, AIRFOIL, twist=8.0, pitch_axis=0.3),
                Station(4.0, 0.8, AIRFOIL, twist=0.0, pitch_axis=0.3, offset=(0.0, 0.05))]
    geo = _geometry(stations)
    builder = BladeBuilder.from_geometry(geo, n_span=2, element="quad4", wall_elements={"web": 3})
    blade = builder.build(laminates, FSDTModel())
    mesh = blade.mesh
    for z in (0.0, 2.0, 4.0):
        mid = geo.section_at(z).mid_surface()
        nodes = np.array([mesh.node_coordinates(n)[:2] for n in blade.section_nodes(z)])
        # every mesh node lies on the cut's wall mid-surface
        P = mid.points
        A, B = P[[s.start for s in mid.segments]], P[[s.end for s in mid.segments]]
        for p in nodes:
            t = np.clip(np.einsum("ij,ij->i", p - A, B - A) / np.einsum("ij,ij->i", B - A, B - A), 0.0, 1.0)
            assert np.min(np.linalg.norm(A + t[:, None] * (B - A) - p, axis=1)) < 1e-9
    web = [e for e in mesh.shell_elements if e.element_id in set(blade.elements("shear_web_0"))]
    assert len(web) == 3 * 2


def test_geometry_mesh_errors(laminates):
    lib = create_default_library()
    with pytest.raises(ValueError, match="mm thick"):
        BladeBuilder.from_geometry(_geometry(), n_span=1, element="quad4").build(
            dict(laminates, cap=Laminate.from_angles(lib.get("T300/5208"), [0] * 8, 0.5e-3)), FSDTModel())
    stations = [Station(0.0, 2.0, AIRFOIL, pitch_axis=0.3), Station(10.0, 1.0, AIRFOIL, pitch_axis=0.3)]
    dropped = BladeGeometry(stations, "2D-F", 0.004, spar_cap_thickness=0.008, shear_web_thickness=0.004,
                            web_offsets=(0.4, 1.2))
    with pytest.raises(ValueError, match="different walls"):
        BladeBuilder.from_geometry(dropped, n_span=1, element="quad4").build(laminates, FSDTModel())


def test_geometry_mesh_laminates_as_functions_of_span(laminates):
    lib = create_default_library()
    glass = lib.get("E-glass/epoxy")
    geo = BladeGeometry(_prismatic(), "2D-F", lambda z: 0.003 if z < 2.0 else 0.004, spar_cap_thickness=0.008,
                        shear_web_thickness=0.002, web_positions=(0.18, 0.48))
    thin, thick = laminates["skin"], Laminate.from_angles(glass, [0, 45, -45, 0, 0, -45, 45, 0], 0.5e-3)
    blade = BladeBuilder.from_geometry(geo, n_span=[0.0, 0.5, 1.0], element="quad4").build(
        dict(laminates, skin=lambda z: thin if z < 2.0 else thick), FSDTModel())
    t = {round(e.laminate.thickness, 6) for e in blade.mesh.shell_elements
         if e.element_id in set(blade.elements("skin_suction_surface:leading_edge"))}
    assert t == {0.003, 0.004}


from blade_geometry_model.section_geometry.layout.system_layout import SYSTEM_TYPE_KEYS  # noqa: E402


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
