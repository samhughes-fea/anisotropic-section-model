"""2D section meshes of implicit-geometry-kernel cuts, against the thin-walled section of the same cut.

On thin isotropic walls the 2D and thin-walled (``hoop="free"``) models agree to the
order of ``t / b`` at the corners. The composite 2D-F section also differs by the cap
material the mesher puts under the webs: a cap that ends under a web is carried to the
web's far face, so the web lands on one face (about +0.6 % in EA here).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("implicit_geometry_kernel")

from implicit_geometry_kernel import Airfoil, BladeGeometry, Station  # noqa: E402

from pre_processing.section_integration.solid.from_geometry import (  # noqa: E402
    section_mesh_from_cut,
    write_section_mesh_2d,
)
from pre_processing.section_integration.solid.mesh import parse_section_mesh_2d  # noqa: E402
from pre_processing.section_integration.solid.section_analysis import SolidSectionAnalysis  # noqa: E402
from pre_processing.section_integration.thin_walled import Ply, PlyMaterial, WallLaminate  # noqa: E402
from pre_processing.section_integration.thin_walled.from_geometry import thin_walled_from_cut  # noqa: E402

GLASS = PlyMaterial(E1=38.6e9, E2=8.27e9, G12=4.14e9, nu12=0.26, G13=4.14e9, G23=3.063e9, rho=1800.0)
CARBON = PlyMaterial(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28, G13=7.17e9, G23=3.679e9, rho=1600.0)
ISO = PlyMaterial(E1=70e9, E2=70e9, G12=26.9e9, nu12=0.3, G13=26.9e9, G23=26.9e9, rho=2700.0)


def lam(mat, angles, t=0.5e-3):
    return WallLaminate([Ply(mat, a, t) for a in angles])


COMPOSITE = {"skin": lam(GLASS, [0, 45, -45, -45, 45, 0]), "web": lam(GLASS, [45, -45, -45, 45]),
             "cap": lam(CARBON, [0] * 16)}
ISOTROPIC = {"skin": lam(ISO, [0] * 6), "web": lam(ISO, [0] * 4), "cap": lam(ISO, [0] * 16)}


def stations():
    af = Airfoil.naca4("0024")
    return [Station(0.0, 1.0, af, pitch_axis=0.3), Station(6.0, 1.0, af, pitch_axis=0.3)]


def numbers(a):
    K = a.classical_stiffness
    yt, zt = a.tension_centre
    EA = K[0, 0]
    return {"EA": EA, "EI_flap": K[1, 1] - EA * zt**2, "EI_edge": K[2, 2] - EA * yt**2,
            "GJ": 1.0 / a.flexibility[5, 5], "shear_centre_y": a.shear_centre[0]}


def compare(cut, lams, h):
    mesh = section_mesh_from_cut(cut, lams, element_size=h)
    return mesh, numbers(SolidSectionAnalysis(mesh)), numbers(thin_walled_from_cut(cut, lams).analysis())


@pytest.mark.parametrize("layout", ["0A", "2A-F"])
def test_isotropic_layouts_match_thin_walled(layout):
    cut = BladeGeometry(stations(), layout, 0.003, 0.008, 0.002).section_at(3.0)
    mesh, a, b = compare(cut, ISOTROPIC, 0.01)
    for key in ("EA", "EI_flap", "EI_edge", "GJ"):
        assert a[key] == pytest.approx(b[key], rel=5e-3), key
    assert a["shear_centre_y"] == pytest.approx(b["shear_centre_y"], abs=1e-3)      # 0.1 % of chord
    # the mesh conforms: every interior edge is shared, so there are no ties
    assert not mesh.ties


def test_composite_box_spar_section():
    cut = BladeGeometry(stations(), "2D-F", 0.003, 0.008, 0.002, web_positions=(0.18, 0.48)).section_at(3.0)
    mesh, a, b = compare(cut, COMPOSITE, 0.02)
    for key in ("EA", "EI_flap", "EI_edge", "GJ"):
        assert a[key] == pytest.approx(b[key], rel=0.02), key
    assert a["shear_centre_y"] == pytest.approx(b["shear_centre_y"], abs=1e-3)
    assert {"shear_web_0", "shear_web_1", "spar_cap_suction_surface"} <= set(mesh.labels)
    # one row of elements per ply: the cap carries 16 rows, the skin 6, the webs 4
    n_cap = sum(1 for lb in mesh.labels if lb == "spar_cap_suction_surface")
    assert n_cap % 16 == 0


def test_mesh_refinement_converges():
    cut = BladeGeometry(stations(), "2D-F", 0.003, 0.008, 0.002, web_positions=(0.18, 0.48)).section_at(3.0)
    coarse = numbers(SolidSectionAnalysis(section_mesh_from_cut(cut, COMPOSITE, element_size=0.02)))
    fine = numbers(SolidSectionAnalysis(section_mesh_from_cut(cut, COMPOSITE, element_size=0.01)))
    for key in ("EA", "EI_flap", "GJ"):
        assert coarse[key] == pytest.approx(fine[key], rel=2e-3), key


def test_write_and_read_section_mesh_2d(tmp_path):
    cut = BladeGeometry(stations(), "0A", 0.003).section_at(1.0)
    mesh = section_mesh_from_cut(cut, ISOTROPIC, element_size=0.03, section_id=7)
    path = tmp_path / "section_mesh_2d.txt"
    write_section_mesh_2d(str(path), {7: mesh})
    back = parse_section_mesh_2d(str(path), {})[7]
    assert len(back.elements) == len(mesh.elements)
    np.testing.assert_allclose(back.nodes, mesh.nodes)
    assert back.properties()["A"] == pytest.approx(mesh.properties()["A"], rel=1e-12)


def test_missing_laminate_names_the_wall():
    cut = BladeGeometry(stations(), "1A-F", 0.003, 0.008, 0.002).section_at(1.0)
    with pytest.raises(KeyError, match="shear_web_0"):
        section_mesh_from_cut(cut, {"skin": ISOTROPIC["skin"]})
