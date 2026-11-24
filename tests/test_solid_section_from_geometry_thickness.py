"""Ply-thickness sensitivities of 2D blade cuts (``section_mesh_from_cut`` velocities).

The mesh tags every element with its laminate ply and gives each ply's node velocity
(``SectionMesh.velocities``): skins stay centred on the skin mid-line, caps stack inward
from the skin, webs stay centred on their mid-line and their feet slide along the inner
face (carrying a cap edge that ends under a web with them). The derivatives are checked
against meshes moved by ``h V`` (same topology, so the finite difference is clean), and
against meshes rebuilt with the thicker ply, which re-run the mesher's own geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("implicit_geometry_kernel")

from implicit_geometry_kernel import Airfoil, BladeGeometry, Station  # noqa: E402

from anisotropic_section_model.solid.from_geometry import (  # noqa: E402
    section_mesh_from_cut,
    write_section_mesh_2d,
)
from anisotropic_section_model.solid.layup_sensitivity import solid_section_derivative  # noqa: E402
from anisotropic_section_model.solid.mesh import SectionMesh, parse_section_mesh_2d  # noqa: E402
from anisotropic_section_model.solid.section_analysis import SolidSectionAnalysis  # noqa: E402
from anisotropic_section_model.thin_walled import Ply, PlyMaterial, WallLaminate  # noqa: E402
from anisotropic_section_model.thin_walled.layup_sensitivity import LayupParameter  # noqa: E402

GLASS = PlyMaterial(E1=38.6e9, E2=8.27e9, G12=4.14e9, nu12=0.26, G13=4.14e9, G23=3.063e9, rho=1800.0)
CARBON = PlyMaterial(E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28, G13=7.17e9, G23=3.679e9, rho=1600.0)


def lam(mat, angles, t=0.5e-3):
    return WallLaminate([Ply(mat, a, t) for a in angles])


SKIN, WEB, CAP = lam(GLASS, [0, 45, -45, -45, 45, 0]), lam(GLASS, [45, -45, -45, 45]), lam(CARBON, [0] * 16)
LIBRARY = {11: SKIN, 12: WEB, 13: CAP}
BINDING = {"skin": 11, "web": 12, "cap": 13}
H = 0.02


def _stations():
    af = Airfoil.naca4("0024")
    return [Station(0.0, 1.0, af, pitch_axis=0.3), Station(6.0, 1.0, af, pitch_axis=0.3)]


def _cut():
    return BladeGeometry(_stations(), "2D-F-LC-0R", 0.003, 0.008, 0.002, web_positions=(0.18, 0.48),
                         pivot_fraction=0.3).section_at(3.0)


@pytest.fixture(scope="module")
def box():
    cut = _cut()
    mesh = section_mesh_from_cut(cut, BINDING, LIBRARY, element_size=H, section_id=3)
    return cut, mesh, SolidSectionAnalysis(mesh)


def _moved(mesh, step, V):
    return SectionMesh(mesh.nodes + step * V, mesh.elements, mesh.kinds, mesh.material_ids, mesh.theta,
                       mesh.alpha, mesh.materials, plies=mesh.plies)


def test_cut_mesh_carries_ply_tags_and_velocities(box):
    _, mesh, _ = box
    # library laminates keep their ids, and every ply of every laminate has a velocity
    assert mesh.laminates == {11: SKIN, 12: WEB, 13: CAP}
    assert set(mesh.velocities) == {(lid, j) for lid, lm in LIBRARY.items() for j in range(len(lm.plies))}
    assert mesh.tie_points == [] and not mesh.ties
    assert all(t is not None for t in mesh.plies)
    for e, (lid, j) in enumerate(mesh.plies):
        assert mesh.theta[e] == LIBRARY[lid].plies[j].angle_deg
    # a skin ply moves the outer face out by half its change (the stack stays on the mid-line)
    V = mesh.velocities[(11, 2)]
    speed = np.linalg.norm(V, axis=1)
    assert speed.max() == pytest.approx(0.5, rel=0.05)


def test_unnamed_laminates_get_fresh_ids():
    cut = _cut()
    mesh = section_mesh_from_cut(cut, {"skin": SKIN, "web": 12, "cap": CAP}, {12: WEB}, element_size=0.04)
    assert mesh.laminates[12] is WEB
    assert {id(v) for k, v in mesh.laminates.items() if k != 12} == {id(SKIN), id(CAP)}
    assert min(k for k in mesh.laminates if k != 12) == 13


@pytest.mark.parametrize("param", [(11, 0), (11, 5), (12, 1), (13, 0), (13, 15)])
def test_thickness_derivative_matches_moved_meshes(box, param):
    _, mesh, an = box
    V = mesh.velocities[param]
    d = solid_section_derivative(an, LayupParameter(*param, "thickness"))
    step = 1e-6
    fd = (SolidSectionAnalysis(_moved(mesh, step, V)).stiffness
          - SolidSectionAnalysis(_moved(mesh, -step, V)).stiffness) / (2 * step)
    np.testing.assert_allclose(d.stiffness, fd, rtol=0, atol=1e-5 * np.abs(fd).max())


def _thicker(lid, ply, dt):
    lm = LIBRARY[lid]
    plies = list(lm.plies)
    plies[ply] = Ply(plies[ply].material, plies[ply].angle_deg, plies[ply].thickness + dt)
    return {**LIBRARY, lid: WallLaminate(plies)}


@pytest.mark.parametrize("param", [(11, 0), (12, 0), (13, 15)])
def test_thickness_derivative_matches_rebuilt_meshes(box, param):
    """Rebuilding re-runs the mesher (web feet, cap edges carried to a web's far face): the
    diagonal stiffnesses agree to within 0.5 %."""
    cut, _, an = box
    d = solid_section_derivative(an, LayupParameter(*param, "thickness"))
    dt = 2e-6
    Kp, Km = (SolidSectionAnalysis(section_mesh_from_cut(cut, BINDING, _thicker(*param, s * dt),
                                                         element_size=H)).stiffness for s in (1, -1))
    fd = np.diag(Kp - Km) / (2 * dt)
    np.testing.assert_allclose(np.diag(d.stiffness), fd, rtol=5e-3)


def test_velocities_survive_section_mesh_2d(box, tmp_path):
    _, mesh, an = box
    path = tmp_path / "section_mesh_2d.txt"
    write_section_mesh_2d(str(path), {3: mesh})
    back = parse_section_mesh_2d(str(path), {})[3]
    assert back.plies == mesh.plies
    for key, V in mesh.velocities.items():
        np.testing.assert_allclose(back.velocities[key], V, atol=1e-15)
    p = LayupParameter(12, 2, "thickness")
    np.testing.assert_allclose(solid_section_derivative(SolidSectionAnalysis(back), p).stiffness,
                               solid_section_derivative(an, p).stiffness, rtol=1e-9, atol=1e-6)


def test_velocities_can_be_left_out():
    mesh = section_mesh_from_cut(_cut(), BINDING, LIBRARY, element_size=0.04, velocities=False)
    assert mesh.velocities is None and mesh.plies is not None
