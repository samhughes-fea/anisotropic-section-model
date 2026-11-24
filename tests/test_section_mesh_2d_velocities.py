"""Contract 2.1: node velocities and tie points survive ``section_mesh_2d.txt``."""

from __future__ import annotations

import numpy as np
import pytest

from anisotropic_section_model.solid.from_geometry import write_section_mesh_2d
from anisotropic_section_model.solid.layup_sensitivity import solid_section_derivative
from anisotropic_section_model.solid.mesh import SectionMesh, parse_section_mesh_2d
from anisotropic_section_model.solid.section_analysis import SolidSectionAnalysis
from anisotropic_section_model.thin_walled import ThinWalledSection, Wall, WallLaminate
from anisotropic_section_model.thin_walled.laminate import Ply, PlyMaterial
from anisotropic_section_model.thin_walled.layup_sensitivity import LayupParameter

CFRP = PlyMaterial(E1=135e9, E2=10e9, G12=5e9, nu12=0.3, G13=5e9, G23=3.8e9, rho=1550.0)


def _mesh():
    """A box whose webs are thicker: butt-jointed corners, so the mesh has ties."""
    pts = {1: (-0.1, -0.05), 2: (0.1, -0.05), 3: (0.1, 0.05), 4: (-0.1, 0.05)}
    walls = [Wall(1, 1, 2, 1), Wall(2, 2, 3, 2), Wall(3, 3, 4, 1), Wall(4, 4, 1, 2)]
    lams = {1: WallLaminate([Ply(CFRP, a, 1e-3) for a in (30, -30, 0)]),
            2: WallLaminate([Ply(CFRP, a, 1.5e-3) for a in (45, -45, 0)])}
    mesh = SectionMesh.from_thin_walled(ThinWalledSection(pts, walls), lams, elements_along=2)
    mesh.section_id = 4
    return mesh


def test_velocities_and_tie_points_round_trip(tmp_path):
    mesh = _mesh()
    assert mesh.ties and mesh.tie_points and mesh.velocities
    path = tmp_path / "section_mesh_2d.txt"
    write_section_mesh_2d(str(path), {4: mesh})
    text = path.read_text()
    assert "[Section_Node_Velocity]" in text and "[Section_Tie_Point]" in text
    back = parse_section_mesh_2d(str(path), {})[4]
    for key, V in mesh.velocities.items():
        if np.any(V):
            np.testing.assert_array_equal(back.velocities[key], V)
        else:
            assert not np.any(back.velocities.get(key, 0.0))
    assert back.tie_points == [(int(n), int(e), float(x), float(y)) for n, e, x, y in mesh.tie_points]


@pytest.mark.parametrize("ply", [(1, 0), (2, 2)])
def test_thickness_derivative_of_a_mesh_read_from_file(tmp_path, ply):
    mesh = _mesh()
    path = tmp_path / "section_mesh_2d.txt"
    write_section_mesh_2d(str(path), {4: mesh})
    back = parse_section_mesh_2d(str(path), {})[4]
    p = LayupParameter(*ply, "thickness")
    d0 = solid_section_derivative(SolidSectionAnalysis(mesh), p)
    d1 = solid_section_derivative(SolidSectionAnalysis(back), p)
    np.testing.assert_allclose(d1.stiffness, d0.stiffness, rtol=1e-12, atol=1e-12 * np.abs(d0.stiffness).max())
    np.testing.assert_allclose(d1.mass_matrix, d0.mass_matrix, rtol=1e-12, atol=1e-14)


def test_a_mesh_without_velocities_writes_no_blocks(tmp_path):
    mesh = SectionMesh.rectangle(0.1, 0.02, 4, 2, _mesh().materials[1])
    path = tmp_path / "m.txt"
    write_section_mesh_2d(str(path), {1: mesh})
    assert "[Section_Node_Velocity]" not in path.read_text()
    back = parse_section_mesh_2d(str(path), {})[1]
    assert back.velocities is None and back.tie_points is None
