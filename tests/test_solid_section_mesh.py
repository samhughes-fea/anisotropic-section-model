"""2D section meshes and 3D materials (the inputs of the 2D section solver)."""

from __future__ import annotations

import numpy as np
import pytest

from pre_processing.section_integration.solid.elements import quadrature, shape
from pre_processing.section_integration.solid.material import SolidMaterial, bond_matrix, material_axes
from pre_processing.section_integration.solid.mesh import SectionMesh, parse_section_mesh_2d, read_gmsh
from pre_processing.section_integration.thin_walled import ThinWalledSection, Wall, WallLaminate
from pre_processing.section_integration.thin_walled.laminate import Ply, PlyMaterial

CFRP = PlyMaterial(E1=135e9, E2=10e9, G12=5e9, nu12=0.3, G13=5e9, G23=3.8e9, rho=1550.0)
NODES = {"Q8": np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]], float),
         "T6": np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]], float)}


# ------------------------------------------------------------------ elements
@pytest.mark.parametrize("kind", ["Q8", "T6"])
def test_shape_functions(kind):
    X = NODES[kind]
    for a, p in enumerate(X):            # interpolating: N_b(node a) = delta_ab
        N, _ = shape(kind, *p)
        np.testing.assert_allclose(N, np.eye(len(X))[a], atol=1e-14)
    p = np.array([0.2, 0.3])
    N, dN = shape(kind, *p)
    assert N.sum() == pytest.approx(1.0)
    h = 1e-6
    for d in range(2):
        e = np.zeros(2)
        e[d] = h
        fd = (shape(kind, *(p + e))[0] - shape(kind, *(p - e))[0]) / (2 * h)
        np.testing.assert_allclose(dN[d], fd, atol=1e-8)


@pytest.mark.parametrize("kind,exact", [("Q8", lambda i, j: (1 - (-1) ** (i + 1)) / (i + 1) * (1 - (-1) ** (j + 1)) / (j + 1)),
                                        ("T6", None)])
def test_quadrature_is_exact_to_degree_five(kind, exact):
    from math import factorial

    pts, w = quadrature(kind)
    for i in range(6):
        for j in range(6 - i if kind == "T6" else 6):
            val = w @ (pts[:, 0] ** i * pts[:, 1] ** j)
            ref = exact(i, j) if exact else factorial(i) * factorial(j) / factorial(i + j + 2)
            assert val == pytest.approx(ref, abs=1e-12)


# ------------------------------------------------------------------ material
def test_material_rotation_conventions():
    m = SolidMaterial.from_ply(CFRP)
    C = m.stiffness()
    # theta = 0, alpha = 0: e1 = x, e2 = y, e3 = z
    np.testing.assert_allclose(m.stiffness_in_section(0, 0), C, rtol=1e-12)
    # fibre along the wall direction s = y
    assert m.stiffness_in_section(90, 0)[1, 1] == pytest.approx(C[0, 0])
    # a wall along z: e2 = s = z
    assert m.stiffness_in_section(0, 90)[2, 2] == pytest.approx(C[1, 1])
    R = material_axes(30, 40)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-14)
    assert np.linalg.det(R) == pytest.approx(1.0)
    iso = SolidMaterial.isotropic(70e9, 0.33)
    np.testing.assert_allclose(iso.stiffness_in_section(37, -71), iso.stiffness(), rtol=1e-10, atol=1e-3)
    # bond matrix of a composition is the composition of bond matrices
    R2 = material_axes(-15, 100)
    np.testing.assert_allclose(bond_matrix(R @ R2), bond_matrix(R) @ bond_matrix(R2), atol=1e-12)


@pytest.mark.parametrize("theta,alpha", [(0, 0), (30, 0), (-45, 60), (90, 135)])
def test_plane_stress_of_the_3d_material_is_the_ply_stiffness(theta, alpha):
    """Condensing sigma_nn = sigma_sn = sigma_xn = 0 out of the 3D stiffness, in wall axes,
    gives the thin-walled ply's Q_bar: the two models put the fibre in the same place."""
    C = SolidMaterial.from_ply(CFRP).stiffness_in_section(theta, alpha)
    al = np.deg2rad(alpha)
    wall = np.column_stack([[1, 0, 0], [0, np.cos(al), np.sin(al)], [0, -np.sin(al), np.cos(al)]])  # x, s, n
    Ti = np.linalg.inv(bond_matrix(wall))
    Cw = Ti @ C @ Ti.T                                     # stiffness in wall axes [xx, ss, nn, sn, xn, xs]
    keep, drop = [0, 1, 5], [2, 3, 4]
    Q = Cw[np.ix_(keep, keep)] - Cw[np.ix_(keep, drop)] @ np.linalg.solve(Cw[np.ix_(drop, drop)], Cw[np.ix_(drop, keep)])
    np.testing.assert_allclose(Q, Ply(CFRP, theta, 1e-3).rotated_stiffness(), rtol=1e-9, atol=1e-3)


# ------------------------------------------------------------------ meshes
def test_rectangle_properties():
    b, h = 0.04, 0.1
    m = SectionMesh.rectangle(b, h, 3, 5, SolidMaterial.isotropic(1.0, 0.3), centre=(0.01, -0.02))
    p = m.properties()
    assert p["A"] == pytest.approx(b * h)
    assert p["S_z"] == pytest.approx(0.01 * b * h)
    assert p["I_yy"] == pytest.approx(b * h**3 / 12 + b * h * 0.02**2)
    assert p["I_zz"] == pytest.approx(h * b**3 / 12 + b * h * 0.01**2)
    assert p["I_yz"] == pytest.approx(b * h * 0.01 * -0.02)


def _i_section():
    pts = {1: (-0.075, 0.15), 2: (0.0, 0.15), 3: (0.075, 0.15), 4: (-0.075, -0.15), 5: (0.0, -0.15), 6: (0.075, -0.15)}
    walls = [Wall(1, 1, 2, 1), Wall(2, 2, 3, 1), Wall(3, 4, 5, 1), Wall(4, 5, 6, 1), Wall(5, 5, 2, 2)]
    lams = {1: WallLaminate([Ply(CFRP, a, 1e-3) for a in (0, 45, -45, 0, 0, -45, 45, 0)]),
            2: WallLaminate([Ply(CFRP, a, 1e-3) for a in (45, -45, -45, 45)])}
    return ThinWalledSection(pts, walls), lams


def test_thin_walled_section_as_a_2d_mesh():
    sec, lams = _i_section()
    m = SectionMesh.from_thin_walled(sec, lams, elements_along=6)
    p = m.properties()
    # each strip is its wall's mid-line length times its thickness, so the junctions count
    # twice, exactly as in the mid-line model
    A_mid = 0.15 * 2 * 8e-3 + 0.3 * 4e-3
    assert p["A"] == pytest.approx(A_mid, rel=1e-10)
    assert p["S_y"] == pytest.approx(0.0, abs=1e-12) and p["S_z"] == pytest.approx(0.0, abs=1e-12)
    # one row of elements per ply, each with its own ply angle; the flange walls lie along y, the web along z
    assert len(m.elements) == 4 * 6 * 8 + 6 * 4
    web = [e for e, a in enumerate(m.alpha) if abs(a - 90.0) < 1e-9]
    assert len(web) == 6 * 4 and sorted({m.theta[e] for e in web}) == [-45.0, 45.0]
    # walls share the nodes at the section points
    assert m.nodes.shape[0] == len(np.unique(np.round(m.nodes, 12), axis=0))


GMSH = """$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
1
2 7 "core"
$EndPhysicalNames
$Entities
0 0 1 0
3 0 0 0 1 1 0 1 7 0
$EndEntities
$Nodes
1 8 1 8
2 3 0 8
1
2
3
4
5
6
7
8
0 0 0
2 0 0
2 1 0
0 1 0
1 0 0
2 0.5 0
1 1 0
0 0.5 0
$EndNodes
$Elements
1 1 1 1
2 3 16 1
1 1 2 3 4 5 6 7 8
$EndElements
"""


def test_gmsh_reader(tmp_path):
    path = tmp_path / "s.msh"
    path.write_text(GMSH)
    nodes, elems = read_gmsh(str(path))
    assert nodes.shape == (8, 2) and len(elems) == 1 and elems[0][1] == "core" and elems[0][2] == "Q8"
    m = SectionMesh.from_gmsh(str(path), {"core": (3, 0.0, 0.0)}, {3: SolidMaterial.isotropic(1.0, 0.25)})
    assert m.properties()["A"] == pytest.approx(2.0)


def test_section_mesh_2d_file(tmp_path):
    path = tmp_path / "section_mesh_2d.txt"
    path.write_text(
        "[Section_Node]\n[section_id] [node_id] [y] [z]\n"
        "4 1 0 0\n4 2 1 0\n4 3 0 1\n4 4 0.5 0\n4 5 0.5 0.5\n4 6 0 0.5\n"
        "[Section_Element]\n[section_id] [element_id] [type] [nodes] [material_id] [theta] [alpha]\n"
        "4 1 T6 1 2 3 4 5 6 9 30 0\n"
        "[Solid_Material]\n[material_id] [E1] [E2] [E3] [G12] [G13] [G23] [nu12] [nu13] [nu23] [rho]\n"
        "9 100 10 10 5 5 4 0.3 0.3 0.25 1000\n")
    meshes = parse_section_mesh_2d(str(path), {})
    m = meshes[4]
    assert m.properties()["A"] == pytest.approx(0.5)
    assert m.theta == [30.0] and m.materials[9].E1 == 100.0


def test_inverted_elements_are_refused():
    X = NODES["Q8"][[1, 0, 3, 2, 4, 7, 6, 5]]               # clockwise
    with pytest.raises(ValueError, match="inverted"):
        SectionMesh(X, [np.arange(8)], ["Q8"], [1], [0.0], [0.0], {1: SolidMaterial.isotropic(1.0, 0.3)})


def test_wall_junctions_are_tied():
    sec, lams = _i_section()
    m = SectionMesh.from_thin_walled(sec, lams, elements_along=6)
    # the web (listed last) meets the flanges at points 2 and 5: its end-edge nodes off the
    # flange mid-line node are tied into the flange strips
    assert len(m.ties) == 2 * (2 * 4 + 1 - 1)
    for n, row in m.ties.items():
        assert sum(c for _, c in row) == pytest.approx(1.0)
        assert all(mm not in m.ties for mm, _ in row)
        p = sum(c * m.nodes[mm] for mm, c in row)
        # inside the flange the tie reproduces the node's position
        if abs(abs(m.nodes[n][1]) - 0.15) <= 4e-3:
            np.testing.assert_allclose(p, m.nodes[n], atol=1e-12)
