"""Timoshenko 6 x 6 and Timoshenko-Vlasov 7 x 7 beam stiffness: thin-wall theory and the shell model."""

from types import SimpleNamespace

import numpy as np
import pytest

from laminate_modelling_package import (
    FEAMesh,
    FSDTModel,
    Laminate,
    Ply,
    ThinWalledSection,
    beam_section_properties,
    create_default_library,
    transform_beam_stiffness,
)
from laminate_modelling_package.laminate_materials import MaterialCard
from laminate_modelling_package.laminate_materials_3d import ElasticProperties3D, StrengthProperties3D

E, NU, T = 68.9e9, 0.33, 0.004
G = E / (2 * (1 + NU))


@pytest.fixture(scope="module")
def aluminium():
    return Laminate.from_angles(create_default_library().get("Al-6061-T6"), [0], T)


def tube(lam, R=0.5, n=128):
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return ThinWalledSection(np.c_[R * np.cos(th), R * np.sin(th)], [(k, (k + 1) % n, lam) for k in range(n)])


def test_thin_tube_timoshenko_stiffness(aluminium):
    R = 0.5
    s = tube(aluminium, R)
    K = s.timoshenko_stiffness()
    A = 2 * np.pi * R * T
    # the shear correction factor of a thin circular tube is 1/2
    assert K[0, 0] == pytest.approx(0.5 * G * A, rel=1e-3)
    assert K[1, 1] == pytest.approx(0.5 * G * A, rel=1e-3)
    assert K[2, 2] == pytest.approx(E * A, rel=1e-3)
    assert K[3, 3] == pytest.approx(E * np.pi * R**3 * T, rel=1e-3)
    assert K[5, 5] == pytest.approx(s.GJ, rel=1e-9)
    assert np.allclose(K, K.T)
    # the shear flows carry exactly the unit shear forces
    basis = s.shear_flow_basis()
    for key, (fx, fy) in (("open_x", (1, 0)), ("open_y", (0, 1))):
        q, w = basis[key], basis["w"]
        assert np.sum(q * w * s.tangent[:, :1]) == pytest.approx(fx, abs=1e-9)
        assert np.sum(q * w * s.tangent[:, 1:]) == pytest.approx(fy, abs=1e-9)


def test_channel_shear_centre_and_web_shear(aluminium):
    b, h = 0.1, 0.2
    s = ThinWalledSection(np.array([(0, 0), (b, 0), (b, h), (0, h)]), [(0, 1, aluminium), (1, 2, aluminium),
                                                                         (2, 3, aluminium)])
    C = np.linalg.inv(s.timoshenko_stiffness())
    xs, ys = -C[5, 1] / C[5, 5], C[5, 0] / C[5, 5]
    assert xs == pytest.approx(b + 3 * b**2 / (h + 6 * b), rel=1e-9)  # textbook shear centre
    assert ys == pytest.approx(h / 2, rel=1e-9)
    # open section: Saint-Venant torsion only from the walls' twisting
    K = s.timoshenko_stiffness(s.shear_centre)
    assert K[5, 5] == pytest.approx(s.GJ, rel=1e-6)
    assert abs(K[0, 5]) < 1e-6 * K[0, 0] and abs(K[1, 5]) < 1e-6 * K[1, 1]
    # vertical shear goes through the web: GA_y below G h t, above G h t times 5/6-ish
    assert 0.7 * G * h * T < K[1, 1] < G * h * T


def test_moving_the_reference_point(aluminium):
    s = tube(aluminium)
    K0 = s.timoshenko_stiffness()
    K1 = s.timoshenko_stiffness((0.1, -0.2))
    assert np.allclose(transform_beam_stiffness(K0, (0.1, -0.2)), K1, rtol=1e-8, atol=1e-6 * K0.max())
    assert np.allclose(transform_beam_stiffness(K1, (-0.1, 0.2)), K0, rtol=1e-8, atol=1e-6 * K0.max())
    # turning the axes by 90 degrees swaps the two bending and the two shear terms
    Kr = transform_beam_stiffness(s.timoshenko_stiffness((0.0, 0.0)) + np.diag([1, 2, 0, 3, 4, 0]), (0, 0), np.pi / 2)
    assert Kr[0, 0] - K0[0, 0] == pytest.approx(2, abs=1e-3) and Kr[3, 3] - K0[3, 3] == pytest.approx(4, abs=1e-2)


def test_vlasov_7x7(aluminium):
    s = tube(aluminium)
    b, h = 0.6, 0.3
    pts = [(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, h / 2), (-b / 2, h / 2)]
    box = ThinWalledSection(np.array(pts), [(k, (k + 1) % 4, aluminium) for k in range(4)])
    K7 = box.vlasov_stiffness(box.shear_centre)
    assert K7[6, 6] == pytest.approx(box.EGamma)
    assert np.allclose(K7[:6, :6], box.timoshenko_stiffness(box.shear_centre))
    assert np.allclose(K7[6, :6], 0.0)
    assert s.vlasov_stiffness()[6, 6] == pytest.approx(0.0, abs=1e-6 * s.EI_xx)


def shell_box(lam, b=0.5, h=0.25, L=3.0, nb=6, nh=3, nz=24):
    """Quad8 box on the wall mid-surfaces along z, in the form beam_section_properties takes."""
    corners = [(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, h / 2), (-b / 2, h / 2)]
    ring = []
    for k in range(4):
        p, q = np.array(corners[k]), np.array(corners[(k + 1) % 4])
        m = 2 * (nb if k % 2 == 0 else nh)
        ring += [p + (q - p) * i / m for i in range(m)]
    zs = np.linspace(0.0, L, 2 * nz + 1)
    mesh, nid, n = FEAMesh(), {}, 1
    for j, z in enumerate(zs):
        for i, p in enumerate(ring):
            if i % 2 and j % 2:
                continue
            mesh.add_node(n, (p[0], p[1], z))
            nid[(i, j)] = n
            n += 1
    nr = len(ring)
    for jj in range(nz):
        j = 2 * jj
        for ii in range(0, nr, 2):
            i2 = (ii + 2) % nr
            conn = [nid[(ii, j)], nid[(i2, j)], nid[(i2, j + 2)], nid[(ii, j + 2)],
                    nid[(ii + 1, j)], nid[(i2, j + 1)], nid[(ii + 1, j + 2)], nid[(ii, j + 1)]]
            mesh.add_element(len(mesh.element_ids) + 1, conn, lam, FSDTModel(), element_type="quad8",
                             material_axis=(0, 0, 1))
    at = lambda z: [m for m in mesh.node_ids if abs(mesh.node_coordinates(m)[2] - z) < 1e-9]  # noqa: E731
    sets = {"root": at(0.0), "tip": at(L)}
    return SimpleNamespace(mesh=mesh, span_positions=zs[::2], nodes=lambda name: sets[name], section_nodes=at,
                           element_sets={})


def test_shell_6x6_and_7x7_match_thin_wall_theory():
    """With no Poisson coupling, thin-wall theory and the shell agree on every term of the 6 x 6.

    (With Poisson's ratio the shell's shear stiffness rises a few percent above the
    thin-wall value: the bending stress's Poisson strain in the walls varies along a
    beam in shear, as in Cowper's shear coefficient, which the shear flows leave out.)
    """
    card = MaterialCard("nu0", ElasticProperties3D.transversely_isotropic(E, E, 0.0, E / 2, 0.0),
                        StrengthProperties3D(1e12, 1e12, 1e12, 1e12, 1e12), density=2700.0)
    box = shell_box(Laminate([Ply(card, T, 0.0)]), nz=32)
    p = beam_section_properties(box, reference=(0.05, 0.02), warping=True, end_ribs=0.2)
    s = ThinWalledSection.from_blade_mesh(box, z=1.5)
    Cs, Ct = p.compliance_6x6, np.linalg.inv(s.timoshenko_stiffness((0.05, 0.02)))
    assert np.diag(Cs) == pytest.approx(np.diag(Ct), rel=0.005)
    assert Cs[0, 5] == pytest.approx(Ct[0, 5], rel=0.01) and Cs[1, 5] == pytest.approx(Ct[1, 5], rel=0.01)
    assert np.allclose(p.shear_centre, s.shear_centre, atol=1e-6)
    # the classical 4 x 4 is the 6 x 6 with the shear forces free
    assert np.allclose(p.stiffness, np.linalg.inv(Cs[2:, 2:]))
    assert p.stiffness_7x7.shape == (7, 7)
    # the restrained twist of the rigid-section shell gives EΓ and Benscoter's S
    # (converged with the span mesh: -3.5% and -10%)
    assert p.EGamma == pytest.approx(s.EGamma, rel=0.06)
    assert p.warping_shear_stiffness == pytest.approx(s.S_warping, rel=0.15)
