"""TimoshenkoBeam against closed forms, and the GBT section, axial and distributed loads."""

import numpy as np
import pytest

from laminate_modelling_package import (GeneralizedBeam, Laminate, ThinWalledSection, TimoshenkoBeam,
                                        create_default_library, euler_bernoulli_beam)

EI, GA, EA, GJ, L = 2e6, 5e5, 1e8, 1e6, 3.0
K = np.diag([GA, GA, EA, EI, EI, GJ])


def test_cantilever_tip_force():
    s = TimoshenkoBeam(K, L, 3).clamp(0.0).add_force(L, [0.0, 1.0, 0.0]).solve()
    assert s.node_displacement(L)[1] == pytest.approx(L**3 / (3 * EI) + L / GA, rel=1e-10)
    # gamma_y = u_y' + theta_x, so the section turns by -F L^2 / 2EI at the tip
    assert s.node_displacement(L)[3] == pytest.approx(-L**2 / (2 * EI), rel=1e-10)
    z = 1.5
    assert s.displacement(z)[0, 1] == pytest.approx(z**2 * (3 * L - z) / (6 * EI) + z / GA, rel=1e-10)
    assert s.reaction(0.0)[1] == pytest.approx(-1.0)
    assert s.section_forces(0.0)[0, 4] == pytest.approx(0.0, abs=1e-12)
    assert abs(s.section_forces(0.0)[0, 3]) == pytest.approx(L)


def test_uniform_load_exact_with_one_element():
    s = TimoshenkoBeam(K, L, 1).clamp(0.0).add_distributed([0.0, 1.0, 0.0]).solve()
    assert s.node_displacement(L)[1] == pytest.approx(L**4 / (8 * EI) + L**2 / (2 * GA), rel=1e-10)
    z = 1.0
    exact = z**2 * (6 * L**2 - 4 * L * z + z**2) / (24 * EI) + (L * z - z**2 / 2) / GA
    assert s.displacement(z)[0, 1] == pytest.approx(exact, rel=1e-10)


def test_simply_supported_uniform_load():
    s = TimoshenkoBeam(K, L, 2).pin(0.0).pin(L, axial=False).add_distributed([0.0, 1.0, 0.0]).solve()
    assert s.node_displacement(L / 2)[1] == pytest.approx(5 * L**4 / (384 * EI) + L**2 / (8 * GA), rel=1e-10)


def test_euler_bernoulli_drops_the_shear():
    s = TimoshenkoBeam(K, L, 3, shear=False).clamp(0.0).add_force(L, [0.0, 1.0, 0.0]).solve()
    assert s.node_displacement(L)[1] == pytest.approx(L**3 / (3 * EI), rel=1e-10)
    s4 = euler_bernoulli_beam(np.diag([EA, EI, EI, GJ]), L).clamp(0.0).add_force(L, [0.0, 1.0, 0.0]).solve()
    assert s4.node_displacement(L)[1] == pytest.approx(L**3 / (3 * EI), rel=1e-10)


def test_coupled_stiffness_is_mesh_independent():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    Kc = (A @ A.T + 6 * np.eye(6)) * 1e5
    f = [0.3, 1.0, -0.2, 0.1, 0.2, 0.5]
    s1 = TimoshenkoBeam(Kc, L, 1).clamp(0.0).add_force(L, f).solve()
    s7 = TimoshenkoBeam(Kc, L, 7).clamp(0.0).add_force(L, f).solve()
    assert np.allclose(s1.node_displacement(L), s7.node_displacement(L), rtol=1e-9)
    assert np.allclose(s1.displacement(1.1), s7.displacement(1.1), rtol=1e-9)


def test_off_axis_force_adds_its_moment():
    s = TimoshenkoBeam(K, L, 2).clamp(0.0).add_force(L, [0.0, 0.0, 1.0], point=(0.0, 0.2)).solve()
    assert s.node_displacement(L)[3] == pytest.approx(0.2 * L / EI, rel=1e-10)


@pytest.fixture(scope="module")
def box():
    al = Laminate.from_angles(create_default_library().get("Al-6061-T6"), [0], 0.004)
    b, h = 0.5, 0.25
    pts = [(-b / 2, -h / 2), (0, -h / 2), (b / 2, -h / 2), (b / 2, 0), (b / 2, h / 2), (0, h / 2), (-b / 2, h / 2),
           (-b / 2, 0)]
    return ThinWalledSection(np.array(pts), [(k, (k + 1) % 8, al) for k in range(8)])


def test_gbt_section_and_wall_loads_match_a_slender_beam(box):
    Lb = 10.0
    K6 = box.timoshenko_stiffness(box.shear_centre)
    g = GeneralizedBeam(box, Lb, warping="classical", distortional_modes=0, n_elements=20).clamp(0.0).diaphragm(Lb)
    tip = g.add_section_force(Lb, (0.0, 1.0)).solve().section_displacement(Lb)[:, 1].mean()
    ref = TimoshenkoBeam(K6, Lb, 2).clamp(0.0).add_force(Lb, [0.0, 1.0, 0.0]).solve().node_displacement(Lb)[1]
    assert tip == pytest.approx(ref, rel=0.005)
    top = [k for k in range(8) if box.points[box.start[k], 1] > 0.1 and box.points[box.end[k], 1] > 0.1]
    g = GeneralizedBeam(box, Lb, warping="classical", distortional_modes=0, n_elements=20).clamp(0.0)
    tip = g.add_distributed_wall_load(top, (0.0, 1.0)).solve().section_displacement(Lb)[:, 1].mean()
    ref = TimoshenkoBeam(K6, Lb, 2).clamp(0.0).add_distributed([0.0, 1.0, 0.0]).solve().node_displacement(Lb)[1]
    assert tip == pytest.approx(ref, rel=0.005)


@pytest.mark.parametrize("extensible", [False, True])
def test_gbt_axial_force_and_poisson_at_the_clamp(box, extensible):
    Lb = 10.0
    s = GeneralizedBeam(box, Lb, warping="full", distortional_modes=None, n_elements=40,
                        extensible=extensible).clamp(0.0).add_axial_force(Lb, 1.0).solve()
    r = s.wall_resultants(Lb / 2)
    total = sum(x[:, 0].mean() * box.length[k] for k, x in enumerate(r))
    assert total == pytest.approx(1.0, rel=1e-9)
    assert np.allclose([x[:, 0] for x in r], 1.0 / box.length.sum(), rtol=1e-6)  # uniform for one material
    if extensible:
        # the clamp stops the walls contracting: hoop strain zero, so N_ss = nu N_zz
        for x in s.wall_resultants(0.0):
            assert np.allclose(x[:, 1] / x[:, 0], 0.33, rtol=1e-3)
        assert np.allclose([x[:, 1] for x in r], 0.0, atol=1e-6 / box.length.sum())
