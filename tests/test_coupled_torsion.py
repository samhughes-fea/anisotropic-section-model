"""Torsion from the coupled section analysis: warping, Benscoter S, GJ_free and the 7 x 7 couplings."""

from types import SimpleNamespace

import numpy as np
import pytest

from anisotropic_section_model import GeneralizedBeam, RestrainedTorsionBeam, ThinWalledSection, TimoshenkoBeam
from shell_fem import (BoundaryConditions, Laminate, LoadCase, StructuralGlobalSolver, add_reference_node,
    create_default_library, rigid_link)

from test_torsion import _shell_box, box


@pytest.fixture(scope="module")
def library():
    return create_default_library()


def _carbon(library, angles, t=0.001):
    return Laminate.from_angles(library.get("T300/5208"), angles, t)


@pytest.mark.parametrize("angles", [[0], [0, 45, -45, 0]])
def test_balanced_walls_keep_the_classical_warping(library, angles):
    lam = Laminate.from_angles(library.get("Al-6061-T6"), [0], 0.004) if angles == [0] else _carbon(library, angles)
    s = box(lam)
    c = s.classical
    scale = np.max(np.abs(c["omega"]))
    assert np.allclose(s.omega, c["omega"], atol=1e-9 * scale)
    assert s.S_warping == pytest.approx(c["S_warping"], rel=1e-9)
    assert s.Ip == pytest.approx(c["Ip"], rel=1e-4)
    assert s.GJ_free == pytest.approx(s.GJ, rel=1e-5)
    K7 = s.vlasov_stiffness(s.shear_centre)
    assert np.allclose(K7[6, :6], 0.0, atol=1e-9 * np.sqrt(K7[6, 6] * np.max(np.diag(K7))))


def test_unbalanced_box_twists_by_gj_free(library):
    # every wall the same off-axis layup round the contour: extension-twist coupling (CUS)
    s = box(_carbon(library, [30, 30, 30, 30]))
    assert s.GJ_free < 0.6 * s.GJ  # the coupling roughly halves the free torsional stiffness
    # a 6 x 6 beam under a tip torque alone twists at T / GJ_free
    L, T = 4.0, 100.0
    tip = TimoshenkoBeam(s.timoshenko_stiffness(s.shear_centre), L, n_elements=4).clamp(0.0) \
        .add_force(L, [0, 0, 0, 0, 0, T]).solve()
    assert tip.node_displacement(L)[5] == pytest.approx(T * L / s.GJ_free, rel=1e-8)
    free = RestrainedTorsionBeam.from_section(s, L)
    held = RestrainedTorsionBeam.from_section(s, L, other_strains="held")
    assert free.GJ == pytest.approx(s.GJ_free) and held.GJ == pytest.approx(s.GJ)
    with pytest.raises(ValueError):
        RestrainedTorsionBeam.from_section(s, L, other_strains="clamped")


def test_unbalanced_box_against_shell(library):
    """A shell box of one off-axis layup under a tip torque twists as GJ_free says, not GJ."""
    lam = _carbon(library, [30, 30, 30, 30])
    b, h, L = 0.4, 0.2, 3.2
    mesh, sets = _shell_box(lam, b, h, L, 4, 2, 24)
    blade = SimpleNamespace(mesh=mesh, element_sets=sets, span_positions=np.linspace(0, L, 13))
    sec = ThinWalledSection.from_blade_mesh(blade, z=L)
    root = [n for n in mesh.node_ids if abs(mesh.node_coordinates(n)[2]) < 1e-12]
    z1, z2 = 0.5 * L, 0.75 * L
    bcs = BoundaryConditions().clamp(root)
    tip = [n for n in mesh.node_ids if abs(mesh.node_coordinates(n)[2] - L) < 1e-12]
    ref = add_reference_node(mesh, (0.0, 0.0, L))
    rigid_link(mesh, bcs, ref, tip, dofs=(0, 1))  # a rib: rigid in its plane, free to warp and stretch
    bcs.fix([ref], [2, 3, 4])
    case = LoadCase()
    case.add_nodal_load(ref, 5, 100.0)
    sol = StructuralGlobalSolver().solve(mesh, bcs, case)

    def shell_twist(z):
        nodes = [n for n in mesh.node_ids if abs(mesh.node_coordinates(n)[2] - z) < 1e-12 and n != ref]
        P = np.array([mesh.node_coordinates(n)[:2] for n in nodes])
        d = np.array([sol.node_displacement(n)[:2] for n in nodes])
        r, ut = P - P.mean(0), d - d.mean(0)
        return np.sum(r[:, 0] * ut[:, 1] - r[:, 1] * ut[:, 0]) / np.sum(r * r)

    # twist rate in the middle of the span, away from the root's and the rib's boundary layers
    rate = (shell_twist(z2) - shell_twist(z1)) / (z2 - z1)
    assert rate == pytest.approx(100.0 / sec.GJ_free, rel=0.05)
    assert abs(rate - 100.0 / sec.GJ) > 0.3 * rate  # the held-strain GJ is far too stiff
    beam = RestrainedTorsionBeam.from_section(sec, L)
    r = beam.cantilever_tip_torque(100.0, z=[z1, z2])
    assert (r.twist[1] - r.twist[0]) / (z2 - z1) == pytest.approx(rate, rel=0.05)


def test_gbt_classical_modes_use_the_coupled_warping(library):
    s = box(_carbon(library, [30, 30, 30, 30]))
    g = GeneralizedBeam(s, 3.0, 10, "classical", 0)
    assert np.allclose(g.modes.warping[:, 3], s.omega)
