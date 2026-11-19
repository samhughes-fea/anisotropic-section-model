"""Section strength checks: ply stresses and failure exposures of one section under its forces."""

from types import SimpleNamespace

import numpy as np
import pytest

from anisotropic_section_model import ThinWalledSection, TimoshenkoBeam
from anisotropic_section_model.ply_failure import PlyStrength, exposures
from anisotropic_section_model.section_strength import SectionStrength, to_analysis_forces
from anisotropic_section_model.solid.mesh import SectionMesh
from anisotropic_section_model.solid.section_analysis import SolidSectionAnalysis
from shell_fem import (BoundaryConditions, ExactStressRecoverer, Laminate, LoadCase, StructuralGlobalSolver,
    add_reference_node, create_default_library, rigid_link)

from test_torsion import _shell_box, box

B, H, L = 0.4, 0.2, 2.0


@pytest.fixture(scope="module")
def lib():
    return create_default_library()


@pytest.fixture(scope="module")
def aluminium_box(lib):
    return box(Laminate.from_angles(lib.get("Al-6061-T6"), [0], 0.004), B, H)


@pytest.fixture(scope="module")
def composite(lib):
    return Laminate.from_angles(lib.get("T300/5208"), [0, 45, -45, 90, 90, -45, 45, 0], 0.25e-3)


@pytest.fixture(scope="module")
def composite_box(composite):
    return box(composite, B, H)


ALU = PlyStrength(Xt=290e6, Xc=290e6, Yt=290e6, Yc=290e6, S12=190e6)


def test_bending_stress_is_my_over_i(aluminium_box):
    s = aluminium_box
    check = SectionStrength.from_thin_walled(s, ALU)
    M = 2.0e3
    r = check.evaluate([0, 0, 0, M, 0, 0], reference=s.centroid)
    y = check.position[:, 1] - s.centroid[1]
    mid = np.abs(check.n) < 1e-12
    E = 68.9e9
    # uniaxial wall stress at the mid-surface: sigma = E kappa y (walls free in the hoop direction)
    assert np.allclose(r.stress[mid, 0], E * M / s.EI_xx * y[mid], rtol=1e-2, atol=1e-3 * np.abs(r.stress[:, 0]).max())
    # the maximum-stress exposure is sigma / X at the extreme fibre
    c = r.critical("max_stress")
    assert c.exposure == pytest.approx(np.abs(r.stress[:, 0]).max() / ALU.Xt, rel=1e-9)
    assert c.mode in ("s11t", "s11c")


def test_forces_about_a_reference_point(composite_box):
    s = composite_box
    check = SectionStrength.from_thin_walled(s, points_per_wall=3)
    F = np.array([1.0e3, -2.0e3, 5.0e3, 4.0e2, -3.0e2, 1.0e2])
    ref = (0.05, -0.02)
    about_origin = to_analysis_forces(F, ref)
    a = check.stresses(F, reference=ref)
    b = check.stresses(about_origin, order="analysis")
    assert np.allclose(a, b)
    # the same forces as a beam would carry them: a force at ref is its resultant plus the moment
    beam = TimoshenkoBeam(s.timoshenko_stiffness(), 1.0, 1).clamp(0.0).add_force(1.0, F[:3], point=ref)
    f0 = beam.solve().section_forces(1.0)[0]
    Fr = np.concatenate([F[:3], np.zeros(3)])
    assert np.allclose(check.stresses(f0), check.stresses(Fr, reference=ref), atol=1e-6 * np.abs(a).max())


def test_exposures_scale_linearly_and_match_ply_failure(composite_box):
    check = SectionStrength.from_thin_walled(composite_box, points_per_wall=3)
    F = np.array([2.0e3, 5.0e3, -1.0e4, 3.0e3, -1.0e3, 8.0e2])
    r1, r2 = check.evaluate(F), check.evaluate(2.5 * F)
    for c in ("max_stress", "tsai_wu", "hashin", "puck", "exposure"):
        assert np.allclose(r2.exposure(c), 2.5 * r1.exposure(c), rtol=1e-6)
    st = check._strengths[0]
    assert np.allclose(exposures(r1.stress, st)["tsai_wu"], r1.exposure("tsai_wu"))
    c = r1.critical()
    assert c.exposure == pytest.approx(r1.exposure().max())
    assert c.criterion in ("max_stress", "tsai_wu", "hashin", "puck")
    assert c.reserve_factor == pytest.approx(1.0 / c.exposure)
    assert set(r1.by_name()) == {"flange", "web"}


def test_missing_strengths_are_left_out(composite_box, lib):
    carbon = lib.get("T300/5208")
    only_carbon = SectionStrength.from_thin_walled(composite_box, {carbon: PlyStrength(1e9, 1e9, 5e7, 2e8, 7e7)})
    assert only_carbon.n_points > 0
    with pytest.raises(ValueError):
        SectionStrength.from_thin_walled(composite_box, {})


def test_solid_recovery_agrees_with_thin_walled(composite_box):
    s = composite_box
    a = s.analysis
    mesh = SectionMesh.from_thin_walled(a.section, a.laminates, elements_along=4)
    solid = SectionStrength.from_solid(SolidSectionAnalysis(mesh), SectionStrength.ply_strengths(s))
    thin = SectionStrength.from_thin_walled(s, points_per_wall=5)

    def away_from_corners(c):  # the solid strips overlap at the corners, which the mid-line model does not see
        x, y = c.position[:, 0], c.position[:, 1]
        return (np.abs(x) < B / 2 - 0.04) | (np.abs(y) < H / 2 - 0.04)

    for F in ([0, 0, 0, 2.0e3, 0, 0], [0, 0, 0, 0, 0, 1.0e3], [0, 0, 2.0e5, 0, 0, 0], [1.0e4, 0, 0, 0, 0, 0]):
        e_thin = thin.evaluate(F).critical(where=away_from_corners(thin))
        e_solid = solid.evaluate(F).critical(where=away_from_corners(solid))
        assert e_solid.exposure == pytest.approx(e_thin.exposure, rel=0.01)
        assert e_solid.criterion == e_thin.criterion


def test_ply_stresses_against_a_shell_box(composite):
    """Ply stresses at the middle of a shell box under a tip force and torque, both ways."""
    Ls = 1.6
    mesh, sets = _shell_box(composite, B, H, Ls, 4, 2, 8)
    blade = SimpleNamespace(mesh=mesh, element_sets=sets, span_positions=np.linspace(0, Ls, 9))
    sec = ThinWalledSection.from_blade_mesh(blade, z=Ls / 2)
    root = [n for n in mesh.node_ids if abs(mesh.node_coordinates(n)[2]) < 1e-12]
    tip = [n for n in mesh.node_ids if abs(mesh.node_coordinates(n)[2] - Ls) < 1e-12]
    bcs = BoundaryConditions().clamp(root)
    ref = add_reference_node(mesh, (0.0, 0.0, Ls))
    rigid_link(mesh, bcs, ref, tip, dofs=(0, 1))
    bcs.fix([ref], [2, 3, 4])
    load = np.array([1.0e3, 3.0e3, 0.0, 0.0, 0.0, 4.0e2])
    case = LoadCase()
    for d in (0, 1, 5):
        case.add_nodal_load(ref, d, load[d])
    sol = StructuralGlobalSolver().solve(mesh, bcs, case)
    forces = TimoshenkoBeam(sec.timoshenko_stiffness(), Ls, 1).clamp(0.0).add_force(Ls, load).solve()
    check = SectionStrength.from_thin_walled(sec, points_per_wall=3)
    rec = ExactStressRecoverer()
    shell, ours = [], []
    for e in mesh.shell_elements:
        X = np.array([mesh.node_coordinates(n) for n in e.node_ids])
        if not (X[:, 2].min() < 0.75 < X[:, 2].max()):
            continue
        u = sol.element_displacements(e.element_id)
        xyz = e.global_coordinates(0.0, 0.0)
        d = np.linalg.norm(check.position - xyz[:2], axis=1)
        s_ours = check.stresses(forces.section_forces(xyz[2])[0])
        for k, (zb, zt) in enumerate(e.laminate.ply_bounds()):
            j = np.where((d < 1e-9) & (check.ply == k) & np.isclose(check.n, 0.5 * (zb + zt)))[0]
            if j.size:
                shell.append(rec.recover(e, u, 0.5 * (zb + zt), 0.0, 0.0).material_stress_3d())
                ours.append(s_ours[j[0]])
    shell, ours = np.array(shell), np.array(ours)
    assert len(shell) > 50
    for c in (0, 1, 5):  # sigma_11, sigma_22, tau_12
        scale = np.abs(shell[:, c]).max()
        assert np.sqrt(np.mean((shell[:, c] - ours[:, c]) ** 2)) < 0.05 * scale
