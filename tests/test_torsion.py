"""Thin-walled torsion: Saint-Venant section properties, restrained warping and GBT distortion."""

import numpy as np
import pytest

from laminate_modelling_package import (
    BoundaryConditions,
    FEAMesh,
    FSDTModel,
    GeneralizedBeam,
    Laminate,
    LoadCase,
    RestrainedTorsionBeam,
    SectionModes,
    StructuralGlobalSolver,
    ThinWalledSection,
    add_reference_node,
    create_default_library,
    rigid_link,
)

E, NU, T = 68.9e9, 0.33, 0.004
G = E / (2 * (1 + NU))


@pytest.fixture(scope="module")
def aluminium():
    return Laminate.from_angles(create_default_library().get("Al-6061-T6"), [0], T)


def box(lam, b=0.6, h=0.3, nb=6, nh=3):
    corners = [(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, h / 2), (-b / 2, h / 2)]
    segs = []
    for k in range(4):
        p, q = np.array(corners[k]), np.array(corners[(k + 1) % 4])
        m = nb if k % 2 == 0 else nh
        segs += [(p + (q - p) * s / m, p + (q - p) * (s + 1) / m, lam, "flange" if k % 2 == 0 else "web")
                 for s in range(m)]
    return ThinWalledSection.from_segments(segs)


def test_box_closed_form(aluminium):
    b, h = 0.6, 0.3
    s = box(aluminium, b, h)
    assert s.n_cells == 1
    assert s.J_membrane == pytest.approx(4 * (b * h) ** 2 * G * T / (2 * (b + h)), rel=1e-10)
    assert s.GJ_walls == pytest.approx(G * T**3 / 3 * 2 * (b + h), rel=1e-10)
    assert np.allclose(s.shear_centre, 0.0, atol=1e-12)
    assert np.max(np.abs(s.omega)) == pytest.approx(b * h / 4 * (b - h) / (b + h), rel=1e-10)
    assert s.EGamma_primary == pytest.approx(E * T * b**2 * h**2 * (b - h) ** 2 / (24 * (b + h)), rel=1e-10)
    assert s.Ip == pytest.approx(G * T * (2 * b * (h / 2) ** 2 + 2 * h * (b / 2) ** 2), rel=1e-10)
    assert s.EI_xx == pytest.approx(E * T * (2 * b * (h / 2) ** 2 + 2 * h**3 / 12), rel=1e-10)


def test_open_sections(aluminium):
    bf, hw = 0.3, 0.5
    i_beam = ThinWalledSection.from_segments([
        ((-bf / 2, hw / 2), (0, hw / 2), aluminium), ((0, hw / 2), (bf / 2, hw / 2), aluminium),
        ((-bf / 2, -hw / 2), (0, -hw / 2), aluminium), ((0, -hw / 2), (bf / 2, -hw / 2), aluminium),
        ((0, -hw / 2), (0, 0), aluminium), ((0, 0), (0, hw / 2), aluminium)])
    assert i_beam.n_cells == 0
    assert i_beam.GJ == pytest.approx(G * T**3 / 3 * (2 * bf + hw), rel=1e-10)
    assert i_beam.EGamma_primary == pytest.approx(E * T * bf**3 * hw**2 / 24, rel=1e-10)
    assert i_beam.benscoter_mu == pytest.approx(1.0)
    channel = ThinWalledSection.from_segments([
        ((bf, hw / 2), (0, hw / 2), aluminium), ((0, hw / 2), (0, -hw / 2), aluminium),
        ((0, -hw / 2), (bf, -hw / 2), aluminium)])
    assert channel.shear_centre[0] == pytest.approx(-3 * bf**2 / (hw + 6 * bf), rel=1e-10)
    assert channel.EGamma_primary == pytest.approx(
        E * T * bf**3 * hw**2 / 12 * (3 * bf + 2 * hw) / (6 * bf + hw), rel=1e-10)


def test_multicell_shear_flows_balance(aluminium):
    # a two-cell box: flows balance at the junctions and the load they give is a pure torque
    pts = [(-0.4, -0.1), (0.0, -0.1), (0.4, -0.1), (0.4, 0.1), (0.0, 0.1), (-0.4, 0.1)]
    walls = [(0, 1, aluminium), (1, 2, aluminium), (2, 3, aluminium), (3, 4, aluminium), (4, 5, aluminium),
             (5, 0, aluminium), (1, 4, aluminium)]
    s = ThinWalledSection(pts, walls)
    assert s.n_cells == 2
    q, b = s.shear_flow, s.length
    assert np.allclose((q[:, None] * b[:, None] * s.tangent).sum(0), 0.0, atol=1e-6 * np.abs(q).max())
    # symmetric cells: no flow in the central web
    assert abs(q[6]) < 1e-9 * np.abs(q).max()


def test_restrained_torsion_closed_form_and_fe(aluminium):
    s = box(aluminium)
    for S in (s.S_warping, None):
        beam = RestrainedTorsionBeam(s.GJ, s.EGamma, 3.0, S)
        cf = beam.cantilever_tip_torque(1000.0)
        fe = beam.solve(tip_torque=1000.0)
        assert fe.twist[-1] == pytest.approx(cf.twist[-1], rel=1e-5)
        assert fe.bimoment[0] == pytest.approx(cf.bimoment[0], rel=1e-3)
        assert np.allclose(fe.torque, 1000.0, rtol=1e-4)
        cf_r = beam.cantilever_tip_torque(1000.0, tip_warping="restrained")
        fe_r = beam.solve(tip_torque=1000.0, supports={0.0: (True, True), 3.0: (False, True)})
        assert fe_r.twist[-1] == pytest.approx(cf_r.twist[-1], rel=1e-5)
    # restrained warping always stiffens; Vlasov more than Benscoter
    sv = 1000.0 * 3.0 / s.GJ
    ben = RestrainedTorsionBeam.from_section(s, 3.0).cantilever_tip_torque(1000.0).twist[-1]
    vla = RestrainedTorsionBeam.from_section(s, 3.0, vlasov=True).cantilever_tip_torque(1000.0).twist[-1]
    assert vla < ben < sv


def test_distortional_modes_are_work_free(aluminium):
    s = box(aluminium)
    m = SectionModes(s, "classical", None)
    N = len(s.points)
    # tangential displacement of each wall for each in-plane mode
    v = np.array([[0.5 * (m.in_plane[3 * i:3 * i + 2, c] + m.in_plane[3 * j:3 * j + 2, c]) @ s.tangent[k]
                   for k, (i, j) in enumerate(zip(s.start, s.end))] for c in range(m.n_in_plane)])
    work = v @ (s.shear_flow * s.length)
    assert abs(work[2]) > 0.0  # the rotation does work with the Saint-Venant flows
    assert np.allclose(work[3:], 0.0, atol=1e-8 * abs(work[2]))
    assert m.n_in_plane == 3 + (3 * N - len(s.walls) - 3)
    assert np.all(np.diff(m.distortional_stiffness) >= -1e-9 * m.distortional_stiffness.max())


def test_gbt_rigid_section_reproduces_benscoter(aluminium):
    s = box(aluminium)
    ben = RestrainedTorsionBeam.from_section(s, 3.0).cantilever_tip_torque(1000.0, z=[0.0, 3.0])
    sol = GeneralizedBeam(s, 3.0, 30, "classical", 0).clamp(0.0).add_torque(3.0, 1000.0).solve()
    assert sol.twist([3.0])[0] == pytest.approx(ben.twist[-1], rel=2e-3)
    assert sol.bimoment([0.0])[0] == pytest.approx(ben.bimoment[0], rel=2e-2)


def test_gbt_nodes_and_stations(aluminium):
    s = box(aluminium)
    z = GeneralizedBeam.graded_nodes(3.0, 20, [1.0, 2.0])
    assert {1.0, 2.0} <= set(np.round(z, 12)) and z[0] == 0.0 and z[-1] == 3.0
    beam = GeneralizedBeam(s, 3.0, 20, "classical", 2)
    with pytest.raises(ValueError):
        beam.diaphragm(1.2345)


def _shell_box(lam, b, h, L, nb, nh, nz):
    """Quad8 box on the wall mid-surfaces along z (counter-clockwise, outward normals)."""
    corners = [(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, h / 2), (-b / 2, h / 2)]
    ring, names = [], []
    for k in range(4):
        p, q = np.array(corners[k]), np.array(corners[(k + 1) % 4])
        m = 2 * (nb if k % 2 == 0 else nh)
        ring += [p + (q - p) * i / m for i in range(m)]
        names += ["flange" if k % 2 == 0 else ("web_r" if k == 1 else "web_l")] * m
    zs = np.linspace(0.0, L, 2 * nz + 1)
    mesh, nid, n = FEAMesh(), {}, 1
    for j, z in enumerate(zs):
        for i, p in enumerate(ring):
            if i % 2 and j % 2:
                continue
            mesh.add_node(n, (p[0], p[1], z))
            nid[(i, j)] = n
            n += 1
    nr, eid = len(ring), 1
    sets = {"flange": [], "web_r": [], "web_l": []}
    for jj in range(nz):
        j = 2 * jj
        for ii in range(0, nr, 2):
            i2 = (ii + 2) % nr
            conn = [nid[(ii, j)], nid[(i2, j)], nid[(i2, j + 2)], nid[(ii, j + 2)],
                    nid[(ii + 1, j)], nid[(i2, j + 1)], nid[(ii + 1, j + 2)], nid[(ii, j + 1)]]
            mesh.add_element(eid, conn, lam, FSDTModel(), element_type="quad8", material_axis=(0, 0, 1))
            sets[names[ii]].append(eid)
            eid += 1
    return mesh, sets


@pytest.mark.parametrize("load", ["rib", "couple"])
def test_gbt_against_shell_box(aluminium, load):
    b, h, L = 0.4, 0.2, 1.6
    mesh, sets = _shell_box(aluminium, b, h, L, 4, 2, 12)
    from types import SimpleNamespace

    blade = SimpleNamespace(mesh=mesh, element_sets=sets, span_positions=np.linspace(0, L, 13))
    sec = ThinWalledSection.from_blade_mesh(blade, z=L)
    root = [n for n in mesh.node_ids if abs(mesh.node_coordinates(n)[2]) < 1e-12]
    tip = [n for n in mesh.node_ids if abs(mesh.node_coordinates(n)[2] - L) < 1e-12]
    bcs = BoundaryConditions().clamp(root)
    case = LoadCase()
    beam = GeneralizedBeam(sec, L, 16, "full", None).clamp(0.0)
    if load == "rib":
        ref = add_reference_node(mesh, (0.0, 0.0, L))
        rigid_link(mesh, bcs, ref, tip, dofs=(0, 1))
        bcs.fix([ref], [2, 3, 4])
        case.add_nodal_load(ref, 5, 100.0)
        beam.diaphragm(L).add_torque(L, 100.0)
    else:
        F = 100.0 / b
        for sign in (1.0, -1.0):
            nodes = [n for n in tip if abs(mesh.node_coordinates(n)[0] - sign * b / 2) < 1e-9]
            case.add_line_load(nodes, (0.0, sign * F / h, 0.0))
        beam.add_wall_load(L, beam.walls_named("web_r"), (0.0, F))
        beam.add_wall_load(L, beam.walls_named("web_l"), (0.0, -F))
    sol = StructuralGlobalSolver().solve(mesh, bcs, case)
    P = sec.points
    d = np.array([sol.node_displacement(m)[:2] for m in sec.mesh_nodes])
    r = P - P.mean(0)
    ut = d - d.mean(0)
    shell_twist = np.sum(r[:, 0] * ut[:, 1] - r[:, 1] * ut[:, 0]) / np.sum(r * r)
    gbt_twist = beam.solve().mean_twist([L])[0]
    rigid = GeneralizedBeam(sec, L, 16, "classical", 0).clamp(0.0).add_torque(L, 100.0).solve().mean_twist([L])[0]
    assert gbt_twist == pytest.approx(shell_twist, rel=0.02)
    if load == "couple":
        assert shell_twist > 3.0 * rigid  # the couple mostly distorts the box


def test_extensible_walls(aluminium):
    s = box(aluminium)
    m = SectionModes(s, "classical", None, extensible=True)
    assert m.n_in_plane == 3 * len(s.points)
    stiff = GeneralizedBeam(s, 3.0, 20, "classical", None).clamp(0.0).add_torque(3.0, 1e3).solve().twist([3.0])[0]
    soft = GeneralizedBeam(s, 3.0, 20, "classical", None, extensible=True).clamp(0.0).add_torque(3.0, 1e3).solve()
    assert soft.twist([3.0])[0] == pytest.approx(stiff, rel=0.02)
    assert soft.wall_resultants(1.5)[0].shape[1] == 6
