"""Section, wall, Gauss and ply formulation fields of ThinWalledSectionAnalysis."""

import numpy as np
import pytest

from anisotropic_section_model import (
    FORMULATION_CSV_SCHEMA,
    FormulationWriter,
)
from anisotropic_section_model.thin_walled import (
    Ply,
    PlyMaterial,
    ThinWalledSection,
    ThinWalledSectionAnalysis,
    Wall,
    WallLaminate,
)
from anisotropic_section_model.thin_walled.stress_recovery import SaintVenantRecovery

E, NU = 70e9, 0.3
CFRP = PlyMaterial(E1=140e9, E2=10e9, G12=5e9, nu12=0.3, G13=5e9, G23=3.5e9, rho=1600.0)


def iso(t, rho=0.0):
    return WallLaminate.isotropic(E, NU, t, rho=rho)


def i_beam(b, h, web_lam=1):
    pts = {1: (-b / 2, h / 2), 2: (0, h / 2), 3: (b / 2, h / 2),
           4: (-b / 2, -h / 2), 5: (0, -h / 2), 6: (b / 2, -h / 2)}
    walls = [Wall(1, 1, 2, 1), Wall(2, 2, 3, 1), Wall(3, 4, 5, 1), Wall(4, 5, 6, 1), Wall(5, 5, 2, web_lam)]
    return ThinWalledSection(pts, walls)


def box(b, h, flange_lam=1, web_lam=2):
    pts = {1: (-b / 2, -h / 2), 2: (b / 2, -h / 2), 3: (b / 2, h / 2), 4: (-b / 2, h / 2)}
    walls = [Wall(1, 1, 2, flange_lam), Wall(2, 2, 3, web_lam), Wall(3, 3, 4, flange_lam), Wall(4, 4, 1, web_lam)]
    return ThinWalledSection(pts, walls)


def test_gauss_warping_and_wall_strains_match_analysis():
    sa = ThinWalledSectionAnalysis(i_beam(0.2, 0.3), {1: iso(1e-3)}, hoop="free")
    F = np.array([1e3, 2e2, -1e2, 0.0, 0.0, 50.0])
    writer = FormulationWriter(sa, forces=F)
    gp = writer.gauss()
    expected = sa.wall_strains(F)
    assert gp.wall_strains.shape == (len(expected), 4)
    for i, (wall_id, position, e) in enumerate(expected):
        assert gp.wall_ids[i] == wall_id
        assert gp.position[i] == pytest.approx(position)
        assert gp.wall_strains[i] == pytest.approx(e)
        assert gp.omega[i] == pytest.approx(sa.omega_at(wall_id, gp.s[i]), rel=1e-12, abs=1e-16)
    om = sa.point_warping()
    assert abs(om[1]) == pytest.approx(0.2 * 0.3 / 4, rel=1e-10)


def test_gauss_resultants_sum_to_section_forces():
    sa = ThinWalledSectionAnalysis(i_beam(0.2, 0.3), {1: iso(1e-3)})
    F = np.array([1e3, 0.0, 0.0, 0.0, 0.0, 0.0])
    gp = FormulationWriter(sa, forces=F).gauss()
    total = np.zeros(6)
    for i, g in enumerate(sa._gauss_points):
        total += g.weight * g.Z.T @ (gp.K[i] @ gp.wall_strains[i])
    np.testing.assert_allclose(total, F, atol=1e-8 * np.abs(F).max())
    assert gp.Nxx == pytest.approx(np.array([g.K @ e for g, e in zip(sa._gauss_points, gp.wall_strains)])[:, 0])


def test_box_torsion_shear_flow_is_bredt():
    b, h, t = 0.4, 0.2, 2e-3
    sa = ThinWalledSectionAnalysis(box(b, h), {1: iso(t), 2: iso(t)}, hoop="free")
    T = 80.0
    gp = FormulationWriter(sa, forces=[0, 0, 0, 0, 0, T]).gauss()
    q = T / (2.0 * b * h)
    assert gp.Nxs == pytest.approx(q, rel=2e-3)
    assert np.max(np.abs(gp.Qx)) < 1e-6 * abs(q)


def test_section_and_wall_levels():
    sa = ThinWalledSectionAnalysis(box(0.4, 0.2), {1: iso(3e-3), 2: iso(1e-3)})
    writer = FormulationWriter(sa, forces=[1e5, 0, 0, 0, 0, 0])
    sec = writer.section()
    walls = writer.walls()
    np.testing.assert_allclose(sec.stiffness, sa.stiffness)
    np.testing.assert_allclose(sec.classical_stiffness, sa.classical_stiffness)
    assert sec.warping_constant == pytest.approx(sa.warping_constant)
    assert tuple(sec.shear_centre) == pytest.approx(sa.shear_centre)
    assert sec.n_walls == 4
    assert sec.n_gauss == len(sa._gauss_points)
    assert walls.wall_ids.tolist() == [w.wall_id for w in sa.section.walls]
    assert walls.n_gauss.sum() == sec.n_gauss
    assert walls.K.shape == (4, 4, 4)


def test_ply_matches_saint_venant_recovery():
    lam = WallLaminate([Ply(CFRP, 0.0, 1e-3), Ply(CFRP, 90.0, 1e-3)])
    sa = ThinWalledSectionAnalysis(box(0.3, 0.15), {1: lam, 2: lam})
    F = np.array([2e4, 10.0, -4.0, 0.0, 0.0, 3.0])
    writer = FormulationWriter(sa, forces=F, points_per_ply=3)
    ply = writer.ply()
    rec = SaintVenantRecovery(sa)
    gp = writer.gauss()
    assert ply.ply.size == gp.wall_ids.size * 2 * 3
    sample = rec.at(int(gp.wall_ids[0]), float(gp.s[0]), F, points_per_ply=3)
    mask = (ply.wall_ids == gp.wall_ids[0]) & np.isclose(ply.s, gp.s[0])
    recovered = ply.stress_wall[mask]
    expected = np.array([p.stress_wall for p in sample.plies])
    np.testing.assert_allclose(recovered, expected)


def test_csv_export(tmp_path):
    sa = ThinWalledSectionAnalysis(box(0.4, 0.2), {1: iso(3e-3), 2: iso(1e-3)})
    paths = FormulationWriter(sa, forces=[0, 0, 0, 0, 0, 1.0]).to_csv(tmp_path)
    assert set(paths) == {"section", "walls", "gauss", "ply"}
    schema = f"# schema={FORMULATION_CSV_SCHEMA}"
    for path in paths.values():
        assert path.read_text().splitlines()[0] == schema
    gauss_header = paths["gauss"].read_text().splitlines()[1]
    assert "omega" in gauss_header
    assert "Nxs" in gauss_header
    assert "wall_strains_0" in gauss_header
    n_gauss = len(sa._gauss_points)
    assert len(paths["gauss"].read_text().splitlines()) == n_gauss + 2
    totals = dict(line.split(",", 1) for line in paths["section"].read_text().splitlines()[2:])
    assert int(totals["n_gauss"]) == n_gauss
    assert float(totals["warping_constant"]) == pytest.approx(sa.warping_constant)


def test_forces_must_be_six():
    sa = ThinWalledSectionAnalysis(box(0.4, 0.2), {1: iso(1e-3), 2: iso(1e-3)})
    with pytest.raises(ValueError, match="shape"):
        FormulationWriter(sa, forces=[1.0, 0.0])


def test_from_section_takes_public_forces():
    from types import SimpleNamespace

    from anisotropic_section_model.axes import to_engine

    an = ThinWalledSectionAnalysis(box(0.2, 0.1), {1: iso(2e-3), 2: iso(3e-3)})
    public = np.array([10.0, -20.0, 3e3, 40.0, -50.0, 60.0])  # [Fx, Fy, N, Mx, My, Mz]
    w = FormulationWriter.from_section(SimpleNamespace(analysis=an), public)
    np.testing.assert_array_equal(w.forces, to_engine(public))
    np.testing.assert_array_equal(w.forces, [3e3, 40.0, -50.0, 10.0, -20.0, 60.0])  # [N, M_y, M_z, V_y, V_z, T]
    np.testing.assert_allclose(w.gauss().wall_strains, FormulationWriter(an, to_engine(public)).gauss().wall_strains)
