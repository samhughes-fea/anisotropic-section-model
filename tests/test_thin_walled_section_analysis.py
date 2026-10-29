"""Saint-Venant analysis of thin-walled sections against closed-form thin-wall results."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from pre_processing.section_integration.thin_walled import (
    Ply,
    PlyMaterial,
    ThinWalledSection,
    Wall,
    WallLaminate,
)
from pre_processing.section_integration.thin_walled.section_analysis import (
    ThinWalledSectionAnalysis,
    stiffness_about,
    translation_matrix,
    wall_law,
)

E, NU = 70e9, 0.3
G = E / (2 * (1 + NU))
CFRP = PlyMaterial(E1=140e9, E2=10e9, G12=5e9, nu12=0.3, G13=5e9, G23=3.5e9, rho=1600.0)
UNSYM = WallLaminate([Ply(CFRP, 30.0, 1e-3), Ply(CFRP, -10.0, 2e-3), Ply(CFRP, 75.0, 1e-3)])


def iso(t, rho=0.0):
    return WallLaminate.isotropic(E, NU, t, rho=rho)


def polygon_tube(R, n):
    pts = {i + 1: (R * np.cos(2 * np.pi * i / n), R * np.sin(2 * np.pi * i / n)) for i in range(n)}
    walls = [Wall(i + 1, i + 1, (i + 1) % n + 1, 1) for i in range(n)]
    return ThinWalledSection(pts, walls)


def i_beam(b, h, web_lam=1):
    pts = {1: (-b / 2, h / 2), 2: (0, h / 2), 3: (b / 2, h / 2),
           4: (-b / 2, -h / 2), 5: (0, -h / 2), 6: (b / 2, -h / 2)}
    walls = [Wall(1, 1, 2, 1), Wall(2, 2, 3, 1), Wall(3, 4, 5, 1), Wall(4, 5, 6, 1), Wall(5, 5, 2, web_lam)]
    return ThinWalledSection(pts, walls)


def box(b, h, flange_lam=1, web_lam=2):
    pts = {1: (-b / 2, -h / 2), 2: (b / 2, -h / 2), 3: (b / 2, h / 2), 4: (-b / 2, h / 2)}
    walls = [Wall(1, 1, 2, flange_lam), Wall(2, 2, 3, web_lam), Wall(3, 3, 4, flange_lam), Wall(4, 4, 1, web_lam)]
    return ThinWalledSection(pts, walls)


def normalised(K):
    d = np.sqrt(np.outer(np.diag(K), np.diag(K)))
    return K / d


# --------------------------------------------------------------------------- #
# Isotropic closed forms (hoop "free" gives the textbook E t^3 / 12 wall bending)
# --------------------------------------------------------------------------- #


def test_polygon_tube_isotropic():
    R, t, n = 0.5, 5e-3, 64
    sec = polygon_tube(R, n)
    sa = ThinWalledSectionAnalysis(sec, {1: iso(t)}, hoop="free")
    K = sa.stiffness
    P = 2 * n * R * np.sin(np.pi / n)
    A_enc = 0.5 * n * R * R * np.sin(2 * np.pi / n)
    EI = 0.0
    for w in sec.walls:
        a, b = sec.endpoints(w)
        L = sec.length(w)
        EI += E * t * L * (a[1] ** 2 + a[1] * b[1] + b[1] ** 2) / 3 + E * t**3 / 12 * sec.normal(w)[1] ** 2 * L
    assert K[0, 0] == pytest.approx(E * P * t, rel=1e-12)
    assert K[1, 1] == pytest.approx(EI, rel=1e-10)
    assert K[2, 2] == pytest.approx(EI, rel=1e-10)
    assert K[5, 5] == pytest.approx(4 * A_enc**2 * G * t / P + G * P * t**3 / 3, rel=1e-10)
    # thin circular tube: shear stiffness G pi R t (shear coefficient 1/2)
    assert K[3, 3] == pytest.approx(G * np.pi * R * t, rel=2e-3)
    assert K[4, 4] == pytest.approx(K[3, 3], rel=1e-10)
    assert np.max(np.abs(normalised(K) - np.eye(6))) < 1e-10
    assert np.allclose(sa.shear_centre, 0.0, atol=1e-12)
    assert np.allclose(sa.tension_centre, 0.0, atol=1e-12)


def test_box_bredt_with_two_thicknesses():
    b, h, tf, tw = 0.4, 0.2, 3e-3, 1e-3
    sa = ThinWalledSectionAnalysis(box(b, h), {1: iso(tf), 2: iso(tw)}, hoop="free")
    K4 = sa.classical_stiffness
    A_enc = b * h
    GJ = 4 * A_enc**2 * G / (2 * b / tf + 2 * h / tw) + G * (2 * b * tf**3 + 2 * h * tw**3) / 3
    assert K4[3, 3] == pytest.approx(GJ, rel=1e-10)
    EI_y = E * (2 * b * tf * (h / 2) ** 2 + 2 * tw * h**3 / 12 + 2 * b * tf**3 / 12)
    assert K4[1, 1] == pytest.approx(EI_y, rel=1e-10)


def test_i_beam_torsion_warping_and_shear():
    b, h, tf, tw = 0.2, 0.3, 1e-3, 1.5e-3
    sa = ThinWalledSectionAnalysis(i_beam(b, h, web_lam=2), {1: iso(tf), 2: iso(tw)}, hoop="free")
    K = sa.stiffness
    assert sa.classical_stiffness[3, 3] == pytest.approx(G * (2 * b * tf**3 + h * tw**3) / 3, rel=1e-10)
    Gamma = tf * b**3 * h**2 / 24 + 2 * tf**3 * b**3 / 144 + tw**3 * h**3 / 144
    assert sa.warping_constant == pytest.approx(E * Gamma, rel=1e-10)
    assert np.allclose(sa.shear_centre, 0.0, atol=1e-12)
    # shear stiffness for V_z from the complementary energy of the classical shear flow
    Iy = 2 * b * tf * (h / 2) ** 2 + tw * h**3 / 12
    x, w = leggauss(20)
    s, ws = (x + 1) / 2 * (b / 2), w * b / 4
    U = 4 * np.sum(ws * (tf * h / 2 * s / Iy) ** 2 / (G * tf))
    z, wz = x * h / 2, w * h / 2
    U += np.sum(wz * ((tf * b * h / 2 + tw * (h * h / 4 - z * z) / 2) / Iy) ** 2 / (G * tw))
    assert K[4, 4] == pytest.approx(1.0 / U, rel=1e-4)
    # flange tips warp by -/+ b h / 4 about the shear centre
    om = sa.point_warping()
    assert abs(om[1]) == pytest.approx(b * h / 4, rel=1e-10)
    assert om[1] == pytest.approx(-om[3], rel=1e-10)
    assert om[2] == pytest.approx(0.0, abs=1e-14)


def test_channel_shear_centre():
    b, h, t = 0.2, 0.3, 1e-4
    pts = {1: (b, h / 2), 2: (0, h / 2), 3: (0, -h / 2), 4: (b, -h / 2)}
    sec = ThinWalledSection(pts, [Wall(1, 1, 2, 1), Wall(2, 2, 3, 1), Wall(3, 3, 4, 1)])
    sa = ThinWalledSectionAnalysis(sec, {1: iso(t)}, hoop="free")
    y_s, z_s = sa.shear_centre
    assert y_s == pytest.approx(-3 * b * b / (h + 6 * b), rel=1e-6)
    assert z_s == pytest.approx(0.0, abs=1e-12)
    assert sa.tension_centre[0] == pytest.approx(b * b / (h + 2 * b), rel=1e-10)


# --------------------------------------------------------------------------- #
# Anisotropic walls
# --------------------------------------------------------------------------- #


def test_cus_tube_extension_twist_coupling():
    """Circumferentially uniform off-axis walls: closed-form Bredt with A16 coupling."""
    R, n = 0.3, 96
    lam = WallLaminate([Ply(CFRP, 20.0, 0.25e-3), Ply(CFRP, 20.0, 0.25e-3)])
    sa = ThinWalledSectionAnalysis(polygon_tube(R, n), {1: lam})
    K4 = sa.classical_stiffness
    Kw = wall_law(lam)
    P = 2 * n * R * np.sin(np.pi / n)
    A_enc = 0.5 * n * R * R * np.sin(2 * np.pi / n)
    assert K4[0, 0] == pytest.approx(P * Kw[0, 0], rel=1e-10)
    assert K4[0, 3] == pytest.approx(2 * A_enc * Kw[0, 1], rel=1e-10)
    assert K4[3, 3] == pytest.approx(4 * A_enc**2 * Kw[1, 1] / P, rel=1e-5)
    assert abs(K4[0, 3]) > 0.5 * np.sqrt(K4[0, 0] * K4[3, 3])


def test_one_cubic_element_per_wall_is_exact():
    sec = i_beam(0.2, 0.3)
    ref = ThinWalledSectionAnalysis(sec, {1: UNSYM}, n_sub=4).stiffness
    one = ThinWalledSectionAnalysis(sec, {1: UNSYM}, n_sub=1).stiffness
    assert np.max(np.abs(normalised(one) - normalised(ref))) < 1e-8


def test_stiffness_symmetric_positive_definite():
    sa = ThinWalledSectionAnalysis(box(0.4, 0.2), {1: UNSYM, 2: iso(2e-3)})
    K = sa.stiffness
    np.testing.assert_allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > 0
    np.testing.assert_allclose(sa.flexibility @ K, np.eye(6), atol=1e-8)


def test_translation_consistency():
    sec = i_beam(0.2, 0.3)
    dy, dz = 0.37, -0.21
    moved = ThinWalledSection({k: (v[0] + dy, v[1] + dz) for k, v in sec.points.items()}, sec.walls)
    a = ThinWalledSectionAnalysis(sec, {1: UNSYM})
    b = ThinWalledSectionAnalysis(moved, {1: UNSYM})
    # the moved section about its origin = the original about the point (-dy, -dz)
    expected = stiffness_about(a.stiffness, -dy, -dz)
    assert np.max(np.abs(normalised(b.stiffness) - normalised(expected))) < 1e-7
    assert b.shear_centre[0] == pytest.approx(a.shear_centre[0] + dy, abs=1e-9)
    assert b.shear_centre[1] == pytest.approx(a.shear_centre[1] + dz, abs=1e-9)
    assert b.tension_centre[0] == pytest.approx(a.tension_centre[0] + dy, abs=1e-9)
    P = translation_matrix(dy, dz)
    back = stiffness_about(stiffness_about(a.stiffness, dy, dz), -dy, -dz)
    assert np.max(np.abs(normalised(back) - normalised(a.stiffness))) < 1e-10
    np.testing.assert_allclose(P.T @ stiffness_about(a.stiffness, dy, dz) @ P, a.stiffness, rtol=1e-9, atol=1e-3)


def test_rotation_consistency():
    sec = box(0.4, 0.2)
    lams = {1: UNSYM, 2: iso(2e-3)}
    al = np.deg2rad(27.0)
    Rm = np.array([[np.cos(al), -np.sin(al)], [np.sin(al), np.cos(al)]])
    rot = ThinWalledSection({k: tuple(Rm @ np.asarray(v)) for k, v in sec.points.items()}, sec.walls)
    Ka = ThinWalledSectionAnalysis(sec, lams).stiffness
    Kb = ThinWalledSectionAnalysis(rot, lams).stiffness
    # psi_b = T psi_a: (-kappa_z, kappa_y) and (gamma_xy, gamma_xz) rotate with the section
    T = np.eye(6)
    S = np.array([[0.0, -1.0], [1.0, 0.0]])  # (kappa_y, kappa_z) -> (-kappa_z, kappa_y)
    T[1:3, 1:3] = np.linalg.inv(S) @ Rm @ S
    T[3:5, 3:5] = Rm
    np.testing.assert_allclose(normalised(T.T @ Kb @ T), normalised(Ka), atol=1e-8)


def test_symmetric_section_and_layup_has_no_couplings():
    cross = WallLaminate([Ply(CFRP, a, 0.25e-3) for a in (0, 90, 90, 0)])
    K = ThinWalledSectionAnalysis(box(0.4, 0.2), {1: cross, 2: iso(2e-3)}).stiffness
    assert np.max(np.abs(normalised(K) - np.eye(6))) < 1e-10


def test_hoop_options():
    sym = WallLaminate([Ply(CFRP, a, 0.25e-3) for a in (30, -30, -30, 30)])
    sec = box(0.4, 0.2)
    a = ThinWalledSectionAnalysis(sec, {1: sym, 2: sym}, hoop="restrained").stiffness
    b = ThinWalledSectionAnalysis(sec, {1: sym, 2: sym}, hoop="free").stiffness
    # with B = 0 only the wall plate-bending terms (D12, D26) differ
    np.testing.assert_allclose(a[0, 0], b[0, 0], rtol=1e-12)
    np.testing.assert_allclose(a[5, 5], b[5, 5], rtol=1e-4)
    assert a[5, 5] >= b[5, 5]
    r = ThinWalledSectionAnalysis(sec, {1: UNSYM, 2: UNSYM}, hoop="restrained").stiffness
    f = ThinWalledSectionAnalysis(sec, {1: UNSYM, 2: UNSYM}, hoop="free").stiffness
    assert np.linalg.eigvalsh(r - f).min() > -1e-9 * np.linalg.eigvalsh(r).max()
    assert r[5, 5] > f[5, 5]
    with pytest.raises(ValueError):
        ThinWalledSectionAnalysis(sec, {1: UNSYM, 2: UNSYM}, hoop="clamped")


def test_wall_resultants_sum_to_section_forces():
    """For forces with no shear, int Z^T K e_w ds returns the applied forces."""
    sa = ThinWalledSectionAnalysis(i_beam(0.2, 0.3), {1: UNSYM})
    for F in (np.array([1e3, 0, 0, 0, 0, 0]), np.array([0, 2e2, -1e2, 0, 0, 50.0])):
        total = np.zeros(6)
        for g, (_, _, e) in zip(sa._gauss_points, sa.wall_strains(F)):
            total += g.weight * g.Z.T @ (g.K @ e)
        np.testing.assert_allclose(total, F, atol=1e-8 * np.abs(F).max())


def test_uniform_axial_strain_for_isotropic_section():
    sa = ThinWalledSectionAnalysis(box(0.4, 0.2), {1: iso(3e-3), 2: iso(1e-3)})
    N = 1e5
    EA = E * (2 * 0.4 * 3e-3 + 2 * 0.2 * 1e-3)
    for _, _, e in sa.wall_strains([N, 0, 0, 0, 0, 0]):
        assert e[0] == pytest.approx(N / EA, rel=1e-10)
        assert abs(e[1]) < 1e-14


def test_mass_properties():
    R, t, n, rho = 0.5, 5e-3, 64, 2700.0
    sa = ThinWalledSectionAnalysis(polygon_tube(R, n), {1: iso(t, rho=rho)})
    mp = sa.mass_properties
    P = 2 * n * R * np.sin(np.pi / n)
    assert mp["m"] == pytest.approx(rho * t * P)
    assert mp["y_m"] == pytest.approx(0.0, abs=1e-14)
    assert mp["I_yy"] == pytest.approx(mp["I_zz"], rel=1e-12)
    assert mp["I_p"] == pytest.approx(rho * t * P * R * R * np.cos(np.pi / n) ** 2, rel=2e-3)
    # an unsymmetric laminate moves the mass off the mid-surface
    lam = WallLaminate([Ply(PlyMaterial.isotropic(E, NU, 4000.0), 0.0, 1e-3), Ply(PlyMaterial.isotropic(E, NU, 0.0), 0.0, 1e-3)])
    flat = ThinWalledSection({1: (0.0, 0.0), 2: (1.0, 0.0)}, [Wall(1, 1, 2, 1)])
    mp = ThinWalledSectionAnalysis(flat, {1: lam}).mass_properties
    assert mp["z_m"] == pytest.approx(-0.5e-3)


def test_missing_laminate():
    with pytest.raises(ValueError, match="no laminate"):
        ThinWalledSectionAnalysis(box(0.4, 0.2), {1: iso(1e-3)})


def test_section_analyses_from_files(tmp_path):
    from pre_processing.section_integration.thin_walled.job_input import section_analyses_from_files

    (tmp_path / "laminate.txt").write_text(
        "[Ply_Material]\n[material_id] [E1] [E2] [G12] [nu12] [G13] [G23]\n"
        f"1 {E} {E} {G} {NU} {G} {G}\n"
        "[Laminate]\n[laminate_id] [ply] [material_id] [angle_deg] [thickness]\n1 1 1 0.0 0.002\n",
        encoding="utf-8",
    )
    (tmp_path / "section_geometry.txt").write_text(
        "[Section_Point]\n[section_id] [point_id] [y] [z]\n"
        "1 1 -0.2 -0.1\n1 2 0.2 -0.1\n1 3 0.2 0.1\n1 4 -0.2 0.1\n"
        "[Section_Wall]\n[section_id] [wall_id] [point_start] [point_end] [laminate_id]\n"
        "1 1 1 2 1\n1 2 2 3 1\n1 3 3 4 1\n1 4 4 1 1\n",
        encoding="utf-8",
    )
    out = section_analyses_from_files(str(tmp_path / "laminate.txt"), str(tmp_path / "section_geometry.txt"), hoop="free")
    K = out[1].stiffness
    assert K[0, 0] == pytest.approx(E * 1.2 * 0.002, rel=1e-10)
