"""Classical laminate theory for thin-walled section walls."""

import numpy as np
import pytest

from pre_processing.section_integration.thin_walled import (
    Ply,
    PlyMaterial,
    WallLaminate,
    beam_wall_stiffness,
    condense,
    condensed_wall_stiffness,
    gbt_wall_stiffness,
)

CFRP = PlyMaterial(E1=140e9, E2=10e9, G12=5e9, nu12=0.3, G13=5e9, G23=3.5e9, rho=1600.0)
E, NU, T = 70e9, 0.3, 2e-3
G = E / (2 * (1 + NU))


def textbook_Q_bar(Q, theta):
    """Explicit rotated stiffness formulas (Jones, Mechanics of Composite Materials)."""
    c, s = np.cos(theta), np.sin(theta)
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    q11 = Q11 * c**4 + 2 * (Q12 + 2 * Q66) * s**2 * c**2 + Q22 * s**4
    q22 = Q11 * s**4 + 2 * (Q12 + 2 * Q66) * s**2 * c**2 + Q22 * c**4
    q12 = (Q11 + Q22 - 4 * Q66) * s**2 * c**2 + Q12 * (s**4 + c**4)
    q66 = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * s**2 * c**2 + Q66 * (s**4 + c**4)
    q16 = (Q11 - Q12 - 2 * Q66) * s * c**3 + (Q12 - Q22 + 2 * Q66) * s**3 * c
    q26 = (Q11 - Q12 - 2 * Q66) * s**3 * c + (Q12 - Q22 + 2 * Q66) * s * c**3
    return np.array([[q11, q12, q16], [q12, q22, q26], [q16, q26, q66]])


def test_isotropic_reduced_stiffness():
    Q = PlyMaterial.isotropic(E, NU).reduced_stiffness()
    assert Q[0, 0] == pytest.approx(E / (1 - NU**2))
    assert Q[1, 1] == pytest.approx(E / (1 - NU**2))
    assert Q[0, 1] == pytest.approx(NU * E / (1 - NU**2))
    assert Q[2, 2] == pytest.approx(G)


def test_isotropic_Q_bar_is_rotation_invariant():
    mat = PlyMaterial.isotropic(E, NU)
    Q = mat.reduced_stiffness()
    for ang in (0.0, 17.0, 45.0, 90.0, -60.0):
        np.testing.assert_allclose(Ply(mat, ang, T).rotated_stiffness(), Q, rtol=1e-12, atol=1e-3)


@pytest.mark.parametrize("ang", [0.0, 15.0, 30.0, 45.0, 60.0, 90.0, -45.0, 123.0])
def test_Q_bar_matches_textbook_formulas(ang):
    ply = Ply(CFRP, ang, T)
    expected = textbook_Q_bar(CFRP.reduced_stiffness(), np.deg2rad(ang))
    np.testing.assert_allclose(ply.rotated_stiffness(), expected, rtol=1e-12, atol=1e-6 * 140e9)


def test_Q_bar_strain_energy_invariance():
    rng = np.random.default_rng(0)
    ply = Ply(CFRP, 37.0, T)
    from pre_processing.section_integration.thin_walled.laminate import strain_rotation

    Tm = strain_rotation(ply.angle_rad)
    for _ in range(5):
        eps = rng.standard_normal(3)
        e_ply = Tm @ eps
        assert eps @ ply.rotated_stiffness() @ eps == pytest.approx(
            e_ply @ CFRP.reduced_stiffness() @ e_ply, rel=1e-12
        )


def test_ninety_degree_ply_swaps_directions():
    Qb = Ply(CFRP, 90.0, T).rotated_stiffness()
    Q = CFRP.reduced_stiffness()
    assert Qb[0, 0] == pytest.approx(Q[1, 1])
    assert Qb[1, 1] == pytest.approx(Q[0, 0])


def test_rotated_transverse_shear():
    c, s = np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0))
    Hs = Ply(CFRP, 30.0, T).rotated_transverse_shear_stiffness()
    assert Hs[0, 0] == pytest.approx(CFRP.G13 * c**2 + CFRP.G23 * s**2)
    assert Hs[1, 1] == pytest.approx(CFRP.G23 * c**2 + CFRP.G13 * s**2)
    assert Hs[0, 1] == pytest.approx((CFRP.G13 - CFRP.G23) * c * s)


def test_isotropic_wall_ABD_and_H():
    lam = WallLaminate.isotropic(E, NU, T)
    Q = PlyMaterial.isotropic(E, NU).reduced_stiffness()
    np.testing.assert_allclose(lam.A(), Q * T)
    np.testing.assert_allclose(lam.B(), 0.0, atol=1e-6)
    np.testing.assert_allclose(lam.D(), Q * T**3 / 12)
    np.testing.assert_allclose(lam.H(), np.eye(2) * 5 / 6 * G * T)
    assert lam.thickness == pytest.approx(T)


def test_split_isotropic_plies_equal_single_ply():
    one = WallLaminate.isotropic(E, NU, T)
    mat = PlyMaterial.isotropic(E, NU)
    four = WallLaminate([Ply(mat, a, T / 4) for a in (0.0, 45.0, -45.0, 90.0)])
    np.testing.assert_allclose(four.ABD(), one.ABD(), rtol=1e-10, atol=1e-6)


def test_symmetric_layup_has_no_coupling():
    lam = WallLaminate([Ply(CFRP, a, 0.125e-3) for a in (0, 45, -45, 90, 90, -45, 45, 0)])
    np.testing.assert_allclose(lam.B(), 0.0, atol=1e-9 * np.abs(lam.A()).max() * 1e-3)


def test_unsymmetric_cross_ply_has_coupling():
    lam = WallLaminate([Ply(CFRP, 0.0, 0.5e-3), Ply(CFRP, 90.0, 0.5e-3)])
    B = lam.B()
    assert abs(B[0, 0]) > 1e-3 * lam.A()[0, 0] * lam.thickness
    assert B[0, 0] == pytest.approx(-B[1, 1])


def test_balanced_angle_ply_has_no_A16():
    lam = WallLaminate([Ply(CFRP, a, 0.25e-3) for a in (30, -30, -30, 30)])
    assert abs(lam.A()[0, 2]) < 1e-9 * lam.A()[0, 0]
    assert abs(lam.D()[0, 2]) > 0.0  # bending-twisting coupling remains


def test_mass_per_area():
    lam = WallLaminate([Ply(CFRP, 0.0, 1e-3), Ply(CFRP, 90.0, 2e-3)])
    assert lam.mass_per_area == pytest.approx(1600.0 * 3e-3)


def test_bad_inputs():
    with pytest.raises(ValueError):
        PlyMaterial(E1=-1.0, E2=1.0, G12=1.0, nu12=0.3, G13=1.0, G23=1.0)
    with pytest.raises(ValueError):
        PlyMaterial(E1=1.0, E2=10.0, G12=1.0, nu12=0.5, G13=1.0, G23=1.0)
    with pytest.raises(ValueError):
        Ply(CFRP, 0.0, 0.0)
    with pytest.raises(ValueError):
        WallLaminate([])


# --------------------------------------------------------------------------- #
# Condensation
# --------------------------------------------------------------------------- #


def test_isotropic_beam_wall_stiffness():
    ws = beam_wall_stiffness(WallLaminate.isotropic(E, NU, T))
    assert ws.strains == ("eps_xx", "gamma_xs", "kappa_xx", "kappa_xs")
    expected = np.diag([E * T, G * T, E * T**3 / 12, G * T**3 / 12])
    np.testing.assert_allclose(ws.matrix, expected, rtol=1e-12, atol=1e-9)
    assert ws["eps_xx", "eps_xx"] == pytest.approx(E * T)


def test_isotropic_gbt_wall_stiffness_keeps_plate_bending_in_s():
    ws = gbt_wall_stiffness(WallLaminate.isotropic(E, NU, T))
    assert ws.strains == ("eps_xx", "gamma_xs", "kappa_xx", "kappa_ss", "kappa_xs")
    D = E * T**3 / (12 * (1 - NU**2))
    assert ws["kappa_ss", "kappa_ss"] == pytest.approx(D)
    assert ws["kappa_xx", "kappa_xx"] == pytest.approx(D)
    assert ws["eps_xx", "eps_xx"] == pytest.approx(E * T)


@pytest.mark.parametrize("released", [("N_ss", "M_ss"), ("N_ss",), ("N_xs", "M_ss", "M_xs")])
def test_condensed_equals_inverse_of_kept_compliance(released):
    lam = WallLaminate([Ply(CFRP, a, 0.3e-3) for a in (0, 30, 90, -60, 45)])
    ws = condensed_wall_stiffness(lam, released)
    C = np.linalg.inv(lam.ABD())
    idx = [("eps_xx", "eps_ss", "gamma_xs", "kappa_xx", "kappa_ss", "kappa_xs").index(s) for s in ws.strains]
    np.testing.assert_allclose(ws.matrix, np.linalg.inv(C[np.ix_(idx, idx)]), rtol=1e-8)
    np.testing.assert_allclose(ws.matrix, ws.matrix.T)
    assert np.all(np.linalg.eigvalsh(ws.matrix) > 0)


def test_condensation_never_stiffens():
    lam = WallLaminate([Ply(CFRP, a, 0.3e-3) for a in (0, 30, 90, -60, 45)])
    K = lam.ABD()
    Kc, kept = condense(K, [1, 4])
    gap = K[np.ix_(kept, kept)] - Kc
    assert np.all(np.linalg.eigvalsh(gap) > -1e-9 * np.abs(K).max())


def test_condense_rejects_bad_input():
    K = np.eye(3)
    with pytest.raises(ValueError):
        condense(K, [3])
    with pytest.raises(ValueError):
        condense(K, [0, 1, 2])
    with pytest.raises(ValueError):
        condensed_wall_stiffness(WallLaminate.isotropic(E, NU, T), ["N_zz"])
