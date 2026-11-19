"""ModalBeam (spectral Timoshenko elements) against exact and closed-form frequencies."""

from __future__ import annotations

import numpy as np
import pytest

from anisotropic_section_model import BeamDynBlade, ModalBeam, prismatic_frequencies
from anisotropic_section_model.beamdyn import rotation6
from anisotropic_section_model.modal import gll_points, mac

pytest.importorskip("scipy")

E, G, RHO = 70e9, 26e9, 2700.0
B, H = 0.1, 0.2
A, IX, IY, J = B * H, B * H ** 3 / 12, H * B ** 3 / 12, 0.229 * H * B ** 3
CANTILEVER = (1.875104069, 4.694091133, 7.854757438)


def rectangle(shear=5.0 / 6.0):
    S = np.diag([shear * G * A, shear * G * A, E * A, E * IX, E * IY, G * J])
    M = np.diag([RHO * A] * 3 + [RHO * IX, RHO * IY, RHO * (IX + IY)])
    return S, M


def coupled():
    """A fully coupled section: every stiffness coupling and an offset mass centre."""
    D = np.sqrt([2e8, 1.5e8, 5e9, 3e7, 8e7, 1e7])
    C = np.eye(6)
    C[2, 5] = C[5, 2] = 0.4      # extension-twist
    C[3, 5] = C[5, 3] = 0.3      # bending-twist
    C[0, 4] = C[4, 0] = 0.2      # shear-bending
    C[1, 3] = C[3, 1] = -0.25
    S = np.outer(D, D) * C
    m, xc, yc = 300.0, 0.12, -0.05
    M = np.zeros((6, 6))
    M[0, 0] = M[1, 1] = M[2, 2] = m
    M[0, 5] = M[5, 0] = -m * yc
    M[1, 5] = M[5, 1] = m * xc
    M[2, 3] = M[3, 2] = m * yc
    M[2, 4] = M[4, 2] = -m * xc
    M[3, 3], M[4, 4] = 4.0 + m * yc * yc, 9.0 + m * xc * xc
    M[3, 4] = M[4, 3] = -0.5 - m * xc * yc
    M[5, 5] = 13.0 + m * (xc * xc + yc * yc)
    return S, M


def test_gll_points():
    assert np.allclose(gll_points(2), [-1.0, 0.0, 1.0])
    assert np.allclose(gll_points(4), [-1.0, -np.sqrt(3 / 7), 0.0, np.sqrt(3 / 7), 1.0])


def test_cantilever_closed_forms():
    """Bending tends to Euler-Bernoulli as the shear stiffness rises and the rotary
    inertia falls (by 100 here, which leaves 1e-4 of their effect); torsion is exact."""
    L = 10.0
    S, M = rectangle()
    S[:2, :2] *= 100.0
    M[3:5, 3:5] *= 0.01
    r = ModalBeam.prismatic(S, M, L).clamp(0.0).modes(12)
    m = RHO * A
    f = r.frequencies
    labels = r.labels()
    fx = f[[i for i, s in enumerate(labels) if s.endswith("x-bending")]]
    fy = f[[i for i, s in enumerate(labels) if s.endswith("y-bending")]]
    for k, (bl, tol) in enumerate(zip(CANTILEVER, (1e-4, 5e-4, 2e-3))):
        assert fx[k] == pytest.approx(bl ** 2 / (2 * np.pi * L ** 2) * np.sqrt(E * IY / m), rel=tol)
        assert fy[k] == pytest.approx(bl ** 2 / (2 * np.pi * L ** 2) * np.sqrt(E * IX / m), rel=tol)
    ft = f[labels.index("1st torsion")]
    assert ft == pytest.approx(np.sqrt(G * J / (RHO * (IX + IY))) / (4 * L), rel=1e-9)


def test_axial_mode_closed_form():
    S, M = rectangle()
    L = 0.5   # short: the axial mode comes early
    r = ModalBeam.prismatic(S, M, L).clamp(0.0).modes(12)
    fa = r.frequencies[r.labels().index("1st axial")]
    assert fa == pytest.approx(np.sqrt(E / RHO) / (4 * L), rel=1e-9)


@pytest.mark.parametrize("ends", [("clamped", "free"), ("pinned", "pinned"), ("clamped", "clamped"),
                                  ("clamped", "pinned")])
def test_fully_coupled_against_exact(ends):
    S, M = coupled()
    L = 20.0
    beam = ModalBeam.prismatic(S, M, L)
    for e, s in zip(ends, (0.0, L)):
        if e != "free":
            (beam.clamp if e == "clamped" else beam.pin)(s)
    r = beam.modes(10)
    exact = prismatic_frequencies(S, M, L, 1.001 * r.frequencies[-1], ends)
    assert len(exact) == 10
    assert np.allclose(r.frequencies, exact, rtol=1e-7)


def test_free_free_has_six_rigid_modes_then_the_exact_ones():
    S, M = coupled()
    L = 20.0
    r = ModalBeam.prismatic(S, M, L).modes(12)
    assert np.all(r.frequencies[:6] < 1e-4 * r.frequencies[6])
    exact = prismatic_frequencies(S, M, L, 1.001 * r.frequencies[-1], ("free", "free"))
    assert np.allclose(r.frequencies[6:], exact, rtol=1e-7)


def test_repeated_frequencies_of_a_square_section():
    S = np.diag([1e8, 1e8, 1e9, 2e6, 2e6, 1e6])
    M = np.diag([50.0, 50.0, 50.0, 0.1, 0.1, 0.2])
    r = ModalBeam.prismatic(S, M, 8.0).clamp(0.0).modes(4)
    exact = prismatic_frequencies(S, M, 8.0, 1.001 * r.frequencies[3])
    assert len(exact) == 4
    assert np.allclose(r.frequencies, exact, rtol=1e-7)
    assert r.frequencies[1] == pytest.approx(r.frequencies[0], rel=1e-10)


def test_frame_angle_and_mass_normalisation():
    """Turning every station's frame turns the modes and keeps the frequencies; the shapes
    are mass-normalised and orthogonal."""
    S, M = coupled()
    a = ModalBeam.prismatic(S, M, 15.0).clamp(0.0).modes(6)
    R = rotation6(30.0)
    b = ModalBeam.prismatic(S, M, 15.0, frame_angle=30.0).clamp(0.0).modes(6)
    assert np.allclose(a.frequencies, b.frequencies, rtol=1e-8)
    # the same frequencies from matrices turned by hand into the beam's axes
    c = ModalBeam.prismatic(R @ S @ R.T, R @ M @ R.T, 15.0).clamp(0.0).modes(6)
    assert np.allclose(b.frequencies, c.frequencies, rtol=1e-8)
    for k in range(6):
        assert mac(b.shapes[k], c.shapes[k]) == pytest.approx(1.0, abs=1e-8)
    phi = b.shapes.reshape(6, -1)
    assert np.allclose(phi @ b.M @ phi.T, np.eye(6), atol=1e-8)
    assert np.allclose(phi @ b.K @ phi.T, np.diag(b.omega ** 2), rtol=1e-8, atol=1e-6 * b.omega[-1] ** 2)


def test_tapered_beam_converges_and_point_mass_lowers_frequencies():
    S0, M0 = rectangle()
    s = np.linspace(0.0, 30.0, 7)
    scale = np.linspace(1.0, 0.3, 7)
    K = np.array([S0 * f ** 2 for f in scale])
    M = np.array([M0 * f for f in scale])
    f6 = ModalBeam(s, K, M).clamp(0.0).modes(6).frequencies
    f8 = ModalBeam(s, K, M, order=8, max_element_length=1.0).clamp(0.0).modes(6).frequencies
    assert np.allclose(f6, f8, rtol=2e-5)
    heavy = ModalBeam(s, K, M).clamp(0.0).add_mass(30.0, 20.0).modes(6).frequencies
    assert np.all(heavy < f6)


def test_curved_axis_straight_limit():
    """A straight axis given as points matches the default straight axis."""
    S, M = coupled()
    a = ModalBeam.prismatic(S, M, 12.0).clamp(0.0).modes(6)
    pts = np.column_stack([np.zeros(5), np.zeros(5), np.linspace(0.0, 12.0, 5)])
    b = ModalBeam.prismatic(S, M, 12.0, axis=pts).clamp(0.0).modes(6)
    assert np.allclose(a.frequencies, b.frequencies, rtol=1e-9)


def test_from_beamdyn_names_flap_and_edge():
    n = 5
    S, M = rectangle()  # bending in x (flapwise in BeamDyn's axes) is the softer
    kp = np.column_stack([np.zeros(4), np.zeros(4), np.linspace(0.0, 20.0, 4)])
    blade = BeamDynBlade(np.linspace(0.0, 1.0, n), [S] * n, [M] * n, kp, np.zeros(4))
    r = ModalBeam.from_beamdyn(blade).modes(3)
    assert r.labels()[:2] == ["1st flap", "1st edge"]
    assert r.table()[0]["flap"] > 0.99
