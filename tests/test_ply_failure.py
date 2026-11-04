"""Ply failure criteria (``pre_processing/section_integration/ply_failure.py``) against hand values."""

from __future__ import annotations

import numpy as np
import pytest

from onedfem.pre_processing.section_integration.ply_failure import (
    CRITERIA,
    PlyStrength,
    exposures,
    hashin,
    max_stress,
    puck,
    tsai_wu,
)

# a carbon-epoxy ply (T300/914C-like strengths, Pa)
ST = PlyStrength(Xt=1500e6, Xc=900e6, Yt=27e6, Yc=200e6, S12=80e6, p_TL_t=0.35, p_TL_c=0.30, p_TT=0.25)


def _s(s11=0.0, s22=0.0, s33=0.0, t23=0.0, t13=0.0, t12=0.0):
    return np.array([s11, s22, s33, t23, t13, t12])


@pytest.mark.parametrize("state, mode", [
    (_s(s11=1500e6), "FT"), (_s(s11=-900e6), "FC"), (_s(s22=27e6), "MT"), (_s(s22=-200e6), "MC"),
])
def test_every_criterion_is_one_on_a_uniaxial_strength(state, mode):
    e = exposures(state, ST)
    for c in CRITERIA:
        assert e[c] == pytest.approx(1.0, rel=1e-9), c
    assert e["hashin_mode"] == mode
    assert e["margin"] == pytest.approx(0.0, abs=1e-9)


def test_pure_in_plane_shear():
    e = exposures(_s(t12=80e6), ST)
    for c in CRITERIA:
        assert e[c] == pytest.approx(1.0, rel=1e-6), c
    assert e["puck_mode"] == "IFF-A"          # sigma_n = 0 on the fracture plane: the tensile branch


def test_exposure_is_linear_in_the_load():
    """The exposure scales with the load (RF = 1 / f), also for the quadratic criteria."""
    s = _s(s11=300e6, s22=-60e6, t12=35e6, t23=5e6)
    for k in (0.5, 2.0, 3.0):
        a, b = exposures(s, ST), exposures(k * s, ST)
        for c in CRITERIA:
            assert b[c] == pytest.approx(k * a[c], rel=1e-6), c


def test_tsai_wu_hand_value():
    s11, s22, t12 = 500e6, 10e6, 30e6
    F1, F2 = 1 / ST.Xt - 1 / ST.Xc, 1 / ST.Yt - 1 / ST.Yc
    F11, F22, F66 = 1 / (ST.Xt * ST.Xc), 1 / (ST.Yt * ST.Yc), 1 / ST.S12**2
    F12 = -0.5 * np.sqrt(F11 * F22)
    a = F11 * s11**2 + F22 * s22**2 + 2 * F12 * s11 * s22 + F66 * t12**2
    b = F1 * s11 + F2 * s22
    RF = (-b + np.sqrt(b * b + 4 * a)) / (2 * a)
    assert tsai_wu(_s(s11=s11, s22=s22, t12=t12), ST) == pytest.approx(1 / RF, rel=1e-12)


def test_hashin_hand_values():
    f, m = hashin(_s(s11=900e6, t12=40e6), ST)
    assert m == "FT" and f == pytest.approx(np.sqrt((900 / 1500) ** 2 + (40 / 80) ** 2), rel=1e-12)
    f, m = hashin(_s(s22=15e6, t12=50e6), ST)
    assert m == "MT" and f == pytest.approx(np.sqrt((15 / 27) ** 2 + (50 / 80) ** 2), rel=1e-12)


def _puck_plane_stress(s2, t21, st):
    """Puck's closed-form plane-stress IFF modes A, B, C (VDI 2014)."""
    RTT = st.R_TT
    t21c = st.S12 * np.sqrt(1 + 2 * st.p_TT)
    if s2 >= 0:
        return np.sqrt((t21 / st.S12) ** 2 + (1 - st.p_TL_t * st.Yt / st.S12) ** 2 * (s2 / st.Yt) ** 2) \
            + st.p_TL_t * s2 / st.S12, "IFF-A"
    if abs(s2 / t21) <= RTT / abs(t21c) if t21 else False:
        return (np.sqrt(t21**2 + (st.p_TL_c * s2) ** 2) + st.p_TL_c * s2) / st.S12, "IFF-B"
    return ((t21 / (2 * (1 + st.p_TT) * st.S12)) ** 2 + (s2 / st.Yc) ** 2) * st.Yc / (-s2), "IFF-C"


@pytest.mark.parametrize("s2, t21", [(10e6, 30e6), (20e6, 5e6), (-20e6, 70e6), (-60e6, 60e6), (-150e6, 20e6),
                                     (-120e6, 0.0)])
def test_puck_reduces_to_modes_a_b_c_in_plane_stress(s2, t21):
    """The action-plane search reproduces the closed-form plane-stress modes. Mode C is
    consistent only with p_TL_c / R_TL = p_TT / R_TT, Puck's own coupling of the two."""
    k = 0.30 * 200e6 / 80e6                       # p_TT 2 (1 + p_TT) / Yc = p_TL_c / S12
    st = PlyStrength(Xt=1500e6, Xc=900e6, Yt=27e6, Yc=200e6, S12=80e6, p_TL_t=0.35, p_TL_c=0.30,
                     p_TT=(-2 + np.sqrt(4 + 8 * k)) / 4)
    assert st.p_TT / st.R_TT == pytest.approx(st.p_TL_c / st.S12, rel=1e-12)
    ref, mode = _puck_plane_stress(s2, t21, st)
    f, m, _ = puck(_s(s22=s2, t12=t21), st)
    assert m == mode
    assert f == pytest.approx(ref, rel=2e-4)


def test_puck_fracture_angle_under_transverse_compression():
    """Pure transverse compression fractures on an inclined plane, about 50 degrees for p_TT = 0.25."""
    _, m, ang = puck(_s(s22=-100e6), ST)
    assert m == "IFF-C"
    # theta_fp = arccos(sqrt(R_TT^A / (R_perp^- ... )): for p_TT = 0.25, cos^2 = 1 / (2 (1 + p_TT))
    assert abs(ang) == pytest.approx(np.degrees(np.arccos(np.sqrt(1 / (2 * (1 + ST.p_TT))))), abs=0.2)


def test_max_stress_mode_names():
    f, m = max_stress(np.stack([_s(s22=-100e6), _s(t13=60e6)]), ST)
    assert list(m) == ["s22c", "t13"]
    assert f == pytest.approx([0.5, 0.75])


def test_strengths_are_checked():
    with pytest.raises(ValueError, match="Xt"):
        PlyStrength(Xt=0.0, Xc=1, Yt=1, Yc=1, S12=1)
    with pytest.raises(ValueError, match="6 components"):
        exposures(np.zeros(5), ST)
