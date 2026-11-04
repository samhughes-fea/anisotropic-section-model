"""GBT bases along a beam of changing sections, and taper and pretwist on GBT sections."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from test_gbt_cross_section_modes import box  # noqa: E402

from pre_processing.section_integration.gbt_alignment import (  # noqa: E402
    align_gbt_sections,
    gbt_signature,
    rotate_gbt_basis,
)
from pre_processing.section_integration.section_analysis import SectionAnalysis  # noqa: E402
from pre_processing.section_integration.thin_walled.taper import SectionTaper  # noqa: E402


def _orthogonal(n, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return q


@pytest.fixture(scope="module")
def channel():
    """An unsymmetric section (no doubly symmetric modes): a box with one wall thicker."""
    from pre_processing.section_integration.thin_walled import ThinWalledSection, ThinWalledSectionAnalysis, Wall
    from pre_processing.section_integration.thin_walled.laminate import WallLaminate

    pts = {1: (-0.15, -0.06), 2: (0.15, -0.06), 3: (0.15, 0.06), 4: (-0.12, 0.06)}
    walls = [Wall(1, 1, 2, 1), Wall(2, 2, 3, 2), Wall(3, 3, 4, 1), Wall(4, 4, 1, 1)]
    lams = {1: WallLaminate.isotropic(70e9, 0.3, 2e-3), 2: WallLaminate.isotropic(70e9, 0.3, 3e-3)}
    tw = ThinWalledSectionAnalysis(ThinWalledSection(pts, walls), lams, hoop="restrained", n_sub=2)
    return SectionAnalysis.from_thin_walled(tw, gbt_modes=4)


def test_turning_the_bases_keeps_the_section_model(channel):
    p = channel.gbt_sv_modes
    r = rotate_gbt_basis(channel, _orthogonal(p, 1), _orthogonal(4, 2))
    np.testing.assert_allclose(np.linalg.eigvalsh(r.gbt_stiffness), np.linalg.eigvalsh(channel.gbt_stiffness),
                               rtol=1e-9, atol=1e-9 * np.abs(channel.gbt_stiffness).max())
    np.testing.assert_allclose(r.gbt_stiffness[:6, :6], channel.gbt_stiffness[:6, :6], rtol=1e-12)
    np.testing.assert_allclose(r.gbt_mass[:6, :6], channel.gbt_mass[:6, :6], rtol=1e-12)


def test_alignment_undoes_a_turn_swap_and_sign_flip(channel):
    p = channel.gbt_sv_modes
    Rm = np.eye(4)[[1, 0, 2, 3]]
    Rm[2] *= -1.0                                       # modes 1 and 2 swapped, mode 3 flipped
    turned = rotate_gbt_basis(channel, _orthogonal(p, 3), Rm)
    out, info = align_gbt_sections([channel, turned], [[0, 1], [1, 2]])
    assert out[0] is channel
    np.testing.assert_allclose(out[1].gbt_stiffness, channel.gbt_stiffness, atol=1e-6 * np.abs(channel.gbt_stiffness).max())
    np.testing.assert_allclose(out[1].gbt_mass, channel.gbt_mass, atol=1e-6 * np.abs(channel.gbt_mass).max())
    assert info[1]["match"] > 0.999
    Sw, Sm = gbt_signature(out[1])
    Sw0, Sm0 = gbt_signature(channel)
    np.testing.assert_allclose(Sm, Sm0, atol=1e-6)


def test_elements_of_one_section_share_its_bases(channel):
    other = rotate_gbt_basis(channel, np.eye(channel.gbt_sv_modes), -np.eye(4))
    out, _ = align_gbt_sections([channel, other, other], [[0, 1], [1, 2], [2, 3]])
    assert out[1] is out[2]
    np.testing.assert_allclose(out[1].gbt_stiffness, channel.gbt_stiffness, atol=1e-6 * np.abs(channel.gbt_stiffness).max())


def test_taper_and_pretwist_reach_the_gbt_stiffness(channel):
    t = SectionTaper(np.array([[1e6, 2e5], [3e4, -1e4], [5e3, 2e4]]), np.array([[4e5, 1e4], [1e4, 3e5]]))
    for new in (channel.with_taper(t), channel.with_pretwist(0.05)):
        dg = new.gbt_stiffness - channel.gbt_stiffness
        ds = new.stiffness - channel.stiffness
        np.testing.assert_allclose(dg[:6, :6], ds, rtol=1e-12, atol=1e-9 * np.abs(ds).max())
        assert not np.any(dg[6:, :]) and not np.any(dg[:, 6:])


def test_prismatic_beam_needs_no_turn():
    sa = SectionAnalysis.from_thin_walled(box(0.3, 0.2, 2e-3, n_sub=2), gbt_modes=2)
    out, info = align_gbt_sections([sa, sa], [[0, 1], [1, 2]])
    assert out[0] is sa and out[1] is sa and not info
