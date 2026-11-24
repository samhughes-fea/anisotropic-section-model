"""The package's one axis convention (``axes``): the engine and BeamDyn conversions."""

import numpy as np
import pytest

from anisotropic_section_model import axes
from anisotropic_section_model.thin_wall import BEAMDYN_FROM_ONEDFEM


def _spd(n, seed=0):
    a = np.random.default_rng(seed).normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def test_engine_names_map_to_public_names():
    # engine (x, y, z) = public (z, x, y): eps stays, kappa_y -> kappa_x, gamma_xy -> gamma_x, phi -> kappa_z
    relabel = {"eps": "eps", "kappa_y": "kappa_x", "kappa_z": "kappa_y",
               "gamma_xy": "gamma_x", "gamma_xz": "gamma_y", "phi": "kappa_z"}
    public = [relabel[axes.ENGINE_STRAINS[i]] for i in axes.FROM_ENGINE]
    assert tuple(public) == axes.STRAINS


@pytest.mark.parametrize("n", [6, 7])
def test_from_engine_and_to_engine_are_inverse(n):
    K = _spd(n)
    np.testing.assert_array_equal(axes.to_engine(axes.from_engine(K)), K)
    v = np.arange(n, dtype=float)
    np.testing.assert_array_equal(axes.from_engine(axes.to_engine(v)), v)
    if n == 7:  # the warping entry keeps its place
        assert axes.from_engine(K)[6, 6] == K[6, 6]


def test_from_engine_conserves_energy():
    K, e = _spd(6, 1), np.random.default_rng(2).normal(size=6)
    assert np.isclose(e @ K @ e, axes.from_engine(e) @ axes.from_engine(K) @ axes.from_engine(e))


def test_from_engine_stacks_matrices():
    K = np.stack([_spd(6, s) for s in range(3)])
    out = axes.from_engine(K)
    for i in range(3):
        np.testing.assert_array_equal(out[i], axes.from_engine(K[i]))


def test_legacy_permutation_is_the_same():
    assert list(axes.FROM_ENGINE) == BEAMDYN_FROM_ONEDFEM


def test_section_analysis_from_public_tables_is_in_the_engine_frame():
    from anisotropic_section_model.section_analysis import SectionAnalysis

    K = _spd(6, 3)
    sa = SectionAnalysis.from_matrices(K, frame="public")
    assert sa.frame == "engine"
    np.testing.assert_allclose(sa.stiffness, axes.to_engine(K), rtol=1e-12)


def test_mirror_is_an_involution_and_kernel_twist_flips():
    np.testing.assert_array_equal(axes.MIRROR @ axes.MIRROR, np.eye(6))
    assert np.linalg.det(axes.MIRROR[:3, :3]) == -1.0
    np.testing.assert_array_equal(axes.twist_from_kernel([10.0, -2.5]), [-10.0, 2.5])
    np.testing.assert_array_equal(axes.twist_to_kernel(axes.twist_from_kernel(7.0)), 7.0)


def test_to_beamdyn_frame_untwists_then_mirrors():
    K = _spd(6, 4)
    K_twisted = axes.rotation6(15.0) @ K @ axes.rotation6(15.0).T  # a section twisted by +15 degrees
    np.testing.assert_allclose(axes.to_beamdyn_frame(K_twisted, 15.0), axes.MIRROR @ K @ axes.MIRROR.T,
                               atol=1e-12)


def test_rejects_short_or_non_square():
    with pytest.raises(ValueError):
        axes.from_engine(np.zeros(5))
    with pytest.raises(ValueError):
        axes.from_engine(np.zeros((6, 7)))
