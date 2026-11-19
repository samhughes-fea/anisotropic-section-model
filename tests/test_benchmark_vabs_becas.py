"""ASM's solid section against published BECAS and VABS results (``benchmarks/vabs_becas``).

The full comparison, with converged meshes and closed forms, is ``benchmarks/vabs_becas/run.py``
and its ``RESULTS.md``. These tests gate the same cases on meshes that run in seconds:

- BECAS user guide squares S1, S2, S3 (0, 22.5 and 90 degree orthotropic) on BECAS's own
  10 x 10 Q4 mesh: ASM reproduces the published 6 x 6 to its printed rounding.
- The same squares and the cylinders C1, C2 on quadratic meshes: ASM matches the closed forms
  where they exist, and BECAS to within its own (coarse-mesh) discretisation.
- VABS: the Chen et al. aluminium tube and the Yu et al. multi-layer composite pipe, all
  published terms including the extension-twist and shear-bending couplings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "vabs_becas"
sys.path.insert(0, str(BENCH))

from anisotropic_section_model.solid import SolidSectionAnalysis  # noqa: E402

from cases import CASES  # noqa: E402
from reference import BECAS_FROM_ASM, PUBLISHED, VABS_FROM_ASM  # noqa: E402
from run import exact, rounding, term  # noqa: E402


def _stiffness(case, n, kind):
    a = SolidSectionAnalysis(CASES[case](n, kind))
    p = BECAS_FROM_ASM if PUBLISHED[case]["order"] == "becas" else VABS_FROM_ASM
    return a, a.stiffness[np.ix_(p, p)]


def _rel(a, b):
    return abs(a / b - 1.0)


@pytest.mark.parametrize("case", ["S1", "S2", "S3_0", "S3_22.5", "S3_90"])
def test_becas_squares_on_becas_mesh(case):
    """Q4 10 x 10 (BECAS's mesh): every published term, couplings included, to its rounding."""
    _, K = _stiffness(case, 1, "Q4")
    for name, ref in PUBLISHED[case]["terms"].items():
        assert _rel(term(K, name), ref) <= rounding(ref) + 2e-4, name


def test_becas_square_offset_is_its_coarse_mesh():
    """The converged S1 shear and torsion stiffness are ~0.85 % below BECAS's 10 x 10 Q4 values,
    and on the exact torsion constant: the published values carry the bilinear mesh's error."""
    _, K = _stiffness("S1", 2, "Q8")
    ex = exact("S1")
    for name in ("K33", "K44", "K55", "K66"):
        assert _rel(term(K, name), ex[name]) < 1e-5, name
    for name in ("K11", "K66"):
        assert -0.0095 < term(K, name) / PUBLISHED["S1"]["terms"][name] - 1.0 < -0.0075


def test_t6_matches_q8():
    _, K8 = _stiffness("S3_22.5", 2, "Q8")
    _, K6 = _stiffness("S3_22.5", 2, "T6")
    np.testing.assert_allclose(K6, K8, rtol=2e-5, atol=1e-6 * abs(K8).max())  # T6 diagonals break symmetry at 1e-7


@pytest.mark.parametrize("case", ["C1", "C2"])
def test_becas_cylinders(case):
    a, K = _stiffness(case, 1, "Q8")
    for name, ref in PUBLISHED[case]["terms"].items():
        assert _rel(term(K, name), ref) < 2e-3, name
    for name, ref in exact(case).items():
        assert _rel(term(K, name), ref) < 1e-4, name
    centres = PUBLISHED[case].get("centres")
    if centres:
        assert a.shear_centre[0] == pytest.approx(centres["shear"][0], rel=1e-3)
        assert a.tension_centre[0] == pytest.approx(centres["tension"][0], rel=1e-3)
        assert abs(a.shear_centre[1]) < 1e-12 and abs(a.tension_centre[1]) < 1e-12


def test_vabs_aluminium_tube():
    """All terms within 0.05 % of VABS, except GJ: VABS's published 3.519e8 is 2.0 % above the
    exact G pi (ro^4 - ri^4) / 2 = 3.449e8, which ASM matches."""
    a, K = _stiffness("tube", 1, "Q8")
    for name, ref in PUBLISHED["tube"]["terms"].items():
        if name != "K44":
            assert _rel(term(K, name), ref) < 5e-4, name
    for name, ref in exact("tube").items():
        assert _rel(term(K, name), ref) < 1e-4, name
    assert term(K, "K44") / PUBLISHED["tube"]["terms"]["K44"] - 1.0 == pytest.approx(-0.020, abs=1e-3)
    m = a.mass_properties
    for name, ref in PUBLISHED["tube"]["mass"].items():
        assert _rel(m[name], ref) < 5e-4, name


def test_vabs_composite_pipe():
    """Main terms within 0.25 % and the three couplings (extension-twist K14, shear-bending K25
    and K36) within 1 % of VABS on a 2-level mesh; ``RESULTS.md`` has them within 0.2 % converged."""
    _, K = _stiffness("pipe", 2, "Q8")
    for name, ref in PUBLISHED["pipe"]["terms"].items():
        tol = 1e-2 if name in ("K14", "K25", "K36") else 2.5e-3
        assert np.sign(term(K, name)) == np.sign(ref), name
        assert _rel(term(K, name), ref) < tol, name


def test_vabs_pipe_layup_sign_is_not_free():
    """Reversing every ply angle flips the couplings' signs: the agreement fixes the angle convention."""
    a = SolidSectionAnalysis(CASES["pipe"](1, "Q8", sign=-1.0))
    K = a.stiffness[np.ix_(VABS_FROM_ASM, VABS_FROM_ASM)]
    for name in ("K14", "K25", "K36"):
        assert np.sign(term(K, name)) == -np.sign(PUBLISHED["pipe"]["terms"][name]), name
