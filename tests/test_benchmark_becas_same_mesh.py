"""ASM's solid section against BECAS on the same meshes (``benchmarks/becas_same_mesh``).

- Always: the composite sections built for the benchmark (MH-104-style two-web section, CAS
  and CUS box beams, layered composite cylinder) survive the BECAS input writer and reader
  unchanged, and carry the couplings they were built to have.
- When ``becas_reference.json`` holds BECAS's numbers for a built case: ASM's full 6 x 6 and
  centres match them.
- When ``BECAS_HOME`` points at a local BECAS: ASM on BECAS's own example meshes matches the
  BECAS and VABS outputs shipped with them (full precision), and the angle conventions of
  ``becas_io`` are the ones that reproduce BECAS's coupling signs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "becas_same_mesh"
sys.path.insert(0, str(BENCH))

from anisotropic_section_model.solid import SolidSectionAnalysis  # noqa: E402

from becas_cases import BUILT  # noqa: E402
from becas_io import becas_home, read_becas, read_becas_results, read_vabs_timoshenko, write_becas  # noqa: E402

P = [3, 4, 0, 1, 2, 5]                     # ASM -> BECAS order
VABS_FROM_BECAS = [2, 0, 1, 5, 3, 4]


def _K(mesh):
    a = SolidSectionAnalysis(mesh)
    return a, a.stiffness[np.ix_(P, P)]


def _scaled(K, R):
    K, R = np.asarray(K), np.asarray(R)
    return float((np.abs(K - R) / np.sqrt(np.abs(np.outer(np.diag(R), np.diag(R))))).max())


@pytest.fixture(scope="module")
def built():
    return {name: build(1) for name, build in BUILT.items()}


@pytest.mark.parametrize("name", list(BUILT))
def test_becas_writer_round_trip(built, name, tmp_path):
    mesh = built[name]
    back = read_becas(write_becas(mesh, tmp_path / name))
    assert len(back.elements) == len(mesh.elements)
    _, K = _K(mesh)
    _, Kb = _K(back)
    assert _scaled(Kb, K) < 1e-12


def test_built_couplings(built):
    """Each built section has the couplings its layup was chosen for (BECAS order)."""
    def c(K, i, j):
        return abs(K[i - 1, j - 1]) / np.sqrt(K[i - 1, i - 1] * K[j - 1, j - 1])
    _, cus = _K(built["box_cus"])
    _, cas = _K(built["box_cas"])
    _, tube = _K(built["layered_cylinder"])
    _, foil = _K(built["mh104"])
    assert c(cus, 3, 6) > 0.1 and c(cus, 1, 4) > 0.1         # extension-twist, shear-bending
    assert c(cas, 4, 6) > 0.1 and c(cas, 1, 3) > 0.1         # bending-twist, shear-extension
    assert c(cas, 3, 6) < 1e-2
    assert c(tube, 3, 6) > 0.01 and c(tube, 1, 4) > 0.01
    assert min(c(foil, i, j) for i, j in ((3, 5), (2, 6), (3, 4))) > 0.01


def _stored():
    path = BENCH / "becas_reference.json"
    return json.loads(path.read_text()) if path.exists() else {}


@pytest.mark.parametrize("name", list(BUILT))
def test_built_against_becas(built, name):
    ref = _stored().get(name)
    if ref is None:
        pytest.skip("BECAS has not been run on this case yet (benchmarks/becas_same_mesh/run.py --becas)")
    a, K = _K(built[name])
    assert _scaled(K, ref["K"]) < 1e-6
    size = float(np.ptp(built[name].nodes, axis=0).max())
    for got, want in ((a.shear_centre, ref["shear_centre"]), (a.tension_centre, ref["tension_centre"])):
        assert np.hypot(got[0] - want[0], got[1] - want[1]) / size < 1e-6


needs_becas = pytest.mark.skipif(becas_home() is None, reason="BECAS_HOME not set (BECAS is licensed, not in this repo)")


@needs_becas
@pytest.mark.parametrize("case, results", [("C1", "C1"), ("C2", "C2"), ("C4", "C4"), ("S3_0", "S3_0"),
                                           ("S2_10", "S2_1")])
def test_examples_against_shipped_becas(case, results):
    ex = becas_home() / "examples"
    R = read_becas_results(ex / results / "BECASresults.txt")
    a, K = _K(read_becas(ex / case))
    assert _scaled(K, R["K"]) < 1e-8
    assert np.allclose(a.shear_centre, R["shear_centre"], atol=1e-10)
    assert np.allclose(a.tension_centre, R["tension_centre"], atol=1e-10)


@needs_becas
@pytest.mark.parametrize("case", ["C3_1", "C3_10", "C3_100", "C3_1000", "C3_10000", "C3_100000"])
def test_two_material_cylinder_against_same_mesh_vabs(case):
    ex = becas_home() / "examples" / case
    Kv = read_vabs_timoshenko(ex / f"{case}.vabs.K")
    inv = np.argsort(VABS_FROM_BECAS)
    _, K = _K(read_becas(ex))
    assert _scaled(K, Kv[np.ix_(inv, inv)]) < 1e-8


@needs_becas
def test_angle_signs_are_fixed_by_the_data():
    """The pipe's (W3) and T2's couplings take BECAS's / VABS's signs only with the signs in
    ``becas_io``; flipping the fibre angle flips the pipe's extension-twist term."""
    ex = becas_home() / "examples"
    _, K = _K(read_becas(ex / "T2"))
    assert K[0, 2] < 0 and K[3, 5] > 0 and K[1, 3] < 0          # T2: K13, K46, K24 as published
    a = SolidSectionAnalysis(read_becas(ex / "W3"))
    Kv = a.stiffness[np.ix_([0, 3, 4, 5, 1, 2], [0, 3, 4, 5, 1, 2])]
    assert Kv[0, 3] == pytest.approx(9.82878e4, rel=2e-3)        # Yu 2005 VABS, extension-twist
    b = SolidSectionAnalysis(read_becas(ex / "W3", theta_sign=-1.0))
    assert b.stiffness[0, 5] == pytest.approx(-a.stiffness[0, 5], rel=1e-9)
