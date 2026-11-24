"""The package boundary with 1d-fem (``docs/package_boundary.md``).

ASM works on one cross-section at a time. Solving along the span (elements, assembly,
supports, solves) is 1d-fem's. These tests stop span-wise solvers creeping back in: a public
callable that takes a span, a mesh or element connectivity must be on the allow-list below,
which holds the frozen reference beams and what contract v1 still carries.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path

import anisotropic_section_model as asm

SRC = Path(asm.__file__).resolve().parent

#: parameters that only make sense for something solved along a beam
SPAN_PARAMETERS = {"length", "span", "z_nodes", "n_elements", "connectivity", "n_nodes"}

#: public callables allowed to take them, with the reason
ALLOWED = {
    # frozen reference beams (no new features): quick checks of a section in the examples
    "anisotropic_section_model.beam.TimoshenkoBeam": "frozen reference beam",
    "anisotropic_section_model.beam.euler_bernoulli_beam": "frozen reference beam",
    "anisotropic_section_model.torsion.RestrainedTorsionBeam": "frozen reference beam",
    "anisotropic_section_model.gbt.GeneralizedBeam": "frozen reference beam",
    # section-level, despite the name: a wall's own length
    "anisotropic_section_model.thin_walled.fields.WallField": "length of a wall, not a span",
}


def _public_callables():
    for info in pkgutil.walk_packages([str(SRC)], prefix="anisotropic_section_model."):
        try:
            module = importlib.import_module(info.name)
        except (ImportError, OSError):     # optional dependencies (geometry kernel, GEOS)
            continue
        for name, obj in vars(module).items():
            if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
                continue
            if inspect.isclass(obj):
                targets = [("", obj.__init__)] + [(f".{m}", f) for m, f in vars(obj).items()
                                                  if not m.startswith("_") and callable(f)]
                for suffix, f in targets:
                    yield f"{module.__name__}.{name}", f"{module.__name__}.{name}{suffix}", f
            elif inspect.isfunction(obj):
                yield f"{module.__name__}.{name}", f"{module.__name__}.{name}", obj


def test_no_span_wise_solver_outside_the_allow_list():
    offending = set()
    for owner, qualname, f in _public_callables():
        f = getattr(f, "__func__", f)
        try:
            params = set(inspect.signature(f).parameters)
        except (TypeError, ValueError):
            continue
        if params & SPAN_PARAMETERS and owner not in ALLOWED:
            offending.add(f"{qualname}({', '.join(sorted(params & SPAN_PARAMETERS))})")
    assert not offending, ("public callables take a span, mesh or connectivity; solving along the beam is "
                           "1d-fem's (docs/package_boundary.md):\n" + "\n".join(sorted(offending)))


def test_no_import_of_the_beam_code():
    """ASM never imports 1d-fem: the dependency runs one way."""
    offending = []
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            offending += [f"{path.relative_to(SRC)}:{node.lineno}" for n in names if n.split(".")[0] == "onedfem"]
    assert not offending, offending


def test_section_contract_v1():
    """What 1d-fem's ``onedfem.sections`` needs from every ``SectionAnalysis`` builder."""
    import numpy as np

    from anisotropic_section_model.section_analysis import SECTION_API_VERSION, SectionAnalysis

    assert SECTION_API_VERSION[0] == 2
    builders = [
        SectionAnalysis.from_isotropic(E=70e9, G=26e9, A=0.01, I_y=2e-5, I_z=5e-6, J_t=1e-5, rho=2700.0),
        SectionAnalysis.from_matrices(np.diag([7e8, 1.4e6, 3.5e5, 2e8, 2e8, 2.6e5]),
                                      np.diag([27.0, 27.0, 27.0, 0.07, 0.054, 0.0135])),
    ]
    for sa in builders:
        assert sa.frame == "engine"
        assert sa.stiffness.shape == (6, 6) and np.allclose(sa.stiffness, sa.stiffness.T)
        assert sa.classical_stiffness.shape == (4, 4)
        assert len(sa.shear_centre) == 2 and len(sa.tension_centre) == 2
        M = sa.mass_matrix()
        assert M.shape == (6, 6) and np.allclose(M, M.T)
        assert np.all(np.linalg.eigvalsh(sa.stiffness) > 0)
