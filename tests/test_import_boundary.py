"""The package boundary of the split:

    implicit_geometry_kernel  <-  composite_laminate_model  <-  anisotropic_section_model

Walls are laminates, so the only names this package takes from
``composite_laminate_model`` are ``Laminate`` and ``integrate_abd``. It does not import
the geometry kernel: ``ThinWalledSection.from_geometry`` reads a cut by duck typing.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "anisotropic_section_model"
ALLOWED = {
    "composite_laminate_model.laminate_mechanics": {"Laminate"},
    "composite_laminate_model.laminate_kinematics": {"integrate_abd"},
}
FORBIDDEN = {"implicit_geometry_kernel", "laminate_modelling_package"}


def _imports():
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                yield path.name, node.module, {a.name for a in node.names}
            elif isinstance(node, ast.Import):
                for a in node.names:
                    yield path.name, a.name, set()


def test_only_laminate_and_integrate_abd_come_from_the_laminate_model():
    for name, module, names in _imports():
        if module.split(".")[0] == "composite_laminate_model":
            assert module in ALLOWED, (name, module)
            assert names <= ALLOWED[module], (name, module, names - ALLOWED[module])


def test_no_geometry_kernel_or_old_package_imports():
    for name, module, _ in _imports():
        assert module.split(".")[0] not in FORBIDDEN, (name, module)

