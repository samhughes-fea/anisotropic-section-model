"""The package boundary of the split:

    laminate_constitutive_model  <-  anisotropic_section_model

Walls are laminates, so the only names this package takes from
``laminate_constitutive_model`` are ``Laminate`` and ``integrate_abd``. It does not import
the geometry kernel (``ThinWalledSection.from_geometry`` reads a cut by duck typing) or
the shell FE (``shell_fem``), which it meets only in tests and examples.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "anisotropic_section_model"
ALLOWED = {
    "laminate_constitutive_model.laminate_mechanics": {"Laminate"},
    "laminate_constitutive_model.laminate_kinematics": {"integrate_abd"},
}
FORBIDDEN = {"implicit_geometry_kernel", "shell_fem", "composite_laminate_model", "laminate_modelling_package"}


def _imports():
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                yield path.name, node.module, {a.name for a in node.names}
            elif isinstance(node, ast.Import):
                for a in node.names:
                    yield path.name, a.name, set()


def test_only_laminate_and_integrate_abd_come_from_the_constitutive_model():
    for name, module, names in _imports():
        if module.split(".")[0] == "laminate_constitutive_model":
            assert module in ALLOWED, (name, module)
            assert names <= ALLOWED[module], (name, module, names - ALLOWED[module])


def test_no_geometry_kernel_shell_fe_or_old_package_imports():
    for name, module, _ in _imports():
        assert module.split(".")[0] not in FORBIDDEN, (name, module)
