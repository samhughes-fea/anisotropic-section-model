"""The package boundary of the split:

    laminate_constitutive_model  <-  anisotropic_section_model

Walls are laminates, so the names this package takes from ``laminate_constitutive_model``
are ``Laminate`` and ``integrate_abd``, and for ply failure the strengths and the array
exposures. LCM laminates and materials otherwise plug in by duck typing
(``WallLaminate.from_laminate``, ``SolidMaterial.from_material_card``). It does not import
the geometry kernel (``ThinWalledSection.from_geometry`` reads a cut by duck typing) or
the shell FE (``shell_fem``), which it meets only in tests and examples.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "anisotropic_section_model"
ALLOWED = {
    "laminate_constitutive_model.laminate_mechanics": {"Laminate"},
    "laminate_constitutive_model.laminate_kinematics": {"integrate_abd"},
    "laminate_constitutive_model.laminate_materials_3d": {"StrengthProperties3D"},
    "laminate_constitutive_model.laminate_ply_exposure": {
        "PuckParameters", "hashin_exposure", "max_stress_exposure", "ply_exposures", "puck_exposure",
        "tsai_wu_exposure",
    },
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


def test_only_the_allowed_names_come_from_the_constitutive_model():
    for name, module, names in _imports():
        if module.split(".")[0] == "laminate_constitutive_model":
            assert module in ALLOWED, (name, module)
            assert names <= ALLOWED[module], (name, module, names - ALLOWED[module])


def test_no_geometry_kernel_shell_fe_or_old_package_imports():
    for name, module, _ in _imports():
        assert module.split(".")[0] not in FORBIDDEN, (name, module)
