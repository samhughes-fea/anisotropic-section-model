"""2D finite-element analysis of beam cross-sections (thick, solid and laminated sections)."""

from anisotropic_section_model.solid.compare import (
    ABSENT,
    StationComparison,
    TermComparison,
    compare_analyses,
    compare_cut,
    compare_thick_stations,
    named_stiffness,
    published_from_record,
)
from anisotropic_section_model.solid.material import SolidMaterial, bond_matrix, material_axes
from anisotropic_section_model.solid.mesh import SectionMesh, parse_section_mesh_2d, read_gmsh
from anisotropic_section_model.solid.section_analysis import SolidSectionAnalysis

__all__ = [
    "ABSENT",
    "SolidMaterial",
    "StationComparison",
    "TermComparison",
    "bond_matrix",
    "compare_analyses",
    "compare_cut",
    "compare_thick_stations",
    "material_axes",
    "named_stiffness",
    "parse_section_mesh_2d",
    "published_from_record",
    "read_gmsh",
    "SectionMesh",
    "SolidSectionAnalysis",
]
