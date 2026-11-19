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
from anisotropic_section_model.solid.layup_sensitivity import solid_section_derivative
from anisotropic_section_model.solid.taper import solid_section_taper, solid_taper_derivative

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
    "solid_section_derivative",
    "solid_section_taper",
    "solid_taper_derivative",
]
