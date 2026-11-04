"""2D finite-element analysis of beam cross-sections (thick, solid and laminated sections)."""

from onedfem.pre_processing.section_integration.solid.material import SolidMaterial, bond_matrix, material_axes
from onedfem.pre_processing.section_integration.solid.mesh import SectionMesh, parse_section_mesh_2d, read_gmsh
from onedfem.pre_processing.section_integration.solid.section_analysis import SolidSectionAnalysis

__all__ = ["SolidMaterial", "bond_matrix", "material_axes", "SectionMesh", "parse_section_mesh_2d", "read_gmsh",
           "SolidSectionAnalysis"]
