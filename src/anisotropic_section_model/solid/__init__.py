"""2D finite-element analysis of beam cross-sections (thick, solid and laminated sections)."""

from anisotropic_section_model.solid.material import SolidMaterial, bond_matrix, material_axes
from anisotropic_section_model.solid.mesh import SectionMesh, parse_section_mesh_2d, read_gmsh
from anisotropic_section_model.solid.section_analysis import SolidSectionAnalysis

__all__ = ["SolidMaterial", "bond_matrix", "material_axes", "SectionMesh", "parse_section_mesh_2d", "read_gmsh",
           "SolidSectionAnalysis"]
