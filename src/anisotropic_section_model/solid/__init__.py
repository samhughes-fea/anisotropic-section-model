"""2D finite-element analysis of beam cross-sections (thick, solid and laminated sections)."""

from pre_processing.section_integration.solid.material import SolidMaterial, bond_matrix, material_axes
from pre_processing.section_integration.solid.mesh import SectionMesh, parse_section_mesh_2d, read_gmsh

__all__ = ["SolidMaterial", "bond_matrix", "material_axes", "SectionMesh", "parse_section_mesh_2d", "read_gmsh"]
