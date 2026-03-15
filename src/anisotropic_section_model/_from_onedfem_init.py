# pre_processing/section_integration/__init__.py
"""Section integration: geometry to A, I_y, I_z, J_t, kappa, shear centre (y_sc, z_sc), optional Gamma."""

from pre_processing.section_integration.closed_form import (
    rectangle_properties,
    i_section_properties,
    channel_properties,
)
from pre_processing.section_integration.geometry_to_properties import (
    section_properties_from_geometry,
)

__all__ = [
    "rectangle_properties",
    "i_section_properties",
    "channel_properties",
    "section_properties_from_geometry",
]
