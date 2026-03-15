# pre_processing/section_integration/geometry_to_properties.py
"""Map section geometry to properties (A, I_y, I_z, J_t, kappa, y_sc, z_sc, optional Gamma)."""

from typing import Dict, Any, Optional

from pre_processing.section_integration.closed_form import (
    rectangle_properties,
    i_section_properties,
    channel_properties,
)


def section_properties_from_geometry(
    section_type: str,
    **kwargs: float,
) -> Dict[str, float]:
    """
    Compute section properties from geometry (closed-form).

    Parameters
    ----------
    section_type : str
        One of "rectangle", "i_section", "channel".
    **kwargs : float
        Dimension arguments (e.g. width, height for rectangle).

    Returns
    -------
    dict
        A, I_x, I_y, I_z, J_t, kappa, y_sc, z_sc. May include Gamma for future use.
    """
    section_type_lower = section_type.strip().lower()
    if section_type_lower == "rectangle":
        return rectangle_properties(
            width=kwargs["width"],
            height=kwargs["height"],
        )
    if section_type_lower == "i_section":
        return i_section_properties(
            width_flange=kwargs["width_flange"],
            height_web=kwargs["height_web"],
            thickness_flange=kwargs["thickness_flange"],
            thickness_web=kwargs["thickness_web"],
        )
    if section_type_lower == "channel":
        return channel_properties(
            width_flange=kwargs["width_flange"],
            height_web=kwargs["height_web"],
            thickness_flange=kwargs["thickness_flange"],
            thickness_web=kwargs["thickness_web"],
        )
    raise ValueError(
        f"Unknown section_type {section_type!r}; use 'rectangle', 'i_section', or 'channel'."
    )
