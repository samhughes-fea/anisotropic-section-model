# anisotropic_section_model/geometry_to_properties.py
"""Map section geometry to properties (A, I_y, I_z, J_t, kappa, y_sc, z_sc, optional Gamma)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

from anisotropic_section_model.closed_form import (
    rectangle_properties,
    i_section_properties,
    channel_properties,
)

Integrals = Union[Mapping[str, Any], object]


def principal_axes_from_moments(ixx: float, iyy: float, ixy: float) -> tuple[float, tuple[float, float]]:
    """Principal angle and moments of the second-moment tensor ``[[ixx, ixy], [ixy, iyy]]``.

    ``ixx`` is ``∫ y²``, ``iyy`` is ``∫ x²`` (``∫ z²`` in beam section axes) and
    ``ixy`` is the product moment. The angle is from the first axis toward the
    second, in ``(-π/2, π/2]``, and the first principal moment is the larger one.
    """
    mean = 0.5 * (ixx + iyy)
    radius = math.hypot(0.5 * (ixx - iyy), ixy)
    if radius <= 1e-14 * max(abs(mean), 1.0):
        return 0.0, (mean, mean)
    # ixx = ∫ y², iyy = ∫ x²: a rotation from the first axis toward the second
    # zeros the product when tan 2θ = -2 ixy / (ixx - iyy).
    theta = 0.5 * math.atan2(-2.0 * ixy, ixx - iyy)
    c, s = math.cos(theta), math.sin(theta)
    about_x = c * c * ixx + s * s * iyy - 2.0 * s * c * ixy
    if about_x < mean:
        theta += math.pi / 2.0
    theta = (theta + math.pi / 2.0) % math.pi - math.pi / 2.0
    return theta, (mean + radius, mean - radius)


def principal_axes_from_mass(mass: Mapping[str, float]) -> tuple[float, tuple[float, float]]:
    """Principal axes of a section mass dictionary, about the mass centre.

    ``mass`` has ``m``, ``y_m``, ``z_m``, ``I_yy`` (``∫ ρ z²``), ``I_zz``
    (``∫ ρ y²``) and ``I_yz`` (``∫ ρ y z``), all about the origin as in
    ``SectionAnalysis.mass``. The angle is from ``y`` toward ``z``.
    """
    m = float(mass.get("m", 0.0))
    ym = float(mass.get("y_m", 0.0))
    zm = float(mass.get("z_m", 0.0))
    iyy = float(mass.get("I_yy", 0.0)) - m * zm * zm
    izz = float(mass.get("I_zz", 0.0)) - m * ym * ym
    iyz = float(mass.get("I_yz", 0.0)) - m * ym * zm
    return principal_axes_from_moments(izz, iyy, iyz)


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


@dataclass(frozen=True)
class ReducedBeamSection:
    """Homogeneous beam section reduced from geometric integrals.

    The integrals are area, centroid and centroidal second moments. A cap
    offset is already in those numbers. This reduction does not read a wall
    graph.

    For a homogeneous section the elastic centre and, at uniform density, the
    mass centre are the centroid. The shear centre of a section known only by
    those moments is the centroid (exact for a doubly symmetric solid, a
    rectangle and a closed box). The warping constant is zero: there is no
    sectorial area in the integrals. ``torsion_constant`` stays 0 unless the
    caller supplies one; the area moments do not determine Saint-Venant J.
    """

    area: float
    centroid: tuple[float, float]
    ixx: float
    iyy: float
    ixy: float
    elastic_centre: tuple[float, float]
    shear_centre: tuple[float, float]
    mass_centre: tuple[float, float]
    principal_angle: float
    principal_moments: tuple[float, float]
    warping_constant: float
    torsion_constant: float = 0.0
    density: float = 0.0

    @property
    def mass(self) -> float:
        return self.density * self.area


def _value(integrals: Integrals, *names: str) -> Any:
    if isinstance(integrals, Mapping):
        for name in names:
            if name in integrals and integrals[name] is not None:
                return integrals[name]
    else:
        for name in names:
            if hasattr(integrals, name):
                value = getattr(integrals, name)
                if value is not None:
                    return value
    raise TypeError(f"integrals need one of: {', '.join(names)}")


def _optional(integrals: Integrals, *names: str) -> Any:
    try:
        return _value(integrals, *names)
    except TypeError:
        return None


def reduced_section_from_integrals(
    integrals: Integrals,
    *,
    density: float = 0.0,
    torsion_constant: Optional[float] = None,
) -> ReducedBeamSection:
    """Elastic centre, shear centre, principal axes, mass centre and warping.

    ``integrals`` is duck typed: ``area``, ``centroid`` ``(x, y)`` (or
    ``centroid_x`` and ``centroid_y``) and centroidal ``ixx``, ``iyy``,
    ``ixy``. Origin moments ``ixx_origin``, ``iyy_origin``, ``ixy_origin``
    are accepted when the centroidal ones are absent. ``ixx`` is
    ``∫ y² dA`` and ``iyy`` is ``∫ x² dA``, in the same frame as the centroid.
    """
    area = float(_value(integrals, "area"))
    if area < 0.0:
        raise ValueError("area must be non-negative")
    centroid = _optional(integrals, "centroid")
    if centroid is None:
        xc = float(_value(integrals, "centroid_x"))
        yc = float(_value(integrals, "centroid_y"))
    else:
        xc, yc = float(centroid[0]), float(centroid[1])
    ixx = _optional(integrals, "ixx")
    iyy = _optional(integrals, "iyy")
    ixy = _optional(integrals, "ixy")
    if ixx is None:
        ixx = float(_value(integrals, "ixx_origin")) - area * yc * yc
    if iyy is None:
        iyy = float(_value(integrals, "iyy_origin")) - area * xc * xc
    if ixy is None:
        origin = _optional(integrals, "ixy_origin")
        ixy = 0.0 if origin is None else float(origin) - area * xc * yc
    ixx, iyy, ixy = float(ixx), float(iyy), float(ixy)
    centre = (xc, yc)
    theta, moments = principal_axes_from_moments(ixx, iyy, ixy)
    given_j = _optional(integrals, "torsion_constant", "j_t")
    j_t = float(torsion_constant if torsion_constant is not None else (0.0 if given_j is None else given_j))
    return ReducedBeamSection(
        area=area,
        centroid=centre,
        ixx=ixx,
        iyy=iyy,
        ixy=ixy,
        elastic_centre=centre,
        shear_centre=centre,
        mass_centre=centre,
        principal_angle=theta,
        principal_moments=moments,
        warping_constant=0.0,
        torsion_constant=j_t,
        density=float(density),
    )
