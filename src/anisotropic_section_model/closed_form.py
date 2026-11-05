# pre_processing/section_integration/closed_form.py
"""Closed-form section properties: rectangle, I-section, channel (A, I_y, I_z, J_t, kappa, y_sc, z_sc)."""

from typing import Dict, Any


def rectangle_properties(
    width: float,
    height: float,
) -> Dict[str, float]:
    """
    Section properties for a rectangle (centroid at origin, width along y, height along z).

    Parameters
    ----------
    width : float
        Width (y-direction) [m].
    height : float
        Height (z-direction) [m].

    Returns
    -------
    dict
        A, I_x, I_y, I_z, J_t, kappa, y_sc, z_sc. I_x = 0 (no polar in 2D); y_sc = z_sc = 0.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    b, h = width, height
    A = b * h
    # I_y = second moment about y (z^2 integrated): I_y = b*h^3/12
    I_y = (b * h**3) / 12.0
    # I_z = second moment about z (y^2 integrated): I_z = h*b^3/12
    I_z = (h * b**3) / 12.0
    # Torsion constant (rectangle): J_t ≈ (1/3) * min(b,h) * max(b,h)^3 * (1 - 0.63*min/max)
    small, large = min(b, h), max(b, h)
    ratio = small / large
    J_t = (1.0 / 3.0) * small * large**3 * (1.0 - 0.63 * ratio * (1.0 - ratio**4 / 12.0))
    kappa = 5.0 / 6.0  # rectangular section
    y_sc = 0.0
    z_sc = 0.0
    return {
        "A": A,
        "I_x": 0.0,
        "I_y": I_y,
        "I_z": I_z,
        "J_t": J_t,
        "kappa": kappa,
        "y_sc": y_sc,
        "z_sc": z_sc,
    }


def i_section_properties(
    width_flange: float,
    height_web: float,
    thickness_flange: float,
    thickness_web: float,
) -> Dict[str, float]:
    """
    Section properties for a doubly symmetric I-section (centroid at origin).

    Parameters
    ----------
    width_flange : float
        Total flange width (y-direction) [m].
    height_web : float
        Web height between inner flange faces [m].
    thickness_flange : float
        Flange thickness [m].
    thickness_web : float
        Web thickness [m].

    Returns
    -------
    dict
        A, I_x, I_y, I_z, J_t, kappa, y_sc, z_sc. Shear centre at centroid (0, 0).
    """
    if any(x <= 0 for x in (width_flange, height_web, thickness_flange, thickness_web)):
        raise ValueError("I-section dimensions must be positive")
    b, h_w, t_f, t_w = width_flange, height_web, thickness_flange, thickness_web
    h_total = h_w + 2 * t_f
    A = 2 * b * t_f + h_w * t_w
    # I_z (about z, bending in x-y): I_z = (b*h_total^3 - (b-t_w)*h_w^3)/12
    I_z = (b * h_total**3 - (b - t_w) * h_w**3) / 12.0
    # I_y (about y, bending in x-z): I_y = 2*t_f*b^3/12 + h_w*t_w^3/12
    I_y = (2 * t_f * b**3 + h_w * t_w**3) / 12.0
    # Torsion constant (thin-walled I): J_t ≈ (2*b*t_f^3 + h_w*t_w^3)/3
    J_t = (2 * b * t_f**3 + h_w * t_w**3) / 3.0
    kappa = 5.0 / 6.0  # typical for I
    y_sc = 0.0
    z_sc = 0.0
    return {
        "A": A,
        "I_x": 0.0,
        "I_y": I_y,
        "I_z": I_z,
        "J_t": J_t,
        "kappa": kappa,
        "y_sc": y_sc,
        "z_sc": z_sc,
    }


def channel_properties(
    width_flange: float,
    height_web: float,
    thickness_flange: float,
    thickness_web: float,
) -> Dict[str, float]:
    """
    Section properties for a channel (open thin-walled): U-section with flanges at top and bottom.

    Centroid is computed; shear centre is offset from centroid along the y-axis (perpendicular
    to the web). Convention: web in z-direction, flanges in y-direction; centroid at (y_c, 0);
    shear centre at (y_c + y_sc_offset, 0) with y_sc_offset positive towards the web.

    Parameters
    ----------
    width_flange : float
        Flange width (y-direction from web to tip) [m].
    height_web : float
        Web height (z-direction) [m].
    thickness_flange : float
        Flange thickness [m].
    thickness_web : float
        Web thickness [m].

    Returns
    -------
    dict
        A, I_x, I_y, I_z, J_t, kappa, y_sc, z_sc. y_sc is the shear centre offset from centroid.
    """
    if any(x <= 0 for x in (width_flange, height_web, thickness_flange, thickness_web)):
        raise ValueError("Channel dimensions must be positive")
    b_f, h_w, t_f, t_w = width_flange, height_web, thickness_flange, thickness_web
    # Area: web + two flanges
    A = h_w * t_w + 2 * b_f * t_f
    # Centroid y (from web): y_c = (2 * b_f * t_f * (b_f/2)) / A = b_f^2 * t_f / A
    y_c = (b_f * b_f * t_f) / A
    # I_z (about centroidal z-axis): web + two flanges with parallel axis
    I_z_web = t_w * h_w**3 / 12.0
    # Each flange: I_own + A*d^2, flange centre at y = b_f/2, distance to centroid = (b_f/2 - y_c)
    d_flange = (b_f / 2.0) - y_c
    I_z_flanges = 2.0 * ((b_f * t_f**3) / 12.0 + (b_f * t_f) * d_flange**2)
    I_z = I_z_web + I_z_flanges
    # I_y (about centroid y-axis)
    I_y = (2 * t_f * b_f**3) / 12.0 + (h_w * t_w**3) / 12.0
    # Torsion constant (thin-walled open): J_t = (1/3)*sum(b*t^3)
    J_t = (1.0 / 3.0) * (h_w * t_w**3 + 2 * b_f * t_f**3)
    # Shear centre offset from centroid (thin-walled channel): e = 3*b_f^2*t_f/(h_w*t_w + 6*b_f*t_f)
    # Shear centre lies between centroid and web; with centroid at y_c, shear centre at y_c - e (towards web)
    e = (3 * b_f * b_f * t_f) / (h_w * t_w + 6 * b_f * t_f)
    # y_sc = offset of shear centre from centroid (positive = towards web if centroid is towards flanges)
    y_sc = -e  # centroid is at +y_c from web; shear centre at y_c - e, so offset = -e
    z_sc = 0.0
    kappa = 5.0 / 6.0
    return {
        "A": A,
        "I_x": 0.0,
        "I_y": I_y,
        "I_z": I_z,
        "J_t": J_t,
        "kappa": kappa,
        "y_sc": y_sc,
        "z_sc": z_sc,
    }
