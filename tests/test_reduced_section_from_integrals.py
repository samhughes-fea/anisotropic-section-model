"""Geometric integrals become a reduced beam section, with no wall graph."""

import math

from anisotropic_section_model.geometry_to_properties import reduced_section_from_integrals


def _rotated_product(ixx, iyy, ixy, theta):
    c, s = math.cos(theta), math.sin(theta)
    ixy2 = s * c * (ixx - iyy) + (c * c - s * s) * ixy
    ixx2 = c * c * ixx + s * s * iyy - 2.0 * s * c * ixy
    return ixx2, ixy2


def test_centres_follow_the_centroid_and_warping_is_zero():
    section = reduced_section_from_integrals({
        "area": 0.02,
        "centroid": (0.10, -0.04),
        "ixx": 4.0e-4,
        "iyy": 1.0e-4,
        "ixy": 0.0,
    }, density=1500.0)
    assert section.elastic_centre == (0.10, -0.04)
    assert section.shear_centre == section.elastic_centre
    assert section.mass_centre == section.centroid
    assert section.mass == 1500.0 * 0.02
    assert section.warping_constant == 0.0
    assert section.torsion_constant == 0.0
    assert section.principal_angle == 0.0
    assert math.isclose(section.principal_moments[0], 4.0e-4)
    assert math.isclose(section.principal_moments[1], 1.0e-4)


def test_a_product_moment_sets_the_principal_axes():
    ixx = iyy = 2.0
    ixy = 1.0
    section = reduced_section_from_integrals({
        "area": 1.0, "centroid_x": 0.0, "centroid_y": 0.0,
        "ixx": ixx, "iyy": iyy, "ixy": ixy,
    })
    assert math.isclose(section.principal_angle, -math.pi / 4.0)
    assert math.isclose(section.principal_moments[0], 3.0)
    assert math.isclose(section.principal_moments[1], 1.0)
    ixx2, ixy2 = _rotated_product(ixx, iyy, ixy, section.principal_angle)
    assert abs(ixy2) < 1e-12
    assert abs(ixx2 - section.principal_moments[0]) < 1e-12


def test_origin_moments_keep_a_cap_offset():
    # A strip of area 0.02 whose mid-line sits 0.05 m off the axis. The
    # centroidal second moment is the strip's own; the offset is the centroid.
    area = 0.02
    e = 0.05
    section = reduced_section_from_integrals({
        "area": area,
        "centroid": (0.0, e),
        "ixx_origin": area * e * e,
        "iyy_origin": 0.02 / 12.0,
        "ixy_origin": 0.0,
    })
    assert section.centroid == (0.0, e)
    assert abs(section.ixx) < 1e-15
    assert section.elastic_centre[1] == e
    assert section.shear_centre[1] == e
