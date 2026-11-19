"""Exact ply-angle and ply-thickness derivatives of 2D-meshed sections (``solid.layup_sensitivity``).

The derivatives are checked against central differences of the section rebuilt with the
changed ply (conforming meshes), or with its nodes moved along the mesh velocity and its
ties re-made at the moved nodes (meshes whose junction nodes would otherwise change
topology), and against the thin-walled derivatives and closed forms.
"""

from __future__ import annotations

import numpy as np
import pytest

from anisotropic_section_model.section_analysis import SectionAnalysis
from anisotropic_section_model.solid.elements import shape
from anisotropic_section_model.solid.layup_sensitivity import solid_section_derivative
from anisotropic_section_model.solid.mesh import SectionMesh, _closest_natural, _merge_nodes
from anisotropic_section_model.solid.section_analysis import SolidSectionAnalysis
from anisotropic_section_model.thin_walled import ThinWalledSection, Wall, WallLaminate
from anisotropic_section_model.thin_walled.laminate import Ply, PlyMaterial
from anisotropic_section_model.thin_walled.layup_sensitivity import (LayupParameter, element_section_derivative,
                                                                     section_derivative)
from anisotropic_section_model.thin_walled.section_analysis import ThinWalledSectionAnalysis

CFRP = PlyMaterial(E1=135e9, E2=10e9, G12=5e9, nu12=0.3, G13=5e9, G23=3.8e9, rho=1550.0)


def _box(two_laminates=False):
    """A 200 x 100 mm box; with ``two_laminates`` its webs are thicker, so its corners are butt joints."""
    pts = {1: (-0.1, -0.05), 2: (0.1, -0.05), 3: (0.1, 0.05), 4: (-0.1, 0.05)}
    web = 2 if two_laminates else 1
    walls = [Wall(1, 1, 2, 1), Wall(2, 2, 3, web), Wall(3, 3, 4, 1), Wall(4, 4, 1, web)]
    lams = {1: WallLaminate([Ply(CFRP, a, 1e-3) for a in (30, -30, 0, 30)]),
            2: WallLaminate([Ply(CFRP, a, 1.5e-3) for a in (45, -45, 0)])}
    return ThinWalledSection(pts, walls), lams


def _two_cell():
    """A box with a middle web: four mitred corners and two T-junctions."""
    pts = {1: (-0.2, -0.05), 2: (0.0, -0.05), 3: (0.2, -0.05), 4: (0.2, 0.05), 5: (0.0, 0.05), 6: (-0.2, 0.05)}
    walls = [Wall(1, 1, 2, 1), Wall(2, 2, 3, 1), Wall(3, 3, 4, 1), Wall(4, 4, 5, 1), Wall(5, 5, 6, 1),
             Wall(6, 6, 1, 1), Wall(7, 2, 5, 2)]
    lams = {1: WallLaminate([Ply(CFRP, a, 1e-3) for a in (45, -45, -45, 45)]),
            2: WallLaminate([Ply(CFRP, a, 1e-3) for a in (0, 90, 90, 0)])}
    return ThinWalledSection(pts, walls), lams


def _changed(lams, lid, ply, d_angle=0.0, d_thickness=0.0):
    out = dict(lams)
    out[lid] = WallLaminate([Ply(p.material, p.angle_deg + (d_angle if j == ply else 0.0),
                                 p.thickness + (d_thickness if j == ply else 0.0))
                             for j, p in enumerate(lams[lid].plies)])
    return out


def _natural(X: np.ndarray, p: np.ndarray, start) -> np.ndarray:
    """Natural coordinates of ``p`` in the ``Q8`` with nodes ``X``, unclamped: a node tied on an
    element's edge stays tied to that element as the edge moves past it."""
    xi = np.array(start, dtype=float)
    for _ in range(30):
        N, dN = shape("Q8", *xi)
        step = np.linalg.solve((dN @ X).T, p - N @ X)
        xi += step
        if np.linalg.norm(step) < 1e-14:
            break
    return xi


def _moved(m: SectionMesh, V: np.ndarray, h: float) -> SectionMesh:
    """``m`` with its nodes moved by ``h V`` and its ties re-made there, on the same masters."""
    X = m.nodes + h * V
    ties = {}
    for n, e, xi0, eta0 in m.tie_points:
        inside = np.linalg.norm(shape("Q8", xi0, eta0)[0] @ m.nodes[m.elements[e]] - m.nodes[n]) < 1e-12
        xi, eta = _natural(X[m.elements[e]], X[n], (xi0, eta0)) if inside else \
            _closest_natural(X[m.elements[e]], X[n])[:2]
        N, _ = shape("Q8", xi, eta)
        row = {}
        for a, c in zip(m.elements[e], N):
            for mm, cc in ties.get(int(a), [(int(a), 1.0)]):
                row[mm] = row.get(mm, 0.0) + c * cc
        ties[n] = [(a, c) for a, c in row.items() if abs(c) > 1e-14]
    return SectionMesh(X, m.elements, m.kinds, m.material_ids, m.theta, m.alpha, m.materials, m.section_id,
                       ties, m.plies)


def _values(an):
    mp = an.mass_properties
    return {"stiffness": an.stiffness, "benscoter": an.benscoter_stiffness, "inertia": np.array([an.warping_inertia]),
            "geometric": an.geometric_terms,
            "mass": np.array([mp["m"], mp["m"] * mp["y_m"], mp["m"] * mp["z_m"], mp["I_yy"], mp["I_zz"], mp["I_yz"]])}


def _exact(d):
    m = d.mass
    return {"stiffness": d.stiffness, "benscoter": d.benscoter_stiffness, "inertia": np.array([d.warping_inertia]),
            "geometric": d.geometric_terms, "mass": np.array([m[k] for k in ("m", "my", "mz", "yy", "zz", "yz")])}


def _check(exact, plus, minus, h, rtol):
    for key, ex in exact.items():
        fd = (plus[key] - minus[key]) / (2.0 * h)
        scale = max(np.abs(fd).max(), 1e-300)
        assert np.abs(ex - fd).max() <= rtol * scale, f"{key}: {np.abs(ex - fd).max() / scale:.2e}"


# ---------------------------------------------------------------------- mesh velocities
@pytest.mark.parametrize("make, lid, ply", [(_box, 1, 0), (_box, 1, 3), (lambda: _box(True), 2, 1)])
def test_node_velocities_are_the_derivative_of_the_mesh(make, lid, ply):
    """The nodes are affine in the ply thicknesses: the velocity is exact."""
    sec, lams = make()
    m = SectionMesh.from_thin_walled(sec, lams, elements_along=3)
    h = 1e-4
    mp = SectionMesh.from_thin_walled(sec, _changed(lams, lid, ply, d_thickness=h), elements_along=3)
    np.testing.assert_allclose((mp.nodes - m.nodes) / h, m.velocities[(lid, ply)], atol=1e-9)


def test_merge_joins_nodes_on_either_side_of_a_rounding_boundary():
    """Coincident nodes merge by distance: a pair that straddles a rounding grid line used to stay apart."""
    tol = 1e-10
    x = 0.5 + 0.5 * tol                                   # x / tol is half-way between two integers
    nodes = np.array([[0.0, 0.0], [1.0, 1.0], [x - 2e-16, 0.0], [x + 2e-16, 0.0]])
    assert np.round(nodes[2, 0] / tol) != np.round(nodes[3, 0] / tol)
    new, _, rank = _merge_nodes(nodes, [np.arange(4)], tol=tol, return_map=True)
    assert rank[2] == rank[3] and new.shape[0] == 3


# ---------------------------------------------------------------------- derivatives
def test_ply_angle_derivatives_match_finite_differences():
    sec, lams = _box(True)
    an = SolidSectionAnalysis(SectionMesh.from_thin_walled(sec, lams, elements_along=3))
    d = solid_section_derivative(an, LayupParameter(2, 1, "angle"), benscoter=True, geometric=True)
    h = 1e-2
    plus, minus = (_values(SolidSectionAnalysis(SectionMesh.from_thin_walled(sec, _changed(lams, 2, 1, d_angle=s),
                                                                               elements_along=3)))
                   for s in (h, -h))
    _check(_exact(d), plus, minus, h, 1e-5)
    assert d.axial_positions is None


def test_ply_thickness_derivatives_match_the_rebuilt_section():
    """A conforming (mitred) box: the section rebuilt with the thicker ply is the moved mesh."""
    sec, lams = _box()
    an = SolidSectionAnalysis(SectionMesh.from_thin_walled(sec, lams, elements_along=3))
    for ply in (0, 2):
        d = solid_section_derivative(an, LayupParameter(1, ply, "thickness"), benscoter=True, geometric=True)
        h = 1e-6
        plus, minus = (_values(SolidSectionAnalysis(SectionMesh.from_thin_walled(
            sec, _changed(lams, 1, ply, d_thickness=s), elements_along=3))) for s in (h, -h))
        _check(_exact(d), plus, minus, h, 1e-5)


@pytest.mark.parametrize("make, lid, ply", [(lambda: _box(True), 2, 1), (_two_cell, 2, 1), (_two_cell, 1, 0)])
def test_ply_thickness_derivatives_through_tied_junctions(make, lid, ply):
    """Butt-jointed corners and T-junctions: the tie weights move with the nodes."""
    sec, lams = make()
    m = SectionMesh.from_thin_walled(sec, lams, elements_along=3)
    assert m.ties
    an = SolidSectionAnalysis(m)
    d = solid_section_derivative(an, LayupParameter(lid, ply, "thickness"), benscoter=True, geometric=True)
    h = 1e-6
    V = m.velocities[(lid, ply)]
    plus, minus = (_values(SolidSectionAnalysis(_moved(m, V, s))) for s in (h, -h))
    _check(_exact(d), plus, minus, h, 2e-5)


def test_solid_and_thin_walled_thickness_derivatives_agree():
    """The two models hold the mid-line and centre the stack on it, so they agree to the
    thin-wall approximation (here within 5 % on a symmetric laminate; an unsymmetric one's
    shear and torsion rates differ more, through the walls' bending-extension coupling,
    which the two models treat differently)."""
    sec, _ = _box()
    lams = {1: WallLaminate([Ply(CFRP, a, 1e-3) for a in (30, -30, -30, 30)])}
    an = SolidSectionAnalysis(SectionMesh.from_thin_walled(sec, lams, elements_along=8))
    tw = ThinWalledSectionAnalysis(sec, lams)
    for ply in range(4):
        p = LayupParameter(1, ply, "thickness")
        ds, dt = solid_section_derivative(an, p).stiffness, section_derivative(tw, p).stiffness
        for i, j in [(0, 0), (1, 1), (2, 2), (5, 5)]:
            assert ds[i, j] == pytest.approx(dt[i, j], rel=0.05), (ply, i, j)


def test_isotropic_box_axial_stiffness_rate_is_its_perimeter():
    """``EA = E P t`` for a mitred isotropic box (mid-line held), so ``d EA / d t = E P``; the
    mass rate is ``rho P``."""
    E, rho = 70e9, 2700.0
    sec = ThinWalledSection({1: (-0.1, -0.05), 2: (0.1, -0.05), 3: (0.1, 0.05), 4: (-0.1, 0.05)},
                            [Wall(1, 1, 2, 1), Wall(2, 2, 3, 1), Wall(3, 3, 4, 1), Wall(4, 4, 1, 1)])
    lams = {1: WallLaminate.isotropic(E, 0.0, 2e-3, rho)}
    an = SolidSectionAnalysis(SectionMesh.from_thin_walled(sec, lams, elements_along=4))
    d = solid_section_derivative(an, LayupParameter(1, 0, "thickness"))
    P = 2 * (0.2 + 0.1)
    # with nu = 0 the axial stiffness of the 2D section is exactly E A
    assert d.stiffness[0, 0] == pytest.approx(E * P, rel=1e-9)
    assert d.mass["m"] == pytest.approx(rho * P, rel=1e-12)


def test_a_ply_the_section_does_not_use_gives_zeros():
    sec, lams = _box()
    an = SolidSectionAnalysis(SectionMesh.from_thin_walled(sec, lams, elements_along=3))
    d = solid_section_derivative(an, LayupParameter(7, 0, "thickness"), benscoter=True)
    assert not np.any(d.stiffness) and not np.any(d.benscoter_stiffness) and d.mass["m"] == 0.0


def test_thickness_needs_node_velocities():
    m = SectionMesh.rectangle(0.1, 0.02, 4, 2, SectionMesh.from_thin_walled(*_box(), elements_along=2).materials[1])
    m.plies = [(1, 0)] * len(m.elements)
    with pytest.raises(ValueError, match="velocities"):
        solid_section_derivative(SolidSectionAnalysis(m), LayupParameter(1, 0, "thickness"))


def test_element_derivative_carries_the_moving_axial_points():
    """``with_pretwist`` of the derivative section is the derivative of ``with_pretwist``:
    the axial points' motion is carried by a five-point stencil, exact for its quartics."""
    sec, lams = _box()
    k = 0.3                                                 # pretwist rate, rad/m
    p = LayupParameter(1, 0, "thickness")

    def element(ls):
        return SectionAnalysis.from_solid(SolidSectionAnalysis(SectionMesh.from_thin_walled(sec, ls, elements_along=3)))

    sa = element(lams)
    an = SolidSectionAnalysis(SectionMesh.from_thin_walled(sec, lams, elements_along=3))
    dsa = element_section_derivative(sa, solid_section_derivative(an, p))
    exact = dsa.with_pretwist(k).stiffness - dsa.stiffness
    h = 1e-6
    fd = [(element(_changed(lams, 1, 0, d_thickness=s)).with_pretwist(k).stiffness
           - element(_changed(lams, 1, 0, d_thickness=s)).stiffness) for s in (h, -h)]
    fd = (fd[0] - fd[1]) / (2 * h)
    np.testing.assert_allclose(exact, fd, rtol=1e-5, atol=1e-6 * np.abs(fd).max())
