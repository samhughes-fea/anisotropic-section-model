# anisotropic_section_model/solid/taper.py
"""Taper coupling of a section meshed in 2D: the Resal effect, as ``thin_walled.taper``.

A point of the section at ``r(x) = (y(x), z(x))`` lies on a material fibre inclined at
``r' = dr/dx`` to the beam axis, so a rigid translation and rotation of the section
stretches the fibre by ``y' gamma_xy + z' gamma_xz`` as well. With the axial modulus
``E_x`` (``SolidSectionAnalysis._axial_modulus``: the in-plane stresses free, the shears
held, the 2D counterpart of the thin-walled wall's axial ``A``) the energy
``1/2 int E_x eps_fibre^2 dA`` adds::

    C[a, g] = int E_x f_a r'_g dA,     f = (1, z, -y)
    Q[g, h] = int E_x r'_g r'_h dA

to the section stiffness, as ``thin_walled.taper.SectionTaper`` (the same 6 x 6 layout).
``solid_section_taper`` takes two meshes of the same topology a distance apart (each
node the same material point at the two positions): the cuts either side of an element's
mid-point, or ``SectionMesh.from_thin_walled`` of the two thin-walled sections.
The section between them is the mesh at the mean of their nodes, and ``r'`` interpolates
the nodes' rates ``(X_end - X_start) / distance``.

``solid_taper_derivative`` differentiates the coupling for a ply angle (``E_x``) or a ply
thickness (the nodes of both meshes moving at their ``SectionMesh.velocities``, so the
mean section and the rates move with them), exactly.

Public API
----------
solid_section_taper, solid_taper_derivative.
"""

from __future__ import annotations

import numpy as np

from anisotropic_section_model.solid.mesh import SectionMesh
from anisotropic_section_model.thin_walled.taper import SectionTaper

__all__ = ["solid_section_taper", "solid_taper_derivative"]


def _middle(start: SectionMesh, end: SectionMesh) -> SectionMesh:
    if start.nodes.shape != end.nodes.shape or len(start.elements) != len(end.elements) or any(
            not np.array_equal(a, b) for a, b in zip(start.elements, end.elements)):
        raise ValueError("the two meshes must have the same nodes and elements")
    return SectionMesh(0.5 * (start.nodes + end.nodes), start.elements, start.kinds, start.material_ids,
                       start.theta, start.alpha, start.materials, start.section_id, start.ties, start.plies)


def solid_section_taper(start: SectionMesh, end: SectionMesh, distance: float) -> SectionTaper:
    """Taper coupling (right-handed) of the section midway between two meshes ``distance`` apart.

    ``end`` lies further along ``x``. The result's ``source`` keeps the two meshes for
    ``solid_taper_derivative``.
    """
    from anisotropic_section_model.solid.section_analysis import SolidSectionAnalysis

    if distance <= 0.0:
        raise ValueError("distance must be positive")
    mid = _middle(start, end)
    an = SolidSectionAnalysis(mid)
    rate = (end.nodes - start.nodes) / distance
    Ex = an._axial_modulus
    C = np.zeros((3, 2))
    Q = np.zeros((2, 2))
    for p in an._points:
        rp = p.N @ rate[mid.elements[p.element]]
        f = np.array([1.0, p.z, -p.y])
        wE = p.weight * Ex[p.element]
        C += wE * np.outer(f, rp)
        Q += wE * np.outer(rp, rp)
    return SectionTaper(C, 0.5 * (Q + Q.T), source=(start, end, float(distance)))


def solid_taper_derivative(taper: SectionTaper, param) -> SectionTaper:
    """``d`` of a ``solid_section_taper`` coupling for a ply ``param`` (``LayupParameter``)."""
    from anisotropic_section_model.solid.layup_sensitivity import (_axial_modulus_derivative, _point_rates,
                                                                   material_derivative)
    from anisotropic_section_model.solid.section_analysis import SolidSectionAnalysis

    if taper.source is None:
        raise ValueError("this taper was not made by solid_section_taper (it keeps no meshes)")
    start, end, distance = taper.source
    mid = _middle(start, end)
    if mid.plies is None:
        raise ValueError("the meshes have no ply tags (SectionMesh.plies)")
    an = SolidSectionAnalysis(mid)
    tag = (param.laminate_id, param.ply)
    elems = {e for e, t in enumerate(mid.plies) if t is not None and tuple(t) == tag}
    rate = (end.nodes - start.nodes) / distance
    Ex = an._axial_modulus
    if param.kind == "angle":
        dEx = {e: _axial_modulus_derivative(mid.element_stiffness(e), material_derivative(
            mid.materials[mid.material_ids[e]], mid.theta[e], mid.alpha[e])) for e in elems}
        rates = None
        drate = np.zeros_like(rate)
    else:
        dEx = {}
        vel = []
        for m in (start, end):
            if m.velocities is None:
                raise ValueError("a thickness derivative needs both meshes' SectionMesh.velocities")
            vel.append(np.asarray(m.velocities.get(tag, np.zeros_like(m.nodes)), dtype=float))
        rates = _point_rates(an, {}, 0.5 * (vel[0] + vel[1])) if elems else None
        drate = (vel[1] - vel[0]) / distance
    dC = np.zeros((3, 2))
    dQ = np.zeros((2, 2))
    if not elems:
        return SectionTaper(dC, dQ)
    for i, p in enumerate(an._points):
        conn = mid.elements[p.element]
        rp, drp = p.N @ rate[conn], p.N @ drate[conn]
        f = np.array([1.0, p.z, -p.y])
        wE = p.weight * Ex[p.element]
        dwE = p.weight * dEx.get(p.element, 0.0)
        df = np.zeros(3)
        if rates is not None:
            q = rates[i]
            dwE += q.dw * Ex[p.element]
            df = np.array([0.0, q.dz, -q.dy])
        dC += dwE * np.outer(f, rp) + wE * (np.outer(df, rp) + np.outer(f, drp))
        dQ += dwE * np.outer(rp, rp) + wE * (np.outer(drp, rp) + np.outer(rp, drp))
    return SectionTaper(dC, 0.5 * (dQ + dQ.T))
