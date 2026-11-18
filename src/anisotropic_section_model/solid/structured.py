# anisotropic_section_model/solid/structured.py
"""Structured 2D section meshes from mapped patches, in ``Q4``, ``Q8`` or ``T6`` elements.

A patch is a map ``(u, v) in [0, 1]^2 -> (y, z)`` with ``u`` along a wall and ``v`` across
it (toward the wall normal ``n = x cross s``, so the map keeps the elements counter-clockwise).
Each patch is cut into ``n_u x n_v`` cells; a ``T6`` mesh splits every cell into two
triangles. Patches that share an edge share its nodes, so a closed ring or a section of
several walls is one conforming mesh.

Every element gets a material id, a ply angle ``theta`` and a wall angle ``alpha`` from the
patch's ``data(u, v)`` at the cell centre. ``alpha`` defaults to the direction of increasing
``u`` there, so a curved wall's plies follow it.

Public API
----------
Patch, structured_mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from anisotropic_section_model.solid.material import SolidMaterial
from anisotropic_section_model.solid.mesh import SectionMesh, _merge_nodes

__all__ = ["Patch", "structured_mesh"]

#: ``data(u, v) -> (material_id, theta)`` or ``(material_id, theta, alpha)``, degrees
PatchData = Callable[[float, float], Tuple]


@dataclass(frozen=True)
class Patch:
    """A mapped region of the section: ``at(u, v) -> (y, z)`` on ``[0, 1]^2``, ``n_u x n_v`` cells."""

    at: Callable[[float, float], Tuple[float, float]]
    n_u: int
    n_v: int
    data: PatchData


def _alpha(at, u: float, v: float, h: float = 1e-6) -> float:
    a = np.asarray(at(min(u + h, 1.0), v), dtype=float)
    b = np.asarray(at(max(u - h, 0.0), v), dtype=float)
    d = a - b
    return float(np.degrees(np.arctan2(d[1], d[0])))


def _cells(patch: Patch, kind: str):
    """Nodes and elements (local node ids) of one patch, with the cell centre ``(u, v)`` per element."""
    k = 1 if kind == "Q4" else 2
    gu, gv = k * patch.n_u + 1, k * patch.n_v + 1
    index = -np.ones((gu, gv), dtype=np.int64)
    pts: List[Tuple[float, float]] = []
    for i in range(gu):
        for j in range(gv):
            if kind == "Q8" and i % 2 == 1 and j % 2 == 1:
                continue
            index[i, j] = len(pts)
            pts.append(tuple(patch.at(i / (gu - 1), j / (gv - 1))))
    elems: List[np.ndarray] = []
    kinds: List[str] = []
    centres: List[Tuple[float, float]] = []
    for i in range(patch.n_u):
        for j in range(patch.n_v):
            uc, vc = (i + 0.5) / patch.n_u, (j + 0.5) / patch.n_v
            I, J = k * i, k * j
            if kind == "Q4":
                elems.append(index[[I, I + 1, I + 1, I], [J, J, J + 1, J + 1]])
                kinds.append("Q4")
                centres.append((uc, vc))
            elif kind == "Q8":
                elems.append(np.array([index[I, J], index[I + 2, J], index[I + 2, J + 2], index[I, J + 2],
                                       index[I + 1, J], index[I + 2, J + 1], index[I + 1, J + 2], index[I, J + 1]]))
                kinds.append("Q8")
                centres.append((uc, vc))
            elif kind == "T6":
                elems.append(np.array([index[I, J], index[I + 2, J], index[I + 2, J + 2],
                                       index[I + 1, J], index[I + 2, J + 1], index[I + 1, J + 1]]))
                elems.append(np.array([index[I, J], index[I + 2, J + 2], index[I, J + 2],
                                       index[I + 1, J + 1], index[I + 1, J + 2], index[I, J + 1]]))
                kinds += ["T6", "T6"]
                centres += [(uc, vc), (uc, vc)]
            else:
                raise ValueError(f"unknown element kind {kind!r}; use Q4, Q8 or T6")
    return np.array(pts, dtype=np.float64), elems, kinds, centres


def structured_mesh(patches: Sequence[Patch], materials: Dict[int, SolidMaterial], kind: str = "Q8",
                    tol: float = 1e-9) -> SectionMesh:
    """One conforming ``SectionMesh`` of ``patches`` (coincident nodes merged, relative ``tol``)."""
    nodes: List[np.ndarray] = []
    elements: List[np.ndarray] = []
    kinds: List[str] = []
    mids: List[int] = []
    theta: List[float] = []
    alpha: List[float] = []
    offset = 0
    for patch in patches:
        pts, elems, ks, centres = _cells(patch, kind)
        for conn, (uc, vc) in zip(elems, centres):
            d = tuple(patch.data(uc, vc))
            mids.append(int(d[0]))
            theta.append(float(d[1]))
            alpha.append(float(d[2]) if len(d) > 2 else _alpha(patch.at, uc, vc))
            elements.append(conn + offset)
        kinds += ks
        nodes.append(pts)
        offset += pts.shape[0]
    all_nodes, merged = _merge_nodes(np.vstack(nodes), elements, tol=tol)
    return SectionMesh(all_nodes, merged, kinds, mids, theta, alpha, dict(materials))


def rectangle_patch(y0: float, z0: float, b: float, h: float, n_u: int, n_v: int, data: PatchData) -> Patch:
    """``b x h`` rectangle with its lower-left corner at ``(y0, z0)``; ``u`` along ``y``."""
    return Patch(lambda u, v: (y0 + u * b, z0 + v * h), n_u, n_v, data)


def arc_patch(centre: Tuple[float, float], r_in: float, r_out: float, phi0: float, phi1: float,
              n_u: int, n_v: int, data: PatchData) -> Patch:
    """Annular sector from polar angle ``phi0`` to ``phi1`` (radians) and radius ``r_in`` to
    ``r_out``. ``u`` runs from ``phi0`` to ``phi1``; ``v`` from ``r_in`` to ``r_out`` when
    ``phi1 > phi0`` (counter-clockwise) and from ``r_out`` to ``r_in`` otherwise, so the
    elements stay counter-clockwise. Nodes lie on the true arcs (the mesh is isoparametric
    only to the element order)."""
    cy, cz = centre
    ccw = phi1 > phi0

    def at(u, v):
        phi = phi0 + u * (phi1 - phi0)
        r = (r_out - v * (r_out - r_in)) if ccw else (r_in + v * (r_out - r_in))
        return cy + r * np.cos(phi), cz + r * np.sin(phi)

    return Patch(at, n_u, n_v, data)
