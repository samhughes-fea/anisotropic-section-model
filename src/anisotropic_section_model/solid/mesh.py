# pre_processing/section_integration/solid/mesh.py
"""2D finite-element meshes of beam cross-sections, with a material and orientation per element.

A ``SectionMesh`` holds nodes ``(y, z)``, quadratic elements (``T6`` or ``Q8``, see
``elements.py``) and, per element, a material id, a ply angle ``theta`` and a wall angle
``alpha`` (``material.py``: the fibre is at ``theta`` from the beam axis toward the wall
direction, which is at ``alpha`` from ``y`` toward ``z``). The section solver
integrates over it with ``quadrature_points``.

Builders
--------
- ``SectionMesh.rectangle``: a solid rectangle.
- ``SectionMesh.from_thin_walled``: every wall of a thin-walled section as a strip of
  ``Q8`` elements, each ply its own row of elements through the thickness, so the same
  section can be analysed both ways. Each strip is its wall's mid-line length long, so
  where walls meet their strips overlap and the junction counts twice, exactly as in the
  mid-line model.
- ``SectionMesh.from_gmsh``: a Gmsh ``.msh`` file (format 4.1 ASCII, ``T6`` = type 9,
  ``Q8`` = type 16), with physical surfaces mapped to ``(material, theta, alpha)``.
- ``section_mesh_2d.txt`` in a job (``parse_section_mesh_2d``).

Public API
----------
SectionMesh, QuadraturePoint, parse_section_mesh_2d, read_gmsh.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from pre_processing.section_integration.solid.elements import n_nodes, quadrature, shape
from pre_processing.section_integration.solid.material import SolidMaterial


@dataclass(frozen=True)
class QuadraturePoint:
    element: int
    y: float
    z: float
    weight: float                       # includes the Jacobian determinant
    N: np.ndarray                       # (n,) shape functions
    dN: np.ndarray                      # (2, n) derivatives in (y, z)
    nodes: np.ndarray                   # (n,) node indices


@dataclass
class SectionMesh:
    nodes: npt.NDArray[np.float64]                       # (n_nodes, 2)
    elements: List[npt.NDArray[np.int64]]                # node indices per element
    kinds: List[str]                                      # "T6" or "Q8" per element
    material_ids: List[int]
    theta: List[float]                                    # ply angle per element, degrees
    alpha: List[float]                                    # wall angle per element, degrees
    materials: Dict[int, SolidMaterial] = field(default_factory=dict)
    section_id: Optional[int] = None
    #: Tied nodes: ``{node: [(master node, weight), ...]}``, the node's displacement being the
    #: weighted sum of its masters' (a wall end joined to another wall's strip).
    ties: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=np.float64).reshape(-1, 2)
        self.elements = [np.asarray(e, dtype=np.int64) for e in self.elements]
        n = len(self.elements)
        for name in ("kinds", "material_ids", "theta", "alpha"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"SectionMesh.{name} needs one entry per element")
        for e, (conn, kind) in enumerate(zip(self.elements, self.kinds)):
            if conn.size != n_nodes(kind):
                raise ValueError(f"element {e}: {kind} needs {n_nodes(kind)} nodes, got {conn.size}")
            if conn.min() < 0 or conn.max() >= self.nodes.shape[0]:
                raise ValueError(f"element {e} refers to a node that does not exist")
        missing = set(self.material_ids) - set(self.materials)
        if missing:
            raise ValueError(f"SectionMesh: no material for ids {sorted(missing)}")
        self._check_jacobians()

    # ------------------------------------------------------------------ integration
    def _check_jacobians(self) -> None:
        for e, (conn, kind) in enumerate(zip(self.elements, self.kinds)):
            X = self.nodes[conn]
            for p in quadrature(kind)[0]:
                _, dN = shape(kind, *p)
                if np.linalg.det(dN @ X) <= 0.0:
                    raise ValueError(f"element {e} is inverted or degenerate (order its corners counter-clockwise)")

    def quadrature_points(self) -> Iterator[QuadraturePoint]:
        for e, (conn, kind) in enumerate(zip(self.elements, self.kinds)):
            X = self.nodes[conn]
            pts, wts = quadrature(kind)
            for p, w in zip(pts, wts):
                N, dNn = shape(kind, *p)
                J = dNn @ X
                y, z = N @ X
                yield QuadraturePoint(e, float(y), float(z), float(w * np.linalg.det(J)), N,
                                      np.linalg.solve(J, dNn), conn)

    def element_stiffness(self, e: int) -> npt.NDArray[np.float64]:
        """3D stiffness ``C`` (6, 6) of element ``e`` in section axes."""
        return self.materials[self.material_ids[e]].stiffness_in_section(self.theta[e], self.alpha[e])

    def properties(self) -> Dict[str, float]:
        """Area and second moments about the origin: ``A``, ``S_y = int z``, ``S_z = int y``,
        ``I_yy = int z^2``, ``I_zz = int y^2``, ``I_yz = int y z``."""
        out = dict(A=0.0, S_y=0.0, S_z=0.0, I_yy=0.0, I_zz=0.0, I_yz=0.0)
        for q in self.quadrature_points():
            w = q.weight
            out["A"] += w
            out["S_y"] += w * q.z
            out["S_z"] += w * q.y
            out["I_yy"] += w * q.z * q.z
            out["I_zz"] += w * q.y * q.y
            out["I_yz"] += w * q.y * q.z
        return out

    # ------------------------------------------------------------------ builders
    @classmethod
    def rectangle(cls, b: float, h: float, n_y: int, n_z: int, material: SolidMaterial,
                  centre: Tuple[float, float] = (0.0, 0.0), theta: float = 0.0, alpha: float = 0.0) -> "SectionMesh":
        """A ``b`` (along ``y``) by ``h`` (along ``z``) rectangle of ``n_y x n_z`` ``Q8`` elements."""
        y0, z0 = centre[0] - 0.5 * b, centre[1] - 0.5 * h
        nodes, elements = _structured_q8(lambda u, v: (y0 + u * b, z0 + v * h), n_y, n_z)
        n = len(elements)
        return cls(nodes, elements, ["Q8"] * n, [1] * n, [theta] * n, [alpha] * n, {1: material})

    @classmethod
    def from_thin_walled(cls, section, laminates: Mapping[int, object], elements_along: int = 8,
                         rows_per_ply: int = 1, materials: Optional[Mapping[object, SolidMaterial]] = None
                         ) -> "SectionMesh":
        """Mesh the walls of a ``ThinWalledSection`` with ``WallLaminate`` s.

        Each wall is a strip from its start point to its end point, its plies stacked from
        ``n = -h/2`` to ``n = +h/2`` about the mid-line, each ply ``rows_per_ply`` elements
        thick and ``elements_along`` long. Ply materials become ``SolidMaterial.from_ply``
        unless ``materials`` maps a ``PlyMaterial`` to one.
        """
        all_nodes: List[np.ndarray] = []
        elements: List[np.ndarray] = []
        strips: Dict[int, List[int]] = {}                   # wall id -> its element indices
        edges: Dict[Tuple[int, int], List[int]] = {}        # (wall id, section point) -> edge nodes
        kinds, mids, thetas, alphas = [], [], [], []
        mat_ids: Dict[int, int] = {}
        mats: Dict[int, SolidMaterial] = {}
        offset = 0
        for w in section.walls:
            a, b = (np.asarray(p, dtype=float) for p in section.endpoints(w))
            t = (b - a) / np.linalg.norm(b - a)
            n_vec = np.array([-t[1], t[0]])                 # x cross s, in (y, z)
            alpha = float(np.degrees(np.arctan2(t[1], t[0])))
            lam = laminates[w.laminate_id]
            h = lam.thickness
            bounds = np.concatenate([[-0.5 * h], -0.5 * h + np.cumsum([p.thickness for p in lam.plies])])
            rows = []
            for k, p in enumerate(lam.plies):
                rows += list(np.linspace(bounds[k], bounds[k + 1], rows_per_ply + 1)[:-1])
            rows.append(0.5 * h)
            rows = np.array(rows)
            ply_of_row = np.repeat(np.arange(len(lam.plies)), rows_per_ply)
            L = float(np.linalg.norm(b - a))

            def at(u, v):
                # v in [0, 1] runs over the rows; map piecewise linearly onto the ply bounds
                r = np.interp(v * (len(rows) - 1), np.arange(len(rows)), rows)
                p = a + u * L * t + r * n_vec
                return p[0], p[1]

            nodes, elems = _structured_q8(at, elements_along, len(rows) - 1)
            gv = 2 * (len(rows) - 1) + 1
            n_edge = gv                                      # nodes on the u = 0 (and u = 1) edge
            edges[(w.wall_id, w.start)] = list(range(offset, offset + n_edge))
            edges[(w.wall_id, w.end)] = list(range(offset + nodes.shape[0] - n_edge, offset + nodes.shape[0]))
            strips[w.wall_id] = list(range(len(elements), len(elements) + len(elems)))
            for e, conn in enumerate(elems):
                row = e % (len(rows) - 1)
                ply = lam.plies[ply_of_row[row]]
                key = id(ply.material)
                if key not in mat_ids:
                    mid = len(mat_ids) + 1
                    mat_ids[key] = mid
                    mats[mid] = (materials or {}).get(ply.material) or SolidMaterial.from_ply(ply.material)
                elements.append(conn + offset)
                kinds.append("Q8")
                mids.append(mat_ids[key])
                thetas.append(float(ply.angle_deg))
                alphas.append(alpha)
            all_nodes.append(nodes)
            offset += nodes.shape[0]
        nodes = np.vstack(all_nodes)
        nodes, elements, rank = _merge_nodes(nodes, elements, return_map=True)
        edges = {k: [int(rank[n]) for n in v] for k, v in edges.items()}
        ties = _junction_ties(section, nodes, elements, strips, edges)
        return cls(nodes, elements, kinds, mids, thetas, alphas, mats, getattr(section, "section_id", None), ties)

    @classmethod
    def from_gmsh(cls, path: str, groups: Mapping[object, Tuple[int, float, float]],
                  materials: Mapping[int, SolidMaterial]) -> "SectionMesh":
        """Read a Gmsh ``.msh`` (4.1 ASCII). ``groups`` maps each physical surface (its tag or
        name) to ``(material_id, theta, alpha)``; its ``T6`` (type 9) and ``Q8`` (type 16)
        elements are kept. The mesh's ``(x, y)`` coordinates are the section's ``(y, z)``."""
        nodes, elems = read_gmsh(path)
        conn, kinds, mids, th, al = [], [], [], [], []
        for (tag, name, kind, c) in elems:
            key = name if name in groups else tag
            if key not in groups:
                continue
            m, t, a = groups[key]
            conn.append(c)
            kinds.append(kind)
            mids.append(int(m))
            th.append(float(t))
            al.append(float(a))
        if not conn:
            raise ValueError(f"{path}: no T6 or Q8 elements in the physical groups {list(groups)}")
        return cls(nodes, conn, kinds, mids, th, al, dict(materials))


# ---------------------------------------------------------------------- helpers
def _structured_q8(at, n_u: int, n_v: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """A ``n_u x n_v`` grid of ``Q8`` on the map ``(u, v) in [0, 1]^2 -> (y, z)``."""
    gu, gv = 2 * n_u + 1, 2 * n_v + 1
    index = -np.ones((gu, gv), dtype=np.int64)
    pts = []
    for i in range(gu):
        for j in range(gv):
            if i % 2 == 1 and j % 2 == 1:
                continue                                    # element centre: not a Q8 node
            index[i, j] = len(pts)
            pts.append(at(i / (gu - 1), j / (gv - 1)))
    elems = []
    for i in range(n_u):
        for j in range(n_v):
            I, J = 2 * i, 2 * j
            elems.append(np.array([index[I, J], index[I + 2, J], index[I + 2, J + 2], index[I, J + 2],
                                   index[I + 1, J], index[I + 2, J + 1], index[I + 1, J + 2], index[I, J + 1]]))
    return np.array(pts, dtype=np.float64), elems


def _junction_ties(section, nodes, elements, strips, edges) -> Dict[int, List[Tuple[int, float]]]:
    """Tie each wall's end edge into the strips of the walls listed before it at that point.

    At a section point the first wall listed is the master. Every later wall's end-edge
    node that is not already a node of an earlier wall's strip there becomes the
    interpolation of the earlier strip's element nearest to it (the point clamped into the
    element). Ties onto tied nodes are resolved, so masters are always free nodes.
    """
    ties: Dict[int, List[Tuple[int, float]]] = {}
    for pid in section.points:
        walls = [w.wall_id for w in section.walls if pid in (w.start, w.end)]
        for k, wid in enumerate(walls[1:], start=1):
            earlier = [e for v in walls[:k] for e in strips[v]]
            earlier_nodes = {int(n) for e in earlier for n in elements[e]}
            for n in edges[(wid, pid)]:
                if n in earlier_nodes or n in ties:
                    continue
                best = None
                for e in earlier:
                    xi, eta, d = _closest_natural(nodes[elements[e]], nodes[n])
                    if best is None or d < best[0]:
                        best = (d, e, xi, eta)
                _, e, xi, eta = best
                N, _ = shape("Q8", xi, eta)
                row: Dict[int, float] = {}
                for m, c in zip(elements[e], N):
                    for mm, cc in ties.get(int(m), [(int(m), 1.0)]):
                        row[mm] = row.get(mm, 0.0) + c * cc
                ties[int(n)] = [(m, c) for m, c in row.items() if abs(c) > 1e-14]
    return ties


def _closest_natural(X: np.ndarray, p: np.ndarray) -> Tuple[float, float, float]:
    """Natural coordinates in ``[-1, 1]^2`` of the point of a ``Q8`` closest to ``p``, and the distance."""
    xi = np.zeros(2)
    for _ in range(30):
        N, dN = shape("Q8", *xi)
        r = N @ X - p
        J = dN @ X                                          # rows: d(y, z)/dxi, d(y, z)/deta
        step = np.linalg.lstsq(J.T, -r, rcond=None)[0]
        xi = np.clip(xi + step, -1.0, 1.0)
        if np.linalg.norm(step) < 1e-13:
            break
    N, _ = shape("Q8", *xi)
    return float(xi[0]), float(xi[1]), float(np.linalg.norm(N @ X - p))


def _merge_nodes(nodes: np.ndarray, elements: List[np.ndarray], tol: float = 1e-10, return_map: bool = False):
    """Merge coincident nodes (walls that meet at a section point)."""
    scale = max(float(np.ptp(nodes, axis=0).max()), 1e-300)
    keys = np.round(nodes / (tol * scale)).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    new_nodes = nodes[first[order]]
    if return_map:
        return new_nodes, [rank[inverse[e]] for e in elements], rank[inverse]
    return new_nodes, [rank[inverse[e]] for e in elements]


def read_gmsh(path: str):
    """Nodes ``(n, 2)`` and ``[(physical tag, physical name, kind, connectivity)]`` of a
    Gmsh 4.1 ASCII mesh's ``T6`` and ``Q8`` elements on physical surfaces."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    def block(name):
        m = re.search(rf"\${name}\n(.*?)\n\$End{name}", text, flags=re.S)
        return m.group(1).split("\n") if m else []

    fmt = block("MeshFormat")
    if not fmt or not fmt[0].split()[0].startswith("4") or fmt[0].split()[1] != "0":
        raise ValueError(f"{path}: only Gmsh format 4.x ASCII is read")
    names = {}
    for ln in block("PhysicalNames")[1:]:
        dim, tag, name = ln.split(maxsplit=2)
        if int(dim) == 2:
            names[int(tag)] = name.strip().strip('"')
    # entities: surface tag -> physical tags
    ent = block("Entities")
    surf_phys: Dict[int, List[int]] = {}
    if ent:
        n_pts, n_cur, n_sur, _ = map(int, ent[0].split())
        for ln in ent[1 + n_pts + n_cur: 1 + n_pts + n_cur + n_sur]:
            v = ln.split()
            tag = int(v[0])
            n_phys = int(v[7])
            surf_phys[tag] = [int(x) for x in v[8:8 + n_phys]]
    nb = block("Nodes")
    _, n_nodes_total, _, _ = map(int, nb[0].split())
    coords: Dict[int, Tuple[float, float]] = {}
    i = 1
    while i < len(nb) and nb[i].strip():
        _, _, param, n = map(int, nb[i].split())
        tags = [int(nb[i + 1 + k]) for k in range(n)]
        for k, t in enumerate(tags):
            x, y, *_ = map(float, nb[i + 1 + n + k].split())
            coords[t] = (x, y)
        i += 1 + 2 * n
    tag_list = sorted(coords)
    index = {t: k for k, t in enumerate(tag_list)}
    nodes = np.array([coords[t] for t in tag_list])
    eb = block("Elements")
    out = []
    kinds = {9: "T6", 16: "Q8"}
    i = 1
    while i < len(eb) and eb[i].strip():
        dim, ent_tag, etype, n = map(int, eb[i].split())
        rows = eb[i + 1:i + 1 + n]
        i += 1 + n
        if dim != 2 or etype not in kinds:
            continue
        for phys in surf_phys.get(ent_tag, [None]):
            for r in rows:
                v = [int(x) for x in r.split()]
                out.append((phys, names.get(phys), kinds[etype], np.array([index[t] for t in v[1:]])))
    return nodes, out


def parse_section_mesh_2d(path: str, materials: Mapping[int, SolidMaterial]) -> Dict[int, SectionMesh]:
    """Read ``section_mesh_2d.txt``: ``{section_id: SectionMesh}``.

    Format::

        [Section_Node]
        [section_id] [node_id] [y] [z]
        [Section_Element]
        [section_id] [element_id] [type] [nodes...] [material_id] [theta] [alpha]

    ``type`` is ``T6`` or ``Q8`` with its 6 or 8 node ids (Gmsh order). An optional
    ``[Solid_Material]`` block, ``[material_id] [E1] [E2] [E3] [G12] [G13] [G23] [nu12]
    [nu13] [nu23] [rho]``, adds or overrides materials (others come from ``laminate.txt``).
    """
    nodes: Dict[int, Dict[int, Tuple[float, float]]] = {}
    elems: Dict[int, List[Tuple[str, List[int], int, float, float]]] = {}
    mats = dict(materials)
    section = None
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#")[0].strip()
            if not line:
                continue
            low = line.lower()
            if low in ("[section_node]", "[section_element]", "[solid_material]"):
                section = low
                continue
            if line.startswith("["):
                continue                                    # column header
            v = line.split()
            if section == "[section_node]":
                nodes.setdefault(int(v[0]), {})[int(v[1])] = (float(v[2]), float(v[3]))
            elif section == "[section_element]":
                kind = v[2].upper()
                n = n_nodes(kind)
                if len(v) != 3 + n + 3:
                    raise ValueError(f"{path}: {kind} row needs {n} node ids and material, theta, alpha: {line!r}")
                elems.setdefault(int(v[0]), []).append(
                    (kind, [int(x) for x in v[3:3 + n]], int(v[3 + n]), float(v[4 + n]), float(v[5 + n])))
            elif section == "[solid_material]":
                f = [float(x) for x in v[1:]]
                mats[int(v[0])] = SolidMaterial(*f[:9], rho=f[9] if len(f) > 9 else 0.0)
    out = {}
    for sid, rows in elems.items():
        if sid not in nodes:
            raise ValueError(f"{path}: section {sid} has elements but no nodes")
        ids = sorted(nodes[sid])
        index = {nid: k for k, nid in enumerate(ids)}
        try:
            conn = [np.array([index[n] for n in r[1]]) for r in rows]
        except KeyError as exc:
            raise ValueError(f"{path}: section {sid} element refers to unknown node {exc}") from None
        out[sid] = SectionMesh(np.array([nodes[sid][n] for n in ids]), conn, [r[0] for r in rows],
                               [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows],
                               {m: mats[m] for m in {r[2] for r in rows} if m in mats}, sid)
    return out


__all__: Sequence[str] = ["SectionMesh", "QuadraturePoint", "parse_section_mesh_2d", "read_gmsh"]
