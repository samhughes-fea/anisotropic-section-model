# pre_processing/section_integration/solid/from_geometry.py
"""Conforming 2D section meshes of an implicit-geometry-kernel blade section.

``section_mesh_from_cut`` meshes a ``SectionCut`` (``BladeGeometry.section_at(z)``) with
layered ``Q8`` elements that follow the true wall geometry, one or more element rows
per ply, and no overlaps or ties:

- **Skins** are one closed layered band around the section. The outer face is the
  outer mould line (the skin midline plus half the skin thickness along the outward
  normal) and the plies stack inward from it.
- **Spar caps** sit inside their skin, as the geometry builds them: their rows continue
  inward from the skin's inner face over the stations the cap is bonded to, sharing
  that face's nodes. A cap's ends are free faces normal to the skin.
- **Webs** are straight layered bands between the innermost faces (skin, or cap where
  there is one) of the two skins. Each web ply interface meets the face at a station
  of the skin mesh, so the web's end nodes are the face's nodes: the junction is the
  true geometry, with no overlap and no ties. A cap edge that ends under a web, or
  within one cap thickness of it, is moved to the web's far face, so the web lands on
  one face.
- **Trailing edge**: where the section is thinner than two skins, the skin columns are
  shortened to reach the camber line from each side, so the material there is the
  solid wedge the two skins fill.

The section is placed like the thin-walled one (``thin_walled.from_geometry``):
coordinates ``(y, z) = (x, y)`` of the blade frame, measured from the pitch-axis point.
Laminates are bound by label, region or kind, plies listed from the inside of the
section out on skins and caps and from the leading-edge face to the trailing-edge face
on webs. Each element gets its ply's material, the ply angle ``theta`` and the wall
angle ``alpha`` of its wall direction ``e_s`` (with ``x cross e_s`` along the stacking
direction), so the 2D section and the thin-walled one see the same laminates.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pre_processing.section_integration.solid.material import SolidMaterial
from pre_processing.section_integration.solid.mesh import SectionMesh
from pre_processing.section_integration.thin_walled.from_geometry import laminate_lookup

__all__ = ["section_mesh_from_cut", "write_section_mesh_2d"]


# ----------------------------------------------------------------------------- loop geometry
class _Loop:
    """A closed polyline (the skin midline) with arc length, positions and outward normals."""

    def __init__(self, pts: np.ndarray):
        self.P = np.asarray(pts, float)
        seg = np.diff(np.vstack([self.P, self.P[:1]]), axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.s[-1])
        x, y = self.P[:, 0], self.P[:, 1]
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        self.ccw = area > 0.0

    def at(self, s: float) -> np.ndarray:
        s = float(s) % self.length
        k = int(np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.P) - 1))
        f = (s - self.s[k]) / self.seg_len[k]
        return (1 - f) * self.P[k] + f * self.P[(k + 1) % len(self.P)]

    def tangent(self, s: float, h: float) -> np.ndarray:
        d = self.at(s + h) - self.at(s - h)
        return d / np.linalg.norm(d)

    def outward(self, s: float, h: float) -> np.ndarray:
        t = self.tangent(s, h)
        left = np.array([-t[1], t[0]])
        return -left if self.ccw else left

    def project(self, p: np.ndarray) -> float:
        """Arc length of the loop point nearest ``p``."""
        A, B = self.P, np.roll(self.P, -1, axis=0)
        AB = B - A
        f = np.clip(np.einsum("ij,ij->i", p - A, AB) / np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-300), 0, 1)
        Q = A + f[:, None] * AB
        k = int(np.argmin(np.linalg.norm(Q - p, axis=1)))
        return float(self.s[k] + f[k] * self.seg_len[k])


def _chain(walls, tol):
    """Skin walls as one closed loop of points (the first point not repeated)."""
    left = [w.xy.copy() for w in walls]
    loop = left.pop(0)
    while left:
        for k, p in enumerate(left):
            if np.linalg.norm(p[0] - loop[-1]) <= tol:
                loop = np.vstack([loop, p[1:]])
                break
            if np.linalg.norm(p[-1] - loop[-1]) <= tol:
                loop = np.vstack([loop, p[::-1][1:]])
                break
        else:
            raise ValueError("the skin walls do not form one closed loop")
        left.pop(k)
    if np.linalg.norm(loop[-1] - loop[0]) > tol:
        raise ValueError("the skin walls do not close")
    loop = loop[:-1]
    keep = np.r_[True, np.linalg.norm(np.diff(loop, axis=0), axis=1) > tol]
    return loop[keep]


def _ray_gap(origin, direction, outer: np.ndarray, skip: float, ccw: bool) -> float:
    """Distance along ``direction`` from ``origin`` (on the polygon ``outer``) to its far side.

    Only edges the ray leaves the polygon through count (their outward normal faces
    the ray), so the edges next to ``origin`` are never hit.
    """
    A, B = outer, np.roll(outer, -1, axis=0)
    d = np.asarray(direction, float)
    best = np.inf
    for a, b in zip(A, B):
        e = b - a
        n_out = np.array([e[1], -e[0]]) if ccw else np.array([-e[1], e[0]])
        if n_out @ d <= 0.0:
            continue
        M = np.array([[d[0], -e[0]], [d[1], -e[1]]])
        det = np.linalg.det(M)
        if abs(det) < 1e-14:
            continue
        lam, mu = np.linalg.solve(M, a - origin)
        if lam > skip and -1e-12 <= mu <= 1 + 1e-12:
            best = min(best, lam)
    return best


# ----------------------------------------------------------------------------- the mesher
def section_mesh_from_cut(
    cut,
    laminates: Mapping[str, object],
    library: Optional[Mapping[int, object]] = None,
    element_size: Optional[float] = None,
    rows_per_ply: int = 1,
    thickness_tol: float = 0.01,
    origin: str = "pitch_axis",
    materials: Optional[Mapping[object, SolidMaterial]] = None,
    section_id: Optional[int] = None,
) -> SectionMesh:
    """Layered ``Q8`` mesh of an implicit-geometry-kernel ``SectionCut``.

    Parameters
    ----------
    cut : SectionCut
    laminates, library :
        Laminate binding, as for ``thin_walled_from_cut``.
    element_size : float, optional
        Element length along the walls (default 1 % of the chord).
    rows_per_ply : int
        Element rows through each ply.
    thickness_tol : float
        Allowed relative difference between a skin or cap laminate and its wall.
    origin : {"pitch_axis", "blade"}
    materials : mapping PlyMaterial -> SolidMaterial, optional
        3D materials for ply materials (default ``SolidMaterial.from_ply``).
    section_id : int, optional

    Returns
    -------
    SectionMesh
        With ``labels``: the geometry label of each element's wall.
    """
    if origin not in ("pitch_axis", "blade"):
        raise ValueError(f"origin must be 'pitch_axis' or 'blade', got {origin!r}")
    if rows_per_ply < 1:
        raise ValueError("rows_per_ply must be at least 1")
    h = float(element_size) if element_size else 0.01 * float(cut.chord)
    o = np.zeros(2) if origin == "blade" else np.asarray(cut.offset, float)
    skins, caps, webs = cut.walls_of("skin"), cut.walls_of("cap"), cut.walls_of("web")
    if not skins:
        raise ValueError("the cut has no skin")
    scale = max(1.0, float(np.abs(np.vstack([w.xy for w in cut.walls])).max()))
    tol = 1e-7 * scale

    def lam_of(wall):
        lam = laminate_lookup(wall, laminates, library)
        if wall.kind != "web" and abs(lam.thickness - wall.thickness) > thickness_tol * wall.thickness:
            raise ValueError(f"wall '{wall.label}' is {wall.thickness * 1e3:.4g} mm thick in the geometry but its "
                             f"laminate is {lam.thickness * 1e3:.4g} mm (thickness_tol {thickness_tol:g})")
        return lam

    # skin loop, started at the trailing edge (the point farthest along the chord)
    pts = _chain(skins, tol)
    t = np.radians(cut.twist)
    chord_dir = np.array([np.cos(t), -np.sin(t)])
    start = int(np.argmax(pts @ chord_dir))
    loop = _Loop(np.roll(pts, -start, axis=0))
    hn = 0.25 * h                                            # stencil for tangents
    skin_lam = {w.label: lam_of(w) for w in skins}
    # one skin laminate per point of the loop: the label of the skin wall each loop point lies on
    labels_by_s = []
    for w in skins:
        s_pts = np.array([loop.project(p) for p in w.xy])
        lo, hi = s_pts.min(), s_pts.max()
        if hi - lo > 0.5 * loop.length:                       # wraps through s = 0
            labels_by_s.append((w.label, hi, lo + loop.length))
        else:
            labels_by_s.append((w.label, lo, hi))

    def skin_label(s):
        s = s % loop.length
        for label, lo, hi in labels_by_s:
            if lo - 1e-9 <= s <= hi + 1e-9 or lo - 1e-9 <= s + loop.length <= hi + 1e-9:
                return label
        return labels_by_s[int(np.argmin([min(abs(s - lo), abs(s - hi)) for _, lo, hi in labels_by_s]))][0]

    t_skin = {lb: lam.thickness for lb, lam in skin_lam.items()}
    if len({round(v, 12) for v in t_skin.values()}) != 1:
        raise ValueError("section_mesh_from_cut needs one skin thickness around the section")
    ts = next(iter(t_skin.values()))

    # caps: the arc of the loop they are bonded to
    cap_ranges = []                                           # (label, s0, s1, laminate)
    for c in caps:
        lam = lam_of(c)
        ss = []
        for host_label, (_, ih) in c.bonds.items():
            host = next(w for w in skins if w.label == host_label)
            ss += [loop.project(host.xy[i]) for i in ih]
        if not ss:
            ss = [loop.project(p) for p in c.xy]
        ss = np.array(ss)
        if ss.max() - ss.min() > 0.5 * loop.length:
            raise ValueError(f"cap '{c.label}' crosses the trailing edge")
        cap_ranges.append([c.label, float(ss.min()), float(ss.max()), lam])

    def cap_at(s):
        for cr in cap_ranges:
            if cr[1] - 1e-12 <= s <= cr[2] + 1e-12:
                return cr
        return None

    def inner_face(s, with_caps=True):
        cr = cap_at(s % loop.length) if with_caps else None
        depth = ts / 2 + (cr[3].thickness if cr is not None else 0.0)
        return loop.at(s) - depth * loop.outward(s, hn)

    # webs: straight bands; each ply interface meets the innermost face of each skin
    web_info = []
    for w in webs:
        lam = lam_of(w)
        a, b = w.xy[0], w.xy[-1]
        d = (b - a) / np.linalg.norm(b - a)
        m = np.array([-d[1], d[0]])
        if m @ chord_dir < 0:                                 # stacking normal towards the TE
            a, b, d, m = b, a, -d, -m
        q = -0.5 * lam.thickness + np.concatenate([[0.0], np.cumsum([p.thickness for p in lam.plies])])
        q_rows = []
        for k in range(len(lam.plies)):
            q_rows += list(np.linspace(q[k], q[k + 1], rows_per_ply + 1)[:-1])
        q_rows = np.array(q_rows + [q[-1]])
        web_info.append(dict(wall=w, lam=lam, a=a, b=b, d=d, m=m, q=q_rows, sa=loop.project(a), sb=loop.project(b)))

    def foot(wi, end, with_caps):
        """Loop arc lengths where each web row line meets the face at ``end``."""
        s0 = wi["sa"] if end == 0 else wi["sb"]
        out = []
        for qk in (wi["q"] if True else ()):
            g = lambda s: float((inner_face(s, with_caps) - wi["a"]) @ wi["m"] - qk)
            span = 3 * (wi["lam"].thickness + ts) + 2 * h
            lo, hi = s0 - span, s0 + span
            xs = np.linspace(lo, hi, 61)
            gs = np.array([g(x) for x in xs])
            k = [i for i in range(60) if gs[i] == 0 or gs[i] * gs[i + 1] < 0]
            if not k:
                raise ValueError(f"web '{wi['wall'].label}' does not meet its skin")
            i = min(k, key=lambda i: abs(xs[i] - s0))
            x0, x1 = xs[i], xs[i + 1]
            for _ in range(60):
                xm = 0.5 * (x0 + x1)
                if g(x0) * g(xm) <= 0:
                    x1 = xm
                else:
                    x0 = xm
            out.append(0.5 * (x0 + x1))
        return np.array(out)

    # move cap edges that end under (or just beside) a web to the web's far face
    for wi in web_info:
        for end in (0, 1):
            f = foot(wi, end, with_caps=False)
            lo, hi = f.min(), f.max()
            for cr in cap_ranges:
                tc = cr[3].thickness
                for k, edge in ((1, cr[1]), (2, cr[2])):
                    if lo - tc - 2 * ts <= edge <= hi + tc + 2 * ts:
                        centre = 0.5 * (cr[1] + cr[2])
                        if centre > hi:
                            cr[k] = lo
                        elif centre < lo:
                            cr[k] = hi
    # a cap that reaches under a web must cover its whole foot: push the edge out until
    # every web row meets the cap face strictly inside the cap
    for wi in web_info:
        for end in (0, 1):
            for cr in cap_ranges:
                centre = 0.5 * (cr[1] + cr[2])
                for _ in range(200):
                    f = foot(wi, end, with_caps=True)
                    lo, hi = f.min(), f.max()
                    if not (cr[1] <= hi + 1e-9 and cr[2] >= lo - 1e-9):
                        break                                 # this cap is not under this web
                    step = 0.02 * cr[3].thickness
                    if centre > hi and cr[1] >= lo - 1e-3 * step:
                        cr[1] = min(cr[1], lo) - step
                    elif centre < lo and cr[2] <= hi + 1e-3 * step:
                        cr[2] = max(cr[2], hi) + step
                    else:
                        break
    for wi in web_info:
        wi["feet"] = [foot(wi, 0, True), foot(wi, 1, True)]
        for f in wi["feet"]:
            if np.any(np.diff(f) * np.sign(f[-1] - f[0]) <= 0):
                raise ValueError(f"web '{wi['wall'].label}' meets a step in the inner face (cap edge under the web)")

    # loop stations: fixed ones (TE, cap edges, web feet) and fill between them
    fixed = [0.0]
    no_fill = []
    for cr in cap_ranges:
        fixed += [cr[1], cr[2]]
    for wi in web_info:
        for f in wi["feet"]:
            fs = np.sort(f)
            fixed += list(fs)
            no_fill += [(fs[i], fs[i + 1]) for i in range(len(fs) - 1)]
    fixed = np.sort(np.mod(fixed, loop.length))
    merged = [fixed[0]]
    for x in fixed[1:]:
        if x - merged[-1] > 1e-6 * h:
            merged.append(x)
    if loop.length - merged[-1] <= 1e-6 * h:
        merged.pop()
    fixed = np.append(merged, loop.length)
    stations = [fixed[0]]
    for a_, b_ in zip(fixed[:-1], fixed[1:]):
        if any(abs(a_ - x) < 1e-9 and abs(b_ - y) < 1e-9 for x, y in no_fill):
            stations.append(b_)
            continue
        n = max(1, int(np.ceil((b_ - a_) / h - 1e-9)))
        stations += list(np.linspace(a_, b_, n + 1)[1:])
    stations = np.array(stations[:-1])                        # the loop closes on station 0
    M = len(stations)

    def half_station(I):
        """Arc length of loop half-index ``I`` (even: station, odd: mid-point)."""
        I %= 2 * M
        if I % 2 == 0:
            return stations[I // 2]
        a_, b_ = stations[I // 2], stations[(I // 2 + 1) % M]
        if b_ < a_:
            b_ += loop.length
        return 0.5 * (a_ + b_)

    # trailing-edge shortening: available depth from the OML towards the other side
    outer_poly = np.array([loop.at(s) + 0.5 * ts * loop.outward(s, hn) for s in np.linspace(0, loop.length, 4 * M, endpoint=False)])

    def depth_scale(s):
        n_out = loop.outward(s, hn)
        O = loop.at(s) + 0.5 * ts * n_out
        gap = _ray_gap(O, -n_out, outer_poly, skip=1e-3 * ts, ccw=loop.ccw)
        return min(1.0, 0.5 * gap / ts)

    # nodes
    pos: List[np.ndarray] = []
    index: Dict[tuple, int] = {}

    def node(key, p):
        if key not in index:
            index[key] = len(pos)
            pos.append(np.asarray(p, float) - o)
        return index[key]

    elements, kinds, mids, thetas, alphas, labels = [], [], [], [], [], []
    mat_ids: Dict[int, int] = {}
    mats: Dict[int, SolidMaterial] = {}

    def material_id(ply):
        key = id(ply.material)
        if key not in mat_ids:
            mat_ids[key] = len(mat_ids) + 1
            mats[mat_ids[key]] = (materials or {}).get(ply.material) or SolidMaterial.from_ply(ply.material)
        return mat_ids[key]

    def rows_of(lam):
        """Row bounds (relative, 0 = first ply's outer... see callers) and the ply of each row."""
        b = np.concatenate([[0.0], np.cumsum([p.thickness for p in lam.plies])])
        r, ply = [], []
        for k in range(len(lam.plies)):
            r += list(np.linspace(b[k], b[k + 1], rows_per_ply + 1)[:-1])
            ply += [k] * rows_per_ply
        return np.array(r + [b[-1]]), ply

    # skin rows: depth measured inward from the OML; skin plies are listed inside first,
    # so the outermost row is the last ply
    def add_q8(c, mid_, label, ply, e_s):
        X = np.array([pos[i] for i in c[:4]])
        area = 0.5 * np.sum(X[:, 0] * np.roll(X[:, 1], -1) - np.roll(X[:, 0], -1) * X[:, 1])
        if area < 0:
            c = [c[0], c[3], c[2], c[1], c[7], c[6], c[5], c[4]]
        elements.append(np.array(c))
        kinds.append("Q8")
        mids.append(mid_)
        thetas.append(float(ply.angle_deg))
        alphas.append(float(np.degrees(np.arctan2(e_s[1], e_s[0]))))
        labels.append(label)

    # skin rows: one set of interface depths (from the OML) for the whole skin, the union
    # of every skin laminate's, so neighbouring skin laminates share their nodes
    depths = set()
    for lam in skin_lam.values():
        rb, _ = rows_of(lam)
        depths.update(np.round(ts - rb, 15))
    depth_skin = np.array(sorted(depths))
    nr = len(depth_skin) - 1
    scale_cache: Dict[int, float] = {}

    def skin_node(I, J):
        I %= 2 * M
        key = ("skin", I, J)
        if key in index:
            return index[key]
        s = half_station(I)
        if I not in scale_cache:
            scale_cache[I] = depth_scale(s)
        d = depth_skin[J // 2] if J % 2 == 0 else 0.5 * (depth_skin[J // 2] + depth_skin[J // 2 + 1])
        n_out = loop.outward(s, hn)
        return node(key, loop.at(s) + (0.5 * ts - scale_cache[I] * d) * n_out)

    def ply_at_depth(lam, d):
        """Ply of ``lam`` (plies inside first) at depth ``d`` from the outer face."""
        b = np.concatenate([[0.0], np.cumsum([p.thickness for p in lam.plies])])
        k = int(np.clip(np.searchsorted(b, lam.thickness - d) - 1, 0, len(lam.plies) - 1))
        return lam.plies[k]

    for i in range(M):
        I0, I1 = 2 * i, 2 * i + 2
        s_mid = half_station(2 * i + 1)
        label = skin_label(s_mid)
        lam = skin_lam[label]
        n_out = loop.outward(s_mid, hn)
        e_s = np.array([n_out[1], -n_out[0]])                 # x cross e_s = n_out
        for j in range(nr):
            J0, J1 = 2 * j, 2 * j + 2
            c = [skin_node(I0, J0), skin_node(I1, J0), skin_node(I1, J1), skin_node(I0, J1),
                 skin_node(I0 + 1, J0), skin_node(I1, J0 + 1), skin_node(I0 + 1, J1), skin_node(I0, J0 + 1)]
            ply = ply_at_depth(lam, 0.5 * (depth_skin[j] + depth_skin[j + 1]))
            add_q8(c, material_id(ply), label, ply, e_s)
        # cap rows below the skin
        cr = cap_at(s_mid)
        if cr is not None and cr[1] - 1e-9 <= stations[i] and half_station(I1) <= cr[2] + 1e-9:
            clam = cr[3]
            cb, cply = rows_of(clam)
            depth_cap = ts + (clam.thickness - cb)[::-1]          # outer (skin-side) first
            cply_from_outer = cply[::-1]

            def cap_node(I, J):
                if J == 0:
                    return skin_node(I, 2 * nr)
                s = half_station(I)
                d = depth_cap[J // 2] if J % 2 == 0 else 0.5 * (depth_cap[J // 2] + depth_cap[J // 2 + 1])
                n_out = loop.outward(s, hn)
                return node(("cap", I % (2 * M), J), loop.at(s) + (0.5 * ts - d) * n_out)

            for j in range(len(cb) - 1):
                J0, J1 = 2 * j, 2 * j + 2
                c = [cap_node(I0, J0), cap_node(I1, J0), cap_node(I1, J1), cap_node(I0, J1),
                     cap_node(I0 + 1, J0), cap_node(I1, J0 + 1), cap_node(I0 + 1, J1), cap_node(I0, J0 + 1)]
                n_out = loop.outward(s_mid, hn)
                ply = clam.plies[cply_from_outer[j]]
                add_q8(c, material_id(ply), cr[0], ply, np.array([n_out[1], -n_out[0]]))

    # the innermost-face node at loop half-index I
    def face_node(I):
        s = half_station(I)
        cr = cap_at(s % loop.length)
        if cr is not None and ("cap", I % (2 * M), 2 * (len(rows_of(cr[3])[0]) - 1)) in index:
            return index[("cap", I % (2 * M), 2 * (len(rows_of(cr[3])[0]) - 1))]
        key = ("skin", I % (2 * M), 2 * nr)
        if key not in index:
            raise ValueError("internal: a web foot is not on a skin node")
        return index[key]

    def station_index(s):
        k = int(np.argmin(np.abs(((stations - s + 0.5 * loop.length) % loop.length) - 0.5 * loop.length)))
        if abs(((stations[k] - s + 0.5 * loop.length) % loop.length) - 0.5 * loop.length) > 1e-7 * loop.length:
            raise ValueError("internal: web foot is not a station")
        return k

    for wn, wi in enumerate(web_info):
        lam = wi["lam"]
        feet = [[station_index(s) for s in f] for f in wi["feet"]]
        nq = len(wi["q"]) - 1
        ends = []
        for f in feet:
            col = []
            for k in range(nq + 1):
                col.append(face_node(2 * f[k]))
                if k < nq:
                    I_lo = 2 * min(f[k], f[k + 1])
                    if abs(f[k + 1] - f[k]) != 1 and not (min(f[k], f[k + 1]) == 0 and max(f[k], f[k + 1]) == M - 1):
                        raise ValueError("internal: web rows are not adjacent stations")
                    I_mid = (2 * f[k] + 1) if (f[k + 1] - f[k]) % M == 1 else (2 * f[k + 1] + 1)
                    col.append(face_node(I_mid))
            ends.append(col)
        bot, top = ends
        Lw = float(np.linalg.norm(pos[top[0]] - pos[bot[0]]))
        n_al = max(1, int(np.ceil(Lw / h)))
        _, wply = rows_of(lam)

        def web_node(i2, j2):
            if i2 == 0:
                return bot[j2]
            if i2 == 2 * n_al:
                return top[j2]
            u = i2 / (2 * n_al)
            return node(("web", wn, i2, j2), (1 - u) * (pos[bot[j2]] + o) + u * (pos[top[j2]] + o))

        e_s = np.array([wi["m"][1], -wi["m"][0]])
        for i in range(n_al):
            for j in range(nq):
                I0, I1, J0, J1 = 2 * i, 2 * i + 2, 2 * j, 2 * j + 2
                c = [web_node(I0, J0), web_node(I1, J0), web_node(I1, J1), web_node(I0, J1),
                     web_node(I0 + 1, J0), web_node(I1, J0 + 1), web_node(I0 + 1, J1), web_node(I0, J0 + 1)]
                ply = lam.plies[wply[j]]
                add_q8(c, material_id(ply), wi["wall"].label, ply, e_s)

    mesh = SectionMesh(np.array(pos), elements, kinds, mids, thetas, alphas, mats, section_id)
    mesh.labels = labels
    return mesh


# ----------------------------------------------------------------------------- writer
def write_section_mesh_2d(path: str, meshes: Mapping[int, SectionMesh]) -> None:
    """Write ``section_mesh_2d.txt`` for ``{section_id: SectionMesh}``, with its materials.

    Material ids are renumbered across sections (one ``[Solid_Material]`` row per
    distinct material object).
    """
    mat_rows: Dict[int, Tuple[int, SolidMaterial]] = {}
    lines_n, lines_e = ["[Section_Node]", "[section_id] [node_id] [y] [z]"], \
        ["[Section_Element]", "[section_id] [element_id] [type] [nodes...] [material_id] [theta] [alpha]"]
    for sid, mesh in meshes.items():
        for k, (y, z) in enumerate(mesh.nodes, start=1):
            lines_n.append(f"{sid} {k} {float(y)!r} {float(z)!r}")
        for e, (conn, kind, m, th, al) in enumerate(zip(mesh.elements, mesh.kinds, mesh.material_ids, mesh.theta,
                                                       mesh.alpha), start=1):
            mat = mesh.materials[m]
            key = id(mat)
            if key not in mat_rows:
                mat_rows[key] = (len(mat_rows) + 1, mat)
            gid = mat_rows[key][0]
            lines_e.append(f"{sid} {e} {kind} " + " ".join(str(int(n) + 1) for n in conn) + f" {gid} {float(th)!r} {float(al)!r}")
    lines_m = ["[Solid_Material]", "[material_id] [E1] [E2] [E3] [G12] [G13] [G23] [nu12] [nu13] [nu23] [rho]"]
    for gid, m in sorted(mat_rows.values(), key=lambda r: r[0]):
        vals = (m.E1, m.E2, m.E3, m.G12, m.G13, m.G23, m.nu12, m.nu13, m.nu23, m.rho)
        lines_m.append(f"{gid} " + " ".join(repr(float(v)) for v in vals))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines_m + [""] + lines_n + [""] + lines_e) + "\n")
