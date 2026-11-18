"""The same-mesh cases: BECAS's own validation meshes, and composite sections built here.

Two kinds of case:

- ``EXAMPLES``: meshes shipped with BECAS (``$BECAS_HOME/examples/<name>``), read with
  ``becas_io.read_becas``. They are licensed and never copied here; the cases run only where
  ``BECAS_HOME`` points at a local BECAS.
- ``BUILT``: sections meshed here with ``solid.structured`` (Q8), written to BECAS input
  with ``becas_io.write_becas`` and run through BECAS on that same mesh. Their geometry,
  materials and layups are **this benchmark's own choice** (stated in each builder); they are
  meant to exercise every coupling, not to reproduce a published specimen.

Every builder takes a refinement ``n`` (1 = the mesh the BECAS reference was made on).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from anisotropic_section_model.solid import SolidMaterial
from anisotropic_section_model.solid.structured import Patch, arc_patch, structured_mesh

from becas_io import becas_home, read_becas

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------- BECAS's meshes
#: BECAS example folder -> what it is (BECAS user guide, Risø-R-1785, section 4)
EXAMPLES: Dict[str, str] = {
    "S1": "isotropic square (Q4)",
    "S2_10": "two-material square, E1/E2 = 10 (Q4)",
    "S3_0": "orthotropic square, 0 deg (Q4)",
    "S3_22.5": "orthotropic square, 22.5 deg (Q4)",
    "S3_45": "orthotropic square, 45 deg (Q4)",
    "C1": "isotropic cylinder (Q4)",
    "C2": "isotropic half cylinder (Q4)",
    "C3_1": "two-material cylinder, E1/E2 = 1 (Q4)",
    "C3_10": "two-material cylinder, E1/E2 = 10 (Q4)",
    "C3_100": "two-material cylinder, E1/E2 = 100 (Q4)",
    "C3_1000": "two-material cylinder, E1/E2 = 1e3 (Q4)",
    "C3_10000": "two-material cylinder, E1/E2 = 1e4 (Q4)",
    "C3_100000": "two-material cylinder, E1/E2 = 1e5 (Q4)",
    "C4": "layered cylinder, two isotropic layers (Q4)",
    "T1": "three-cell section, isotropic (Q4)",
    "T2": "three-cell section, -45 deg orthotropic flanges, isotropic webs (Q4)",
    "T3": "T2 in Q8 elements",
    "W3": "Yu et al. multi-layer composite pipe (Q8)",
    "WTAirfoil": "wind turbine airfoil section, four materials (Q4)",
}


def example(name: str):
    home = becas_home()
    if home is None:
        raise FileNotFoundError("set BECAS_HOME to a local BECAS folder to run the BECAS example meshes")
    return read_becas(home / "examples" / name)


# ---------------------------------------------------------------------------- materials (ours)
#: (E1, E2, E3, G12, G13, G23, nu12, nu13, nu23, rho). Generic values chosen for this
#: benchmark (not a published material) except where noted.
MATERIALS: Dict[str, Tuple[float, ...]] = {
    # glass/epoxy UD, SI; transversely isotropic (G23 = E2 / (2 (1 + nu23)))
    "glass_ud": (39.0e9, 8.6e9, 8.6e9, 3.8e9, 3.8e9, 8.6e9 / 2.774, 0.28, 0.28, 0.387, 1950.0),
    # PVC-type foam core, SI
    "foam": (80e6, 80e6, 80e6, 80e6 / 2.6, 80e6 / 2.6, 80e6 / 2.6, 0.3, 0.3, 0.3, 100.0),
    # BECAS user guide table 4.1, orthotropic #1 (already used in benchmarks/vabs_becas)
    "ortho": (480.0, 120.0, 120.0, 60.0, 50.0, 60.0, 0.19, 0.26, 0.19, 1.0),
    # AS4/3501-6 graphite/epoxy, psi (the Yu et al. pipe's material, benchmarks/vabs_becas)
    "as4": (20.59e6, 1.42e6, 1.42e6, 0.87e6, 0.87e6, 0.87e6, 0.42, 0.42, 0.42, 1.0),
}


def material(name: str) -> SolidMaterial:
    return SolidMaterial(*MATERIALS[name])


def _layer(v: float, bounds: Sequence[float]) -> int:
    """Index of the layer holding the through-thickness coordinate ``v`` in ``[0, 1]``."""
    for k, b in enumerate(bounds[1:]):
        if v < b:
            return k
    return len(bounds) - 2


def _graded(fractions: Sequence[float], cells: Sequence[int]) -> Callable[[float], float]:
    """Map ``[0, 1] -> [0, 1]``, piecewise linear, giving layer ``k`` (thickness fraction
    ``fractions[k]``) exactly ``cells[k]`` uniform cells of a uniform parameter grid."""
    n = sum(cells)
    knots_p = np.concatenate([[0.0], np.cumsum(cells) / n])
    knots_t = np.concatenate([[0.0], np.cumsum(fractions) / sum(fractions)])
    return lambda p: float(np.interp(p, knots_p, knots_t))


# ---------------------------------------------------------------------------- layered cylinder
def layered_cylinder(n: int = 1):
    """Composite tube, outer radius 0.1, wall 0.01 of four equal plies of BECAS's orthotropic
    #1, outer to inner ``[45 / 0 / -30 / 90]`` (unbalanced: extension-twist and shear-bending
    couplings). Two element rows per ply, 96 around."""
    plies = [45.0, 0.0, -30.0, 90.0]
    data = lambda u, v: (1, plies[min(int(v * 4), 3)])
    return structured_mesh([arc_patch((0.0, 0.0), 0.09, 0.1, 0.0, 2 * math.pi, 96 * n, 8 * n, data)],
                           {1: material("ortho")}, "Q8")


# ---------------------------------------------------------------------------- box beams
#: Box: outer width and depth, wall thickness and plies, in inches (AS4/3501-6 in psi). The
#: proportions follow the CAS / CUS box beams of the composite-beam literature, but these are
#: this benchmark's own dimensions, not checked against any specimen.
BOX = dict(width=0.953, depth=0.537, wall=0.030, plies=6)


def _box(layup: Dict[str, Sequence[float]], n: int):
    """Rectangular box, walls in a counter-clockwise loop (bottom ``s = +y``, right ``+z``,
    top ``-y``, left ``-z``; the wall normal ``n = x cross s`` points inward). The top and
    bottom walls run the full width, their end segments forming the corners; the side walls
    sit between them. ``layup[wall]`` lists ply angles from the outer surface in."""
    W, H, t, p = BOX["width"], BOX["depth"], BOX["wall"], BOX["plies"]
    nc, nw, nh = p * n, 20 * n, 12 * n                 # corner cells (= one per ply), web, sides
    rows = p * n

    def ply(wall):
        return lambda u, v: (1, layup[wall][min(int(v * p), p - 1)])

    patches: List[Patch] = []
    # bottom wall: y from -W/2 to W/2, z from -H/2 up (inward)
    ys = [-W / 2, -W / 2 + t, W / 2 - t, W / 2]
    for (a, b), cells in zip(zip(ys[:-1], ys[1:]), (nc, nw, nc)):
        patches.append(Patch(lambda u, v, a=a, b=b: (a + u * (b - a), -H / 2 + v * t), cells, rows, ply("bottom")))
    # top wall: y from W/2 to -W/2, z from H/2 down (inward)
    ys_t = [W / 2, W / 2 - t, -W / 2 + t, -W / 2]
    for (a, b), cells in zip(zip(ys_t[:-1], ys_t[1:]), (nc, nw, nc)):
        patches.append(Patch(lambda u, v, a=a, b=b: (a + u * (b - a), H / 2 - v * t), cells, rows, ply("top")))
    # right wall: z from -H/2 + t up to H/2 - t, y from W/2 in
    patches.append(Patch(lambda u, v: (W / 2 - v * t, -H / 2 + t + u * (H - 2 * t)), nh, rows, ply("right")))
    # left wall: z from H/2 - t down, y from -W/2 in
    patches.append(Patch(lambda u, v: (-W / 2 + v * t, H / 2 - t - u * (H - 2 * t)), nh, rows, ply("left")))
    return structured_mesh(patches, {1: material("as4")}, "Q8")


def box_cus(n: int = 1, theta: float = 15.0):
    """Circumferentially uniform stiffness: every ply of every wall at ``+theta`` in the wall's
    own axes (extension-twist coupling)."""
    p = BOX["plies"]
    return _box({w: [theta] * p for w in ("top", "bottom", "left", "right")}, n)


def box_cas(n: int = 1, theta: float = 15.0):
    """Circumferentially asymmetric stiffness: top ``[+theta]_6``, bottom ``[-theta]_6`` in the
    walls' own axes (the same fibre direction in section axes, so bending-twist coupling),
    side walls ``[+theta / -theta]_3``."""
    p = BOX["plies"]
    side = [theta if k % 2 == 0 else -theta for k in range(p)]
    return _box({"top": [theta] * p, "bottom": [-theta] * p, "left": side, "right": side}, n)


# ---------------------------------------------------------------------------- MH-104-style airfoil
#: MH 104 coordinates (UIUC airfoil database, m-selig.ae.illinois.edu/ads/coord/mh104.dat).
#: The section built on them is this benchmark's own: chord 1 m, the trailing edge cut at
#: ``x_cut`` and closed by a half-circle, a two-ply skin, a spar cap on each side and two
#: sandwich webs at the cap ends. Units SI.
AIRFOIL = dict(chord=1.0, x_cut=0.90, skin=(0.001, 0.001), skin_plies=(45.0, -45.0),
               cap=0.006, cap_rows=3, cap_theta=10.0, cap_x=(0.20, 0.45),
               web=(0.0015, 0.008, 0.0015), web_cells=(1, 4, 1), web_plies=(45.0, 0.0, -45.0))


def _mh104_loop():
    rows = [ln.split() for ln in (HERE / "mh104.dat").read_text().splitlines()[1:] if ln.strip()]
    xy = np.array([[float(a), float(b)] for a, b in rows])
    le = int(np.argmin(xy[:, 0]))
    upper, lower = xy[:le + 1], xy[le:]                 # TE -> LE, LE -> TE
    xc = AIRFOIL["x_cut"]
    zu = float(np.interp(xc, upper[::-1, 0], upper[::-1, 1]))
    zl = float(np.interp(xc, lower[:, 0], lower[:, 1]))
    up = np.vstack([[xc, zu], upper[upper[:, 0] < xc]])
    lo = np.vstack([lower[lower[:, 0] < xc], [xc, zl]])
    r, zc = 0.5 * (zu - zl), 0.5 * (zu + zl)
    phi = np.linspace(-math.pi / 2, math.pi / 2, 13)[1:-1]
    te = np.column_stack([xc + r * np.cos(phi), zc + r * np.sin(phi)])
    pts = np.vstack([up, lo[1:], te]) * AIRFOIL["chord"]     # counter-clockwise, from the upper cut
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    spline = CubicSpline(s, np.vstack([pts, pts[:1]]), bc_type="periodic")
    return spline, s[-1], s[int(np.argmin(pts[:, 0]))], s[len(up) + len(lo) - 2]


def mh104(n: int = 1):
    """MH-104-style two-web blade section (see ``AIRFOIL``): skin ``[+45 / -45]`` glass UD,
    spar caps of three rows at ``+10`` deg (in the wall's own axes, so the caps couple
    extension and twist), webs ``[+45 glass / foam / -45 glass]``. Q8."""
    A = AIRFOIL
    spline, L, s_le, s_lo_cut = _mh104_loop()
    d1 = spline.derivative()

    def P(s):
        return spline(s % L)

    def N(s):                                           # inward (left) unit normal
        d = d1(s % L)
        return np.array([-d[1], d[0]]) / np.hypot(d[0], d[1])

    def point(s, depth):
        p = P(s) + depth * N(s)
        return float(p[0]), float(p[1])

    def s_at(x, upper: bool):
        f = lambda s: float(P(s)[0]) - x * A["chord"]
        return brentq(f, 0.0, s_le) if upper else brentq(f, s_le, s_lo_cut)

    t_skin = sum(A["skin"])
    tw = sum(A["web"])
    wmap = _graded(A["web"], A["web_cells"])
    wc = sum(A["web_cells"])
    # station list along the loop (counter-clockwise from the upper cut), with cells and the
    # parameter map of each segment; caps carry ("cap", side)
    xa, xb = A["cap_x"]
    su_b, su_a = s_at(xb, True), s_at(xa, True)        # upper: aft cap end first
    sl_a, sl_b = s_at(xa, False), s_at(xb, False)
    def lin(p):
        return p

    def rev(f):
        return lambda p: 1.0 - f(1.0 - p)

    segs = [  # (s0, s1, cells, map, cap side or None)
        (0.0, su_b, 30, lin, None),
        (su_b, su_b + tw, wc, wmap, "upper"),           # aft web, upper end
        (su_b + tw, su_a - tw, 16, lin, "upper"),
        (su_a - tw, su_a, wc, wmap, "upper"),           # fore web, upper end
        (su_a, sl_a, 40, lin, None),                    # leading edge
        (sl_a, sl_a + tw, wc, rev(wmap), "lower"),      # fore web, lower end
        (sl_a + tw, sl_b - tw, 16, lin, "lower"),
        (sl_b - tw, sl_b, wc, rev(wmap), "lower"),      # aft web, lower end
        (sl_b, s_lo_cut, 30, lin, None),
        (s_lo_cut, L, 10, lin, None),                   # trailing-edge half-circle
    ]
    skin_bounds = np.concatenate([[0.0], np.cumsum(A["skin"]) / t_skin])
    skin_data = lambda u, v: (1, A["skin_plies"][_layer(v, skin_bounds)])
    cap_data = lambda u, v: (1, A["cap_theta"])
    patches: List[Patch] = []
    for s0, s1, cells, m, cap in segs:
        sig = lambda u, s0=s0, s1=s1, m=m: s0 + m(u) * (s1 - s0)
        patches.append(Patch(lambda u, v, sig=sig: point(sig(u), v * t_skin), cells * n, 2 * n, skin_data))
        if cap:
            patches.append(Patch(lambda u, v, sig=sig: point(sig(u), t_skin + v * A["cap"]),
                                 cells * n, A["cap_rows"] * n, cap_data))
    # webs: u from the lower cap up to the upper cap (s = +z), v toward n = x cross s, i.e.
    # from the aft face to the fore face
    depth = t_skin + A["cap"]
    web_bounds = np.concatenate([[0.0], np.cumsum(A["web"]) / tw])
    mats = {"glass": 1, "foam": 2}

    def web_data(u, v):
        k = _layer(wmap(v), web_bounds)
        return (mats["foam"] if k == 1 else mats["glass"], A["web_plies"][k])

    for up0, lo1 in ((su_b, sl_b), (su_a - tw, sl_a + tw)):    # (aft web), (fore web)
        top = lambda v, up0=up0: np.array(point(up0 + wmap(v) * tw, depth))
        bot = lambda v, lo1=lo1: np.array(point(lo1 - wmap(v) * tw, depth))
        patches.append(Patch(lambda u, v, top=top, bot=bot: tuple((1 - u) * bot(v) + u * top(v)),
                             14 * n, wc * n, web_data))
    return structured_mesh(patches, {1: material("glass_ud"), 2: material("foam")}, "Q8")


BUILT: Dict[str, Callable] = {
    "layered_cylinder": layered_cylinder,
    "box_cus": box_cus,
    "box_cas": box_cas,
    "mh104": mh104,
}
