"""The CAS and CUS composite box beams of examples 09 and 10, built three ways.

The box is the classic coupled-composite test article: a thin graphite/epoxy box whose
plies are wound at an angle so that it either stretches and twists together (CUS) or
bends and twists together (CAS). ``CHANDRA`` follows the size of the Chandra et al. (1990)
box beams, 24.21 x 13.61 mm outside with a 0.762 mm wall of six plies, in the library's
AS4/3501-6 (not the specimens' measured properties, so the comparisons are between models
of the same box and not with the tests). ``SPAR`` is a spar-sized box for example 10.
Every builder takes the box as ``b`` (default ``CHANDRA``).

- ``thin(layup)``: ``ThinWalledSection`` on the wall mid-lines, the loop counter-clockwise
  (bottom, right, top, left), so every wall's normal z x s points into the box.
- ``solid(layup)``: the same box as a solid ``SectionMesh`` (Q8, one element row per ply).
- ``shell(layup, length)``: a Quad8 shell cantilever of the box (``box_spar``).

``layup`` maps "bottom", "right", "top" and "left" to ply angles (degrees) in each wall's
own axes, from the beam axis about the inward normal, listed from the outer surface in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from anisotropic_section_model import ThinWalledSection
from anisotropic_section_model.solid import SolidMaterial
from anisotropic_section_model.solid.structured import Patch, structured_mesh

WALLS = ("bottom", "right", "top", "left")


@dataclass(frozen=True)
class Box:
    width: float      # outer, m
    depth: float
    plies: int
    ply: float        # ply thickness, m
    material: str

    @property
    def wall(self) -> float:
        return self.plies * self.ply


#: the Chandra et al. (1990) box beams' size (example 09)
CHANDRA = Box(24.21e-3, 13.61e-3, 6, 0.127e-3, "AS4/3501-6")
#: a spar-sized box, 0.4 x 0.2 m on the mid-line, four 1 mm carbon plies (example 10)
SPAR = Box(0.404, 0.204, 4, 1.0e-3, "T300/5208")

WIDTH, DEPTH, WALL, PLIES, PLY, MATERIAL = (CHANDRA.width, CHANDRA.depth, CHANDRA.wall, CHANDRA.plies,
                                            CHANDRA.ply, CHANDRA.material)


def cus(theta: float, plies: int = PLIES) -> Dict[str, list]:
    """Circumferentially uniform stiffness: every ply of every wall at +theta in its own axes.
    The fibres wind round the box as one helix: extension-twist coupling."""
    return {w: [theta] * plies for w in WALLS}


def cas(theta: float, plies: int = PLIES) -> Dict[str, list]:
    """Circumferentially asymmetric stiffness: top [+theta]n, bottom [-theta]n in the walls'
    own axes (the same fibre direction seen from outside the beam), sides [+theta/-theta].
    Bending-twist coupling."""
    side = [theta if k % 2 == 0 else -theta for k in range(plies)]
    return {"top": [theta] * plies, "bottom": [-theta] * plies, "left": side, "right": side}


def _laminate(lib, angles: Sequence[float], laminate_cls, b: Box):
    # thin-wall laminates stack from the -n face up; -n is the outer surface here
    return laminate_cls.from_angles(lib.get(b.material), list(angles), b.ply)


def thin(lib, layup, per_wall: int = 8, b: Box = CHANDRA) -> ThinWalledSection:
    from laminate_constitutive_model import Laminate

    bx = b
    w, h = bx.width - bx.wall, bx.depth - bx.wall
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    segs = []
    for k, name in enumerate(WALLS):
        a, b = np.array(corners[k]), np.array(corners[(k + 1) % 4])
        lam = _laminate(lib, layup[name], Laminate, bx)
        segs += [(a + (b - a) * i / per_wall, a + (b - a) * (i + 1) / per_wall, lam, name) for i in range(per_wall)]
    return ThinWalledSection.from_segments(segs)


def solid(lib, layup, refine: int = 1, b: Box = CHANDRA):
    """Solid mesh: walls in a counter-clockwise loop, v running from the outer surface in."""
    W, H, t, p = b.width, b.depth, b.wall, b.plies
    mat = {1: SolidMaterial.from_material_card(lib.get(b.material))}
    nc, nw, nh, rows = max(p, 4) * refine, 20 * refine, 12 * refine, p * refine

    def ply(wall):
        return lambda u, v: (1, layup[wall][min(int(v * p), p - 1)])

    patches = []
    ys = [-W / 2, -W / 2 + t, W / 2 - t, W / 2]
    for (a, b), cells in zip(zip(ys[:-1], ys[1:]), (nc, nw, nc)):
        patches.append(Patch(lambda u, v, a=a, b=b: (a + u * (b - a), -H / 2 + v * t), cells, rows, ply("bottom")))
    ys = ys[::-1]
    for (a, b), cells in zip(zip(ys[:-1], ys[1:]), (nc, nw, nc)):
        patches.append(Patch(lambda u, v, a=a, b=b: (a + u * (b - a), H / 2 - v * t), cells, rows, ply("top")))
    patches.append(Patch(lambda u, v: (W / 2 - v * t, -H / 2 + t + u * (H - 2 * t)), nh, rows, ply("right")))
    patches.append(Patch(lambda u, v: (-W / 2 + v * t, H / 2 - t - u * (H - 2 * t)), nh, rows, ply("left")))
    return structured_mesh(patches, mat, "Q8")


def shell(lib, layup, length: float, n_width: int = 8, n_depth: int = 4, n_span: int = 40, b: Box = CHANDRA):
    """Quad8 FSDT shell cantilever of the box along z (``box_spar``). shell-fem's elements
    here have outward normals, so a ply at +theta about the inward normal is -theta about
    the outward one, and the stack is read from the inside out: the shell gets
    ``[-a for a in reversed(angles)]``. ``ThinWalledSection.from_blade_mesh`` of it gives
    the same section as ``thin``."""
    from shell_fem import Laminate

    from _shared.torsion import box_spar

    names = {"cap_bottom": "bottom", "web_te": "right", "cap_top": "top", "web_le": "left"}
    lams = {k: _laminate(lib, [-a for a in reversed(layup[v])], Laminate, b) for k, v in names.items()}
    return box_spar(b.width - b.wall, b.depth - b.wall, length, lams, n_width, n_depth, n_span)
