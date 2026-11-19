"""The benchmark sections, meshed for ``SolidSectionAnalysis``.

Each builder takes a refinement level ``n`` (1 = the source's own mesh density where it is
known) and an element ``kind`` (``"Q4"``, ``"Q8"`` or ``"T6"``) and returns a ``SectionMesh``
in the source's section coordinates, which are this package's ``(y, z)`` (see
``reference.py``). Plies follow the source's layup; ``theta`` is this package's fibre angle
(from the beam axis toward the wall direction ``s``, ``material.py``).
"""

from __future__ import annotations

import math
from typing import Callable, Dict

from anisotropic_section_model.solid import SolidMaterial
from anisotropic_section_model.solid.structured import arc_patch, rectangle_patch, structured_mesh

from reference import MATERIALS


def material(name: str) -> SolidMaterial:
    return SolidMaterial(*MATERIALS[name])


# ---------------------------------------------------------------- BECAS user guide squares
# 0.1 x 0.1 square centred on the origin; BECAS's (and GXBeam's) mesh is 10 x 10 Q4.
def _square(n: int, kind: str, data, mats) -> "SectionMesh":
    m = 10 * n
    return structured_mesh([rectangle_patch(-0.05, -0.05, 0.1, 0.1, m, m, data)], mats, kind)


def s1(n: int = 1, kind: str = "Q8"):
    return _square(n, kind, lambda u, v: (1, 0.0), {1: material("iso1")})


def s2(n: int = 1, kind: str = "Q8"):
    # isotropic #1 for x < 0, isotropic #2 (alpha = 10) for x > 0
    return _square(n, kind, lambda u, v: (1 if u < 0.5 else 2, 0.0),
                   {1: material("iso1"), 2: material("iso2")})


def s3(theta: float) -> Callable:
    def build(n: int = 1, kind: str = "Q8"):
        return _square(n, kind, lambda u, v: (1, theta), {1: material("ortho")})
    return build


# ---------------------------------------------------------------- BECAS user guide cylinders
# Outer radius 0.1, wall 0.01. GXBeam's reproduction uses 3 x 119 Q4 (C1) and 3 x 59 (C2);
# BECAS's own mesh is not stated.
def c1(n: int = 1, kind: str = "Q8"):
    return structured_mesh([arc_patch((0.0, 0.0), 0.09, 0.1, 0.0, 2 * math.pi, 119 * n, 3 * n,
                                      lambda u, v: (1, 0.0))], {1: material("iso1")}, kind)


def c2(n: int = 1, kind: str = "Q8"):
    # the half x < 0
    return structured_mesh([arc_patch((0.0, 0.0), 0.09, 0.1, math.pi / 2, 3 * math.pi / 2, 59 * n, 3 * n,
                                      lambda u, v: (1, 0.0))], {1: material("iso1")}, kind)


# ---------------------------------------------------------------- VABS: Chen et al. aluminium tube
# Outer radius 0.3 m, wall 0.2 m (thickness ratio t / D = 1/3).
def tube(n: int = 1, kind: str = "Q8"):
    return structured_mesh([arc_patch((0.0, 0.0), 0.1, 0.3, 0.0, 2 * math.pi, 48 * n, 8 * n,
                                      lambda u, v: (1, 0.0))], {1: material("aluminium")}, kind)


# ---------------------------------------------------------------- VABS: multi-layer composite pipe
# Two flat walls (x2 from -1 to 1 in, x3 from 0.3 to 0.5 and -0.5 to -0.3) closed by two
# half-circles of radius 0.3 to 0.5 centred on (+-1, 0). Two plies of 0.1 in each: the flat
# walls [90 inner / 0 outer], the half-circles [45 inner / -45 outer], the angles measured in
# the walls' own axes with the wall direction running clockwise round the section (as VABS's
# and GXBeam's meshes of it). The wall normal n = x cross s then points outward.
def pipe_layup_sign() -> float:
    """``theta`` here = sign * the source's ply angle (see ``README.md``, "Fibre angles")."""
    return 1.0


def pipe(n: int = 1, kind: str = "Q8", sign: float = None):
    sg = pipe_layup_sign() if sign is None else sign
    flat = lambda u, v: (1, sg * (90.0 if v < 0.5 else 0.0))
    arc = lambda u, v: (1, sg * (45.0 if v < 0.5 else -45.0))
    nf, na, nt = 16 * n, 12 * n, 4 * n
    top = rectangle_patch(-1.0, 0.3, 2.0, 0.2, nf, nt, flat)                   # s = +x2, n outward (+x3)
    bottom = rectangle_patch(-1.0, -0.3, 2.0, 0.2, nf, nt, flat)
    bottom = type(bottom)(lambda u, v: (1.0 - 2.0 * u, -0.3 - 0.2 * v), nf, nt, flat)  # s = -x2, n = -x3
    right = arc_patch((1.0, 0.0), 0.3, 0.5, math.pi / 2, -math.pi / 2, na, nt, arc)    # clockwise
    left = arc_patch((-1.0, 0.0), 0.3, 0.5, 3 * math.pi / 2, math.pi / 2, na, nt, arc)
    return structured_mesh([top, right, bottom, left], {1: material("pipe")}, kind)


CASES: Dict[str, Callable] = {
    "S1": s1, "S2": s2, "S3_0": s3(0.0), "S3_22.5": s3(22.5), "S3_90": s3(90.0),
    "C1": c1, "C2": c2, "tube": tube, "pipe": pipe,
}
