# pre_processing/section_integration/thin_walled/from_geometry.py
"""Thin-walled sections cut from an implicit-geometry-kernel blade.

``implicit_geometry_kernel.BladeGeometry.section_at(z)`` returns a ``SectionCut``: the
wall midlines of a blade section with their thickness, kind (skin, cap, web) and region
label (``"skin_suction_surface:bay_0"``, ``"spar_cap_pressure_surface"``,
``"shear_web_1"``). ``thin_walled_from_cut`` turns it into a ``ThinWalledSection`` and
its laminates, using the cut's mid-surface graph (``SectionCut.mid_surface()``):

- each straight midline segment becomes one wall;
- a spar cap is merged with the skin it is bonded to, so along the bond the wall is
  the cap and skin stacked on their combined mid-surface (cap plies inside, skin plies
  outside), as a cap laid on a skin is in a shell model;
- walls take their laminate by label: the full label, then its region (the label
  without the ``:bay_k`` part), then its kind (``"skin"``, ``"cap"``, ``"web"``).

Axes. The blade frame of the geometry has z along the span, x along the chord and y
through the thickness. The beam axis is the span, so the section coordinates are
``(y_s, z_s) = (x, y)`` of the blade frame, a cyclic relabelling that keeps the axes
right-handed. By default they are measured from the cut's pitch-axis point
(``origin="pitch_axis"``: the station offset, prebend and sweep, is taken off), which
is where a blade beam's reference axis runs; ``origin="blade"`` keeps blade-frame
coordinates.

Laminate normals. The thin-walled walls stack their plies along ``e_n = x cross e_s``.
The mid-surface segments are oriented so that this normal points out of the section on
skins and caps and towards the trailing edge on webs. So laminates list their plies
from the inside of the section out (skins, caps) and from the leading-edge face to the
trailing-edge face (webs).

Thicknesses. The geometry's skin and cap thicknesses place the walls (skins grow
inward from the outer mould line, caps sit inside the skin), so each skin and cap
laminate must match its wall within ``thickness_tol``. Webs end on the skin midlines
whatever their thickness, so a web laminate may differ.

The geometry kernel is an optional dependency (Python 3.11 or later):
``pip install "implicit-geometry-kernel @ git+https://github.com/samhughes-fea/implicit-geometry-kernel"``.
Only the cut object is used here, so this module imports nothing from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np

from pre_processing.section_integration.thin_walled.laminate import WallLaminate
from pre_processing.section_integration.thin_walled.thin_walled_geometry import ThinWalledSection, Wall

__all__ = ["GeometrySection", "thin_walled_from_cut", "laminate_lookup"]

LaminateMap = Mapping[str, Union[WallLaminate, int]]


@dataclass(frozen=True)
class GeometrySection:
    """A thin-walled section built from a ``SectionCut``.

    Attributes
    ----------
    section : ThinWalledSection
        Walls numbered from 1 in the order of the cut's mid-surface segments.
    laminates : dict int -> WallLaminate
        One id per distinct wall stack (a skin laminate, a cap-on-skin stack, ...).
    wall_names : dict int -> str
        The geometry label each wall is named after (a cap label along a bond).
    wall_layers : dict int -> tuple of str
        The geometry labels stacked in each wall, inside first.
    stack_names : dict int -> str
        Laminate id to its stack, such as ``"spar_cap_suction_surface+skin_suction_surface:bay_1"``.
    origin : (float, float)
        Blade-frame (x, y) point the section coordinates are measured from.
    z : float
        Span position of the cut.
    """

    section: ThinWalledSection
    laminates: Dict[int, WallLaminate]
    wall_names: Dict[int, str]
    wall_layers: Dict[int, Tuple[str, ...]]
    stack_names: Dict[int, str]
    origin: Tuple[float, float]
    z: float

    def analysis(self, hoop: str = "free", n_sub: int = 1):
        """``ThinWalledSectionAnalysis`` of the section (``hoop="free"`` matches the 2D and shell models)."""
        from pre_processing.section_integration.thin_walled.section_analysis import ThinWalledSectionAnalysis

        return ThinWalledSectionAnalysis(self.section, self.laminates, hoop=hoop, n_sub=n_sub)


def laminate_lookup(wall, laminates: LaminateMap, library: Optional[Mapping[int, WallLaminate]] = None) -> WallLaminate:
    """The laminate of a ``SectionWall``: by label, then region, then kind.

    ``laminates`` maps those keys to a ``WallLaminate`` or to a laminate id in
    ``library`` (as ``region_laminate.txt`` maps them to ``laminate.txt`` ids).
    """
    keys = tuple(dict.fromkeys((wall.label, wall.label.split(":")[0], wall.kind)))
    for key in keys:
        if key in laminates:
            lam = laminates[key]
            if isinstance(lam, WallLaminate):
                return lam
            if library is None or int(lam) not in library:
                raise KeyError(f"region '{key}' names laminate {lam}, which is not defined")
            return library[int(lam)]
    raise KeyError(f"no laminate for wall '{wall.label}' (tried {', '.join(map(repr, keys))})")


def thin_walled_from_cut(
    cut,
    laminates: LaminateMap,
    library: Optional[Mapping[int, WallLaminate]] = None,
    thickness_tol: float = 0.01,
    origin: str = "pitch_axis",
    section_id: int = 0,
) -> GeometrySection:
    """``GeometrySection`` of an implicit-geometry-kernel ``SectionCut``.

    Parameters
    ----------
    cut : SectionCut
        ``BladeGeometry.section_at(z)``.
    laminates : mapping
        Label, region or kind to a ``WallLaminate`` (or to an id in ``library``).
    library : mapping int -> WallLaminate, optional
        Laminates by id, for maps that name ids.
    thickness_tol : float
        Allowed relative difference between a skin or cap laminate and its wall.
    origin : {"pitch_axis", "blade"}
        Measure section coordinates from the pitch-axis point or the blade-frame origin.
    section_id : int

    Raises
    ------
    KeyError
        A wall with no laminate (the message names the wall and the keys tried).
    ValueError
        A skin or cap laminate thicker or thinner than its wall.
    """
    if origin not in ("pitch_axis", "blade"):
        raise ValueError(f"origin must be 'pitch_axis' or 'blade', got {origin!r}")
    mid = cut.mid_surface()
    chosen: Dict[str, WallLaminate] = {}

    def laminate_for(label: str) -> WallLaminate:
        if label not in chosen:
            wall = cut.wall(label)
            lam = laminate_lookup(wall, laminates, library)
            if wall.kind != "web" and abs(lam.thickness - wall.thickness) > thickness_tol * wall.thickness:
                raise ValueError(
                    f"wall '{wall.label}' is {wall.thickness * 1e3:.4g} mm thick in the geometry but its laminate "
                    f"is {lam.thickness * 1e3:.4g} mm (thickness_tol {thickness_tol:g})"
                )
            chosen[label] = lam
        return chosen[label]

    o = np.zeros(2) if origin == "blade" else np.asarray(cut.offset, float)
    points = {k + 1: tuple(p - o) for k, p in enumerate(np.asarray(mid.points, float))}
    stack_id: Dict[Tuple[str, ...], int] = {}
    lams: Dict[int, WallLaminate] = {}
    stack_names: Dict[int, str] = {}
    walls, names, layers = [], {}, {}
    used = set()
    for k, seg in enumerate(mid.segments, start=1):
        if seg.layers not in stack_id:
            parts = [laminate_for(label) for label in seg.layers]
            lam = parts[0] if len(parts) == 1 else WallLaminate(
                [ply for part in parts for ply in part.plies], shear_correction=parts[-1].shear_correction)
            lid = len(stack_id) + 1
            stack_id[seg.layers] = lid
            lams[lid] = lam
            stack_names[lid] = "+".join(seg.layers)
        walls.append(Wall(k, seg.start + 1, seg.end + 1, stack_id[seg.layers]))
        names[k] = seg.name
        layers[k] = tuple(seg.layers)
        used.update((seg.start + 1, seg.end + 1))
    points = {pid: p for pid, p in points.items() if pid in used}
    section = ThinWalledSection(points, walls, lams.keys(), section_id=section_id)
    return GeometrySection(section, lams, names, layers, stack_names, (float(o[0]), float(o[1])), float(cut.z))
