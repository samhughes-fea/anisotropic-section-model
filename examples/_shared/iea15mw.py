"""The IEA 15 MW reference blade as PreComp sections, rebuilt as shell models.

``examples/_shared/data/iea15mw_sections.json`` holds six
stations of the published blade layup as PreComp sees them: the airfoil, sector
boundaries on each surface, the laminae of each sector and the webs. These helpers
turn a station into ``BladeBuilder`` regions, webs and laminates. Example 03 uses them;
the data are those of composite-laminate-model's example 24.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from composite_laminate_model import (Airfoil, BladeBuilder, FSDTModel, Laminate, Ply, ShearWeb,
    SkinRegion, Station)
from composite_laminate_model.laminate_materials import MaterialCard
from composite_laminate_model.laminate_materials_3d import ElasticProperties3D, StrengthProperties3D

DATA_FILE = Path(__file__).resolve().parent / "data" / "iea15mw_sections.json"
DATA = json.loads(DATA_FILE.read_text())
TE_CLOSE = 0.01  # trailing-edge gaps below this fraction of the chord are closed


def material_cards(materials):
    """PreComp materials are 2D orthotropic (E1, E2, G12, nu12); the shell also needs
    transverse shear moduli, taken as transversely isotropic with nu23 = 0.3."""
    big = 1.0e12  # strengths are not used here
    return {name: MaterialCard(name,
                               ElasticProperties3D.transversely_isotropic(m["E1"], m["E2"], m["nu12"], m["G12"], 0.3),
                               StrengthProperties3D(big, big, big, big, big), density=m["rho"])
            for name, m in materials.items()}


CARDS = material_cards(DATA["materials"])


def laminate(laminae):
    """PreComp lists laminae from the outer surface inward; a Laminate runs from the
    bottom face (inside the section, for an outward normal) to the top face."""
    return Laminate([Ply(CARDS[m], t, angle) for m, t, angle in reversed(laminae) if t > 0.0])


def airfoil(st):
    """The PreComp profile as an Airfoil.

    PreComp closes a thin blunt trailing edge by letting one surface step almost
    vertically onto the other over its last point. Here that last point is removed,
    so the gap stays open and ``BladeBuilder`` closes it with a flat trailing-edge web
    (given the last sector's laminate in ``build_segment``). A gap under 1% of the
    chord is closed to a sharp trailing edge instead.
    """
    p = st["profile"]
    x = np.array(p["x"])
    yu, yl = np.array(p["y_upper"]), np.array(p["y_lower"])
    xu = xl = x
    jump = lambda y: abs(y[-1] - y[-2]) > 5.0 * abs(y[-2] - y[-3])  # noqa: E731 - a closing step, not the surface
    if jump(yu):
        xu, yu = x[:-1], yu[:-1]
    elif jump(yl):
        xl, yl = x[:-1], yl[:-1]
    # A gap under 1% of the chord is closed: a web that short would be a sliver of an element,
    # hundreds of times longer (along the span) than it is high, and it adds almost no stiffness.
    if abs(yu[-1] - yl[-1]) < TE_CLOSE:
        n = min(len(yu), len(yl))  # end both surfaces at the same point
        xu = xl = x[:n]
        yu, yl = yu[:n], yl[:n]
        half_gap = 0.5 * (yu[-1] - yl[-1])
        ramp = np.clip((xu - 0.9) / (xu[-1] - 0.9), 0.0, 1.0)  # pinch over the last 10% of the chord
        yu, yl = yu - half_gap * ramp, yl + half_gap * ramp
    pts = np.vstack([np.column_stack([xu[::-1], yu[::-1]]), np.column_stack([xl[1:], yl[1:]])])
    return Airfoil(pts, f"IEA 15 MW r = {st['r']:.1f} m")


MERGE_CHORD = 0.003  # sector boundaries closer than this (in chords) are merged: slivers between
# them, much narrower than the skin is thick, turn inside out when CalculiX expands
# the mid-surface shells into bricks


def build_segment(st, length, n_span, h_perimeter, element, tip_chord=None, twist=(0.0, 0.0), **builder_options):
    """A shell segment with the section of station ``st``, meshed on its PreComp sector boundaries.

    The segment runs ``length`` along z with the station's airfoil and layup; the chord
    tapers linearly to ``tip_chord`` (prismatic by default) and the twist goes from
    ``twist[0]`` to ``twist[1]`` degrees. ``h_perimeter`` is the chordwise element size
    as a fraction of the section perimeter. Other keywords go to ``BladeBuilder``
    (``mesh_surface="mid"``, for example).
    """
    chord, af = st["chord"], airfoil(st)
    webs_in = [(x, lam) for x, lam in zip(st["webs"]["loc"], st["webs"]["laminae"]) if x < 0.999]
    te_web = [lam for x, lam in zip(st["webs"]["loc"], st["webs"]["laminae"]) if x >= 0.999]

    # Region boundaries: every sector edge on either surface and every web
    edges = np.unique(np.round(np.concatenate([st["upper"]["loc"], st["lower"]["loc"], [x for x, _ in webs_in]]), 6))
    bounds = [edges[0]]
    for v in edges[1:]:
        if v - bounds[-1] > MERGE_CHORD:
            bounds.append(v)
    bounds[-1] = 1.0

    xs = np.linspace(0.0, 1.0, 2001)
    arcs = {side: np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(af.surface(side, xs))))])
            for side in ("upper", "lower")}
    h = h_perimeter * (arcs["upper"][-1] + arcs["lower"][-1])  # element size in chords

    regions, laminates = [], {}
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        arc_length = max(np.interp(b, xs, arcs[s]) - np.interp(a, xs, arcs[s]) for s in arcs)
        name = f"sector{k}"
        regions.append(SkinRegion(name, float(a), float(b), max(1, int(np.ceil(arc_length / h)))))
        for side, tag in (("upper", "ss"), ("lower", "ps")):
            loc = np.asarray(st[side]["loc"])
            j = int(np.clip(np.searchsorted(loc, 0.5 * (a + b)) - 1, 0, len(loc) - 2))
            laminates[f"{name}_{tag}"] = laminate(st[side]["laminae"][j])
    webs = []
    for k, (x, lam) in enumerate(webs_in):
        position = bounds[int(np.argmin(np.abs(np.asarray(bounds) - x)))]
        height = af.surface("upper", x) - af.surface("lower", x)
        webs.append(ShearWeb(f"web{k}", float(position), max(2, int(np.ceil(height / h)))))
        laminates[f"web{k}"] = laminate(lam)
    gap = abs(af.surface("upper", 1.0) - af.surface("lower", 1.0))
    if te_web:
        laminates["te_web"] = laminate(te_web[0])
    elif gap > 1e-9:  # PreComp closes the gap with the last sector (the same on both surfaces here)
        laminates["te_web"] = laminate(st["upper"]["laminae"][-1])
    stations = [Station(0.0, chord, af, twist=twist[0], pitch_axis=st["pitch_axis"]),
                Station(length, chord if tip_chord is None else tip_chord, af, twist=twist[1],
                        pitch_axis=st["pitch_axis"])]
    builder = BladeBuilder(stations, regions, webs, n_span=n_span, element=element,
                           te_web_elements=max(1, int(np.ceil(gap / h))), **builder_options)
    return builder.build(laminates, FSDTModel())


__all__ = ["DATA", "CARDS", "material_cards", "laminate", "airfoil", "build_segment"]
