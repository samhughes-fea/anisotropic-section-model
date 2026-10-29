"""A blade from its implicit section geometry: cut it anywhere, mesh it, get its section.

``BladeGeometry`` lofts one of the section geometry's 40 structural layouts (here 2D-F:
two shear webs and continuous box spar caps between them) between spanwise stations.
``section_at(z)`` cuts the walls at any span position. The same cut feeds both
downstream models:

1. ``ThinWalledSection.from_geometry`` builds the thin-walled beam section directly
   from the cut, with no mesh.
2. ``BladeBuilder.from_geometry`` meshes the walls' mid-surfaces along the span as
   curved Quad8 FSDT shells.

The example checks both against the region-based ``BladeBuilder`` model of composite-laminate-model's example 20
on the same prismatic section (NACA 0024, 1 m chord, glass skins and webs, carbon spar
caps), then follows a tapered, twisted segment along its span, and draws a few other
layouts from the catalogue.

Figures: the tapered segment's sections and stiffness along the span; the four routes
on the example 20 section; six layouts with their torsional and axial stiffness.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import SERIES, TEXT_SECONDARY, plt, save
from anisotropic_section_model import ThinWalledSection
from implicit_geometry_kernel import Airfoil, BladeGeometry, ShearWeb, SkinRegion, Station
from composite_laminate_model import BladeBuilder, FSDTModel, Laminate, beam_section_properties, create_default_library

DATA = Path(__file__).resolve().parent / "data" / "results.json"
SKIN_T, CAP_T, WEB_T = 0.003, 0.008, 0.002  # m: wall thicknesses, which place the walls
WEBS_AT = (0.18, 0.48)  # chord fractions, as in example 20
LENGTH = 6.0
KIND_COLOURS = {"skin": SERIES[0], "cap": SERIES[1], "web": SERIES[2]}
GALLERY = ["0A", "1C-F", "2B-F", "2D-F", "3C-CN", "5D-F"]


def laminates(lib):
    glass, carbon = lib.get("E-glass/epoxy"), lib.get("T300/5208")
    return {
        "skin": Laminate.from_angles(glass, [0, 45, -45, -45, 45, 0], 0.5e-3),
        "web": Laminate.from_angles(glass, [45, -45, -45, 45], 0.5e-3),
        "cap": Laminate.from_angles(carbon, [0] * 16, 0.5e-3),
    }


def prismatic():
    af = Airfoil.naca4("0024")
    return [Station(0.0, 1.0, af, pitch_axis=0.3), Station(LENGTH, 1.0, af, pitch_axis=0.3)]


def section_numbers(sec):
    return {"EA": sec.EA, "EI_x": sec.EI_xx, "EI_y": sec.EI_yy, "GJ": sec.GJ,
            "shear_centre_x": float(sec.shear_centre[0])}


def beam_numbers(props):
    return {"EA": props.EA, "EI_x": props.EI_x, "EI_y": props.EI_y, "GJ": props.GJ,
            "shear_centre_x": float(props.shear_centre[0])}


def routes(lams):
    """The example 20 section from the geometry and from example 20's regions, thin wall and shell."""
    geo = BladeGeometry(prismatic(), "2D-F", SKIN_T, CAP_T, WEB_T, web_positions=WEBS_AT)
    shell_geo = BladeBuilder.from_geometry(geo, n_span=12).build(lams, FSDTModel())
    regions = [SkinRegion("le_panel", 0.0, WEBS_AT[0], 8), SkinRegion("spar_cap", *WEBS_AT, 12),
               SkinRegion("te_panel", WEBS_AT[1], 1.0, 16)]
    webs = [ShearWeb("web_fwd", WEBS_AT[0], 6), ShearWeb("web_aft", WEBS_AT[1], 6)]
    # in the region model the spar cap region carries the cap and skin stacked
    region_lams = {"le_panel": lams["skin"], "te_panel": lams["skin"], "web_fwd": lams["web"],
                   "web_aft": lams["web"], "spar_cap": Laminate(lams["cap"].plies + lams["skin"].plies)}
    shell_regions = BladeBuilder(prismatic(), regions, webs, n_span=12).build(region_lams, FSDTModel())
    z = LENGTH / 2
    return {
        "thin wall": {
            "from the cut (from_geometry)": section_numbers(ThinWalledSection.from_geometry(geo.section_at(z), lams)),
            "from the geometry's mesh": section_numbers(ThinWalledSection.from_blade_mesh(shell_geo, z=z)),
            "from the region mesh": section_numbers(ThinWalledSection.from_blade_mesh(shell_regions, z=z)),
        },
        "shell (beam_section_properties)": {
            "geometry mesh": beam_numbers(beam_section_properties(shell_geo)),
            "region mesh": beam_numbers(beam_section_properties(shell_regions)),
        },
        "mesh": {"geometry": [len(shell_geo.mesh.node_ids), len(shell_geo.mesh.element_ids)],
                 "regions": [len(shell_regions.mesh.node_ids), len(shell_regions.mesh.element_ids)]},
    }


def tapered(lams):
    """A tapered, twisted segment with straight webs and spar caps thinning towards the tip."""
    stations = [Station(0.0, 2.0, Airfoil.naca4("0030"), twist=10.0, pitch_axis=0.3),
                Station(10.0, 1.0, Airfoil.naca4("0018"), twist=0.0, pitch_axis=0.3)]
    cap_t = lambda z: 0.012 - 0.0005 * z  # noqa: E731  12 mm at the root to 7 mm at the tip
    geo = BladeGeometry(stations, "2D-F", SKIN_T, cap_t, 0.004, web_offsets=(0.3, 0.75))
    carbon = create_default_library().get("T300/5208")
    cap = lambda z: Laminate.from_angles(carbon, [0] * 16, cap_t(z) / 16)  # noqa: E731
    lam = dict(lams, cap=cap)
    blade = BladeBuilder.from_geometry(geo, n_span=10).build(lam, FSDTModel())
    zs = np.asarray(blade.span_positions, float)
    out = {"z": zs.tolist(), "sections": {}, "geometry": {}, "mesh": {},
           "elements": len(blade.mesh.element_ids)}
    for i, z in enumerate(zs):
        cut = geo.section_at(float(z))
        # the shell's laminates are constant over each element and taken at its middle; the
        # mesh section at z reads the element below it (above it at the root), so the
        # direct section takes that element's cap for a like-for-like check of the mesh
        z_element = 0.5 * (zs[max(i, 1) - 1] + zs[max(i, 1)])
        direct = ThinWalledSection.from_geometry(cut, dict(lams, cap=cap(z_element)), thickness_tol=0.05)
        meshed = ThinWalledSection.from_blade_mesh(blade, z=float(z))
        for name, sec in (("geometry", direct), ("mesh", meshed)):
            for k, v in section_numbers(sec).items():
                out[name].setdefault(k, []).append(v)
        if z in (0.0, 5.0, 10.0):
            out["sections"][f"{z:g}"] = {
                "walls": [{"kind": w.kind, "xy": np.round(w.xy, 5).tolist()} for w in cut.walls],
                "cells": [np.round(c.points[:, :2], 5).tolist() for c in cut.cells],
                "shear_centre": [float(v) for v in direct.shear_centre],
                "chord": cut.chord, "twist": cut.twist,
            }
    return out


def gallery(lams):
    out = {}
    for key in GALLERY:
        cut = BladeGeometry(prismatic(), key, SKIN_T, CAP_T, 0.004).section_at(0.0)
        sec = ThinWalledSection.from_geometry(cut, lams)
        out[key] = {"walls": [{"kind": w.kind, "xy": np.round(w.xy, 5).tolist()} for w in cut.walls],
                    "cells": [np.round(c.points[:, :2], 5).tolist() for c in cut.cells], "GJ": sec.GJ, "EA": sec.EA,
                    "n_cells": sec.n_cells}
    return out


def draw_section(ax, s, title):
    for poly in s["cells"]:
        P = np.asarray(poly)
        ax.fill(P[:, 0], P[:, 1], color=SERIES[0], alpha=0.06, lw=0)
    for w in s["walls"]:
        P = np.asarray(w["xy"])
        ax.plot(P[:, 0], P[:, 1], color=KIND_COLOURS[w["kind"]], lw=1.6 if w["kind"] != "cap" else 2.6)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9.5)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def figure_tapered(t):
    fig = plt.figure(figsize=(10, 6.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15])
    for i, (z, s) in enumerate(t["sections"].items()):
        ax = fig.add_subplot(grid[0, i])
        draw_section(ax, s, f"z = {z} m: chord {s['chord']:.2f} m, twist {s['twist']:.1f}°")
        ax.plot(*s["shear_centre"], "+", color=TEXT_SECONDARY, ms=8)
    ax = fig.add_subplot(grid[1, :])
    z = np.asarray(t["z"])
    for k, c in zip(("EA", "EI_x", "EI_y", "GJ"), SERIES):
        v = np.asarray(t["geometry"][k])
        ax.semilogy(z, v / v[-1], "-", color=c, label=f"{k}, from_geometry")
        ax.semilogy(z, np.asarray(t["mesh"][k]) / v[-1], "o", color=c, ms=4, mfc="none")
    ax.plot([], [], "o", color=TEXT_SECONDARY, mfc="none", ms=4, label="from the geometry's shell mesh")
    ax.set_xlabel("span position z [m]")
    ax.set_ylabel("stiffness / tip value")
    ax.set_title("Thin-walled section along the tapered segment")
    ax.legend(ncol=3, fontsize=8)
    handles = [plt.Line2D([], [], color=c, lw=2, label=k) for k, c in KIND_COLOURS.items()]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=8.5)
    fig.tight_layout()
    save(fig, __file__, "tapered_segment.png")


def figure_routes(r):
    keys = ["EA", "EI_x", "EI_y", "GJ"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    groups = [("thin wall", "from the region mesh", "Thin-walled section, against the region mesh's"),
              ("shell (beam_section_properties)", "region mesh", "Shell model, against the region mesh")]
    for ax, (group, ref_name, title) in zip(axes, groups):
        ref = r[group][ref_name]
        others = [n for n in r[group] if n != ref_name]
        width = 0.8 / len(others)
        for i, (name, c) in enumerate(zip(others, SERIES)):
            dv = [100 * (r[group][name][k] / ref[k] - 1) for k in keys]
            ax.bar(np.arange(len(keys)) + (i - 0.5 * (len(others) - 1)) * width, dv, width, color=c, label=name)
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_xticks(range(len(keys)), keys)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(title, fontsize=9.5)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("difference [%]")
    fig.tight_layout()
    save(fig, __file__, "routes.png")


def figure_gallery(g):
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.2))
    for ax, (key, s) in zip(axes.ravel(), g.items()):
        cells = f"{s['n_cells']} cell{'s' if s['n_cells'] > 1 else ''}"
        draw_section(ax, s, f"{key}: {cells}, GJ {s['GJ'] / 1e6:.2f} MN m², EA {s['EA'] / 1e6:.0f} MN")
    fig.tight_layout()
    save(fig, __file__, "layouts.png")


def report(r, t, g):
    print("Example 20 section (NACA 0024, 1 m chord):")
    for group, ref_name in (("thin wall", "from the region mesh"), ("shell (beam_section_properties)", "region mesh")):
        ref = r[group][ref_name]
        print(f"  {group}, against the {ref_name.replace('from the ', '')}:")
        for name, v in r[group].items():
            print(f"    {name}: " + ", ".join(f"{k} {v[k]:.4g} ({v[k] / ref[k] - 1:+.2%})"
                                          for k in ("EA", "EI_x", "EI_y", "GJ"))
                  + f", shear centre x {1e3 * v['shear_centre_x']:.1f} mm")
    print(f"  meshes (nodes, elements): geometry {r['mesh']['geometry']}, regions {r['mesh']['regions']}")
    err = {k: max(abs(m / d - 1) for m, d in zip(t["mesh"][k], t["geometry"][k])) for k in t["geometry"]
           if k != "shear_centre_x"}
    print(f"Tapered segment ({t['elements']} elements), root / tip: " + ", ".join(
        f"{k} {t['geometry'][k][0] / t['geometry'][k][-1]:.2f}" for k in err)
        + "; largest mesh vs from_geometry difference: " + ", ".join(f"{k} {v:.2%}" for k, v in err.items()))
    print("Layouts: " + ", ".join(f"{k} GJ {v['GJ']:.3g} EA {v['EA']:.3g}" for k, v in g.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", action="store_true", help="redraw from data/results.json")
    args = parser.parse_args()
    if args.reuse and DATA.exists():
        data = json.loads(DATA.read_text())
    else:
        lams = laminates(create_default_library())
        data = {"routes": routes(lams), "tapered": tapered(lams), "gallery": gallery(lams)}
        DATA.parent.mkdir(exist_ok=True)
        DATA.write_text(json.dumps(data))
    report(data["routes"], data["tapered"], data["gallery"])
    figure_tapered(data["tapered"])
    figure_routes(data["routes"])
    figure_gallery(data["gallery"])


if __name__ == "__main__":
    main()
