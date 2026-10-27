"""Bending and axial load on a blade section, layer by layer: shear, shear lag, distortion.

Example 30 took bending and axial load beyond Euler-Bernoulli on a rectangular box.
This example does the same on the three-cell blade section of examples 20 and 28
(NACA 0024, 1 m chord: glass skins, carbon spar caps, two glass shear webs), where
the caps are much stiffer than the skins and the section is far from rigid in its
plane. Each layer is checked against a shell model of the same blade (curved Quad8
FSDT, skins on their mid-surface):

1. Euler-Bernoulli and Timoshenko: a flapwise tip force through a tip rib, with the
   4 x 4 and the 6 x 6 section stiffness (thin-wall theory and the shell's own 6 x 6
   from ``beam_section_properties``), solved with ``TimoshenkoBeam``.
2. Shear lag (GBT warping modes): the axial stress across the spar caps near the
   clamped root, where plane sections put a straight line.
3. Distortion (GBT distortional modes): a suction on the upper skin, which no
   rigid-section beam sees. Ribs stop it.
4. Axial load fed into the caps only, as it is at a root insert or a cap termination:
   how far along the span it takes to spread into the skins, with extensible walls.

Figures: bending (tip deflection by theory, cap stress across the chord at the root);
the skin suction (deflection of the skin and the webs, the distorted tip section,
ribs); the axial load spreading from the caps into the skins.

Run with ``--reuse`` to redraw the figures from ``data/results.json``.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import REFERENCE, SERIES, TEXT_SECONDARY, plt, save
from _shared.torsion import add_rib, contour_coordinate, loop_order, shell_section_motion
from laminate_modelling_package import (Airfoil, BladeBuilder, BoundaryConditions, FSDTModel, GeneralizedBeam,
                                        Laminate, LoadCase, ResultsExtractor, ShearWeb, SkinRegion, Station,
                                        StructuralGlobalSolver, ThinWalledSection, TimoshenkoBeam,
                                        beam_section_properties, create_default_library)

CHORD, LENGTH, N_SPAN = 1.0, 3.0, 16
REGIONS = [SkinRegion("le_panel", 0.0, 0.18, 4), SkinRegion("spar_cap", 0.18, 0.48, 6),
           SkinRegion("te_panel", 0.48, 1.0, 8)]
WEBS = [ShearWeb("web_fwd", 0.18, 4), ShearWeb("web_aft", 0.48, 4)]
TIP_FORCE = 5.0e3    # N, flapwise, through a tip rib
SUCTION = 2.0e3      # Pa, up on the upper skin
CAP_FORCE = 50.0e3   # N, axial, into the caps only
RIB_SPACINGS = [None, 0.75]
N_GBT = 32
DATA = Path(__file__).resolve().parent / "data" / "results.json"


def laminates(lib):
    glass, carbon = lib.get("E-glass/epoxy"), lib.get("T300/5208")
    skin = Laminate.from_angles(glass, [0, 45, -45, -45, 45, 0], 0.5e-3)
    web = Laminate.from_angles(glass, [45, -45, -45, 45], 0.5e-3)
    cap = Laminate.from_angles(carbon, [0] * 16, 0.5e-3)
    return {"le_panel": skin, "te_panel": skin, "web_fwd": web, "web_aft": web, "spar_cap": cap}


def build(lib):
    airfoil = Airfoil.naca4("0024")
    stations = [Station(0.0, CHORD, airfoil, pitch_axis=0.3), Station(LENGTH, CHORD, airfoil, pitch_axis=0.3)]
    return BladeBuilder(stations, REGIONS, WEBS, n_span=N_SPAN, mesh_surface="mid").build(laminates(lib), FSDTModel())


def walls_named(section, *prefixes):
    return [k for k, w in enumerate(section.walls) if w.name.startswith(prefixes)]


def gbt(section, warping="full", distortional=None, extensible=False, ribs=(), modes=None):
    z = GeneralizedBeam.graded_nodes(LENGTH, N_GBT, list(ribs))
    beam = GeneralizedBeam(section, LENGTH, warping=warping, distortional_modes=distortional, z_nodes=z,
                           modes=modes, extensible=extensible).clamp(0.0)
    for zr in ribs:
        beam.diaphragm(zr)
    return beam


def wall_nodes(section, walls):
    return sorted({section.mesh_nodes[p] for k in walls for p in (section.start[k], section.end[k])})


def gbt_wall_values(solution, z, walls, column=0):
    sec = solution.beam.section
    res = solution.wall_resultants(z)
    seen = {}
    out = []
    for p in solution.beam.modes.points_:
        if p.wall in walls:
            i = seen.get(p.wall, 0)
            seen[p.wall] = i + 1
            P0, P1 = sec.points[sec.start[p.wall]], sec.points[sec.end[p.wall]]
            xy = P0 + p.s_fraction * (P1 - P0)
            out.append((xy[0], xy[1], res[p.wall][i, column]))
    return np.array(out)


def gauss_line(results, z_target, z_max):
    g = results.gauss_points()
    zg = np.unique(np.round(g.xyz[g.xyz[:, 2] < z_max, 2], 12))
    z_row = float(zg[np.argmin(np.abs(zg - z_target))])
    sel = np.abs(g.xyz[:, 2] - z_row) < 1e-9
    return z_row, g.xyz[sel], g.N[sel]


# ====================================================================== 1 and 2: bending and shear lag
def bending(blade, sec, sections, zs, props):
    mesh = blade.mesh.copy()
    bcs = BoundaryConditions().clamp(blade.nodes("root"))
    ref = add_rib(mesh, bcs, LENGTH, sec.shear_centre)
    case = LoadCase("tip shear").add_nodal_load(ref, 1, TIP_FORCE)
    sol = StructuralGlobalSolver().solve(mesh, bcs, case)
    shell_uy = [0.0] + [float(shell_section_motion(sol, mesh, sections[float(z)], float(z)).displacement[:, 1].mean())
                        for z in zs[1:]]
    K_thin = sec.timoshenko_stiffness(sec.shear_centre)
    K_shell = props.at(sec.shear_centre)
    tip, curves = {}, {}
    for name, K, shear in (("Euler-Bernoulli", K_thin, False), ("Timoshenko, thin-wall 6 x 6", K_thin, True),
                           ("Timoshenko, shell 6 x 6", K_shell, True)):
        s = TimoshenkoBeam(K, LENGTH, 4, shear=shear).clamp(0.0).add_force(LENGTH, [0.0, TIP_FORCE, 0.0]).solve()
        curves[name] = s.displacement(zs)[:, 1].tolist()
        tip[name] = curves[name][-1]
    runs = {}
    for name, (w, d) in {"GBT plane sections": ("classical", 0), "GBT + all shear-lag modes": ("full", 0),
                         "GBT + distortion": ("full", None)}.items():
        runs[name] = gbt(sec, w, d, ribs=[LENGTH]).add_section_force(LENGTH, (0.0, TIP_FORCE)).solve()
        curves[name] = [float(runs[name].section_displacement(float(z))[:, 1].mean()) for z in zs]
        tip[name] = curves[name][-1]
    tip["shell"] = shell_uy[-1]
    res = ResultsExtractor(sol, case)
    z_row, xyz, N = gauss_line(res, 0.0, zs[1])
    caps = walls_named(sec, "spar_cap")
    upper = [k for k in caps if sec.walls[k].name.endswith("_ss")]
    node_xy = {}
    for k in upper:
        for p in (sec.start[k], sec.end[k]):
            node_xy[p] = sec.points[p]
    ys = np.array([v[1] for v in node_xy.values()])
    y_mid = 0.5 * (ys.min() + ys.max())
    xr = sorted(v[0] for v in node_xy.values())
    sel = (xyz[:, 1] > y_mid) & (xyz[:, 0] >= xr[0] - 1e-6) & (xyz[:, 0] <= xr[-1] + 1e-6)
    o = np.argsort(xyz[sel, 0])
    stress = {"shell": [xyz[sel, 0][o].tolist(), N[sel, 0][o].tolist()]}
    for name in ("GBT plane sections", "GBT + all shear-lag modes", "GBT + distortion"):
        v = gbt_wall_values(runs[name], z_row, upper)
        o = np.argsort(v[:, 0])
        stress[name] = [v[o, 0].tolist(), v[o, 2].tolist()]
    return {"z": zs.tolist(), "shell_uy": shell_uy, "curves": curves, "tip": tip, "cap_row_z": z_row,
            "cap_stress": stress, "EI_flap": float(K_thin[3, 3]), "GA_flap": float(K_thin[1, 1]),
            "GA_flap_shell": float(K_shell[1, 1])}


# ====================================================================== 3: skin suction and distortion
def suction(blade, sec, sections, zs):
    upper_skin = [e for name in ("le_panel_ss", "spar_cap_ss", "te_panel_ss") for e in blade.element_sets[name]]
    walls = [k for k in range(len(sec.walls)) if sec.walls[k].name.endswith("_ss")]
    length = float(np.sum(sec.length[walls]))
    q = SUCTION * length  # N/m of span, up
    P = sec.points
    tip_point = int(max(walls, key=lambda k: sec.length[k]))
    centre = int(sec.start[tip_point])
    webs = [p for k in walls_named(sec, "web") for p in (sec.start[k], sec.end[k])]
    K6 = sec.timoshenko_stiffness(sec.shear_centre)
    out = {"z": zs.tolist(), "ribs": {}}
    for spacing in RIB_SPACINGS:
        ribs = [] if spacing is None else [float(z) for z in np.arange(spacing, LENGTH - 1e-9, spacing)]
        mesh = blade.mesh.copy()
        bcs = BoundaryConditions().clamp(blade.nodes("root"))
        for zr in ribs:
            add_rib(mesh, bcs, zr, sec.shear_centre)
        case = LoadCase("suction").add_pressure(SUCTION, upper_skin)
        sol = StructuralGlobalSolver().solve(mesh, bcs, case)
        shell = [(0.0, 0.0)] + [(float(d[centre, 1]), float(d[webs, 1].mean())) for d in
                                (shell_section_motion(sol, mesh, sections[float(z)], float(z)).displacement
                                 for z in zs[1:])]
        counts = (0, 2, 4, None) if spacing is None else (0, None)
        runs = {}
        for n in counts:
            s = gbt(sec, "full", n, ribs=ribs).add_distributed_wall_load(walls, (0.0, q)).solve()
            runs["all" if n is None else str(n)] = [(float(d[centre, 1]), float(d[webs, 1].mean())) for d in
                                                    (s.section_displacement(float(z)) for z in zs)]
            if spacing is None and n is None:
                tip_gbt = s.section_displacement(LENGTH)
        if spacing is None:
            tip_shell = shell_section_motion(sol, mesh, sections[float(LENGTH)], LENGTH).displacement
        out["ribs"]["none" if spacing is None else f"{spacing:g}"] = {"shell": shell, "gbt": runs}
    tim = TimoshenkoBeam(K6, LENGTH, 8).clamp(0.0).add_distributed([0.0, q, 0.0]).solve()
    out["timoshenko"] = tim.displacement(zs)[:, 1].tolist()
    out["skin_point"] = float(P[centre, 0])
    out["tip_section"] = {"points": P.tolist(), "start": list(map(int, sec.start)), "end": list(map(int, sec.end)),
                          "shell": tip_shell[:, :2].tolist(), "gbt": tip_gbt[:, :2].tolist()}
    return out


# ====================================================================== 4: axial load into the caps
def axial(blade, sec, sections, zs):
    caps = walls_named(sec, "spar_cap")
    tip_sec = sections[float(LENGTH)]
    cap_nodes = wall_nodes(tip_sec, caps)
    length = float(np.sum(sec.length[caps]))
    mesh = blade.mesh.copy()
    bcs = BoundaryConditions().clamp(blade.nodes("root"))
    case = LoadCase("cap load").add_line_load(cap_nodes, (0.0, 0.0, CAP_FORCE / length))
    sol = StructuralGlobalSolver().solve(mesh, bcs, case)
    res = ResultsExtractor(sol, case)
    w = np.zeros(len(sec.points))
    for k in caps:
        w[sec.start[k]] += 0.5 * sec.length[k]
        w[sec.end[k]] += 0.5 * sec.length[k]
    runs = {"GBT plane sections": ("classical", False), "GBT + shear lag": ("full", False),
            "GBT + shear lag, extensible": ("full", True)}
    sols = {k: gbt(sec, wp, None, ex).add_axial_force(LENGTH, CAP_FORCE, distribution=w).solve()
            for k, (wp, ex) in runs.items()}
    te = walls_named(sec, "te_panel")
    z_rows = 0.5 * (zs[:-1] + zs[1:])
    g = res.gauss_points()
    # each Gauss point belongs to the wall whose mid-point is nearest, so the stiff caps
    # next door do not leak into the skin average
    mids = 0.5 * (sec.points[sec.start] + sec.points[sec.end])
    nearest = np.argmin(np.linalg.norm(g.xyz[:, None, :2] - mids[None], axis=2), axis=1)
    on_te = np.isin(nearest, te)
    share = {"shell": [], **{k: [] for k in sols}}
    for a, b, zr in zip(zs[:-1], zs[1:], z_rows):
        sel = (g.xyz[:, 2] >= a) & (g.xyz[:, 2] <= b) & on_te
        share["shell"].append(float(np.mean(g.N[sel, 0])))
        for k, s in sols.items():
            r = s.wall_resultants(float(zr))
            share[k].append(float(np.mean(np.concatenate([r[j][:, 0] for j in te]))))
    far = {k: float(np.mean([v[i] for i in range(len(z_rows)) if 0.8 < z_rows[i] < LENGTH - 0.8])) for k, v in share.items()}
    contour = {}
    order, s0 = loop_order(sec, range(len(sec.walls)))
    names = [sec.walls[k].name for k, _ in order]
    for d_target in (0.25, 1.0):
        z_row, xyz, N = gauss_line(res, LENGTH - d_target, LENGTH)
        s_shell = contour_coordinate(sec, order, s0, xyz[:, :2])
        o = np.argsort(s_shell)
        c = {"shell": [s_shell[o].tolist(), N[o, 0].tolist()]}
        for k, s in sols.items():
            frac = {}
            for p in s.beam.modes.points_:
                frac.setdefault(p.wall, []).append(p.s_fraction)
            ss, nn = [], []
            wr = s.wall_resultants(z_row)
            for (j, fwd), s_start in zip(order, s0[:-1]):
                f = np.array(frac[j])
                ss.extend(s_start + (f if fwd else 1 - f) * sec.length[j])
                nn.extend(wr[j][:, 0])
            oo = np.argsort(ss)
            c[k] = [np.array(ss)[oo].tolist(), np.array(nn)[oo].tolist()]
        contour[f"{LENGTH - z_row:.3f}"] = c
    uniform = CAP_FORCE / float(np.sum(sec.length * sec.Et))
    return {"distance": (LENGTH - z_rows).tolist(), "te_share": share, "te_far": far, "contour": contour,
            "boundaries": [float(s0[i]) for i in range(1, len(order)) if names[i] != names[i - 1]],
            "uniform_strain": uniform, "te_uniform": uniform * float(np.mean(sec.Et[te]))}


# ====================================================================== figures
def figure_bending(b):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]})
    colours = {"Euler-Bernoulli": SERIES[3], "Timoshenko, thin-wall 6 x 6": SERIES[1],
               "Timoshenko, shell 6 x 6": SERIES[7], "GBT plane sections": SERIES[4],
               "GBT + all shear-lag modes": SERIES[2], "GBT + distortion": SERIES[0]}
    ax = axes[0]
    names = list(colours)
    ratios = [b["tip"][k] / b["tip"]["shell"] for k in names]
    y = np.arange(len(names))[::-1]
    ax.barh(y, ratios, color=[colours[k] for k in names], height=0.6)
    ax.axvline(1.0, color=REFERENCE, ls="--", lw=1.0)
    for yy, r in zip(y, ratios):
        ax.text(r - 0.02, yy, f"{r:.3f}", va="center", ha="right", fontsize=8, color="white")
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("tip deflection / shell")
    ax.set_title("Tip deflection by theory", loc="left")
    ax = axes[1]
    z = np.array(b["z"])
    for k in ("Euler-Bernoulli", "Timoshenko, thin-wall 6 x 6", "GBT + distortion"):
        ax.plot(z, 1e3 * np.array(b["curves"][k]), color=colours[k], ls="--" if k.startswith("GBT") else "-", label=k)
    ax.plot(z, 1e3 * np.array(b["shell_uy"]), "o", color=REFERENCE, mfc="none", ms=5, label="shell")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("deflection u_y [mm]")
    ax.set_title(f"Cantilever, L/chord = {LENGTH / CHORD:g}", loc="left")
    ax.legend(fontsize=7.5)
    ax = axes[2]
    x, N = b["cap_stress"]["shell"]
    ax.plot(x, np.array(N) / 1e3, "o", color=REFERENCE, mfc="none", ms=4, label="shell")
    for k in ("GBT plane sections", "GBT + all shear-lag modes", "GBT + distortion"):
        x, N = b["cap_stress"][k]
        ax.plot(x, np.array(N) / 1e3, color=colours[k], ls="--" if k.endswith("distortion") else "-", label=k)
    ax.set_xlabel("x along the upper spar cap [m]")
    ax.set_ylabel("N_zz [kN/m]")
    ax.set_title(f"Shear lag in the cap at z = {b['cap_row_z']:.3f} m", loc="left")
    ax.legend(fontsize=7.5)
    fig.text(0.0, -0.04, f"NACA 0024, {CHORD:g} m chord, {LENGTH:g} m long, root clamped, {TIP_FORCE / 1e3:g} kN flapwise at the tip "
             "through a tip rib. Glass skins and webs, 16-ply carbon caps. Shell: curved Quad8 FSDT on the skins' "
             f"mid-surface, {N_SPAN} elements along; GBT: {N_GBT} elements.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "blade_bending.png")


def figure_suction(d):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1.2, 0.9, 1.0]})
    z = np.array(d["z"])
    none = d["ribs"]["none"]
    ax = axes[0]
    sh = np.array(none["shell"])
    ax.plot(z, 1e3 * np.array(d["timoshenko"]), color=SERIES[1], lw=1.4, label="Timoshenko")
    shades = {"0": SERIES[4], "2": plt.cm.Greens(0.45), "4": plt.cm.Greens(0.75), "all": SERIES[0]}
    for n, v in none["gbt"].items():
        v = np.array(v)
        ax.plot(z, 1e3 * v[:, 0], color=shades[n], lw=1.6,
                label="GBT, rigid section" if n == "0" else f"GBT, {n} distortional modes")
        ax.plot(z, 1e3 * v[:, 1], color=shades[n], lw=1.0, ls="--")
    ax.plot(z, 1e3 * sh[:, 0], "o", color=REFERENCE, mfc="none", ms=5, label="shell")
    ax.plot(z, 1e3 * sh[:, 1], "s", color=REFERENCE, mfc="none", ms=4)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("u_y [mm]")
    ax.set_title("Upper skin (solid, ○) and webs (dashed, □)", loc="left")
    ax.legend(fontsize=7.0, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax = axes[1]
    t = d["tip_section"]
    P, Us, Ug = np.array(t["points"]), np.array(t["shell"]), np.array(t["gbt"])
    scale = 0.12 / np.max(np.abs(Ug))
    for i, j in zip(t["start"], t["end"]):
        ax.plot(P[[i, j], 0], P[[i, j], 1], color=TEXT_SECONDARY, lw=0.7)
        ax.plot(P[[i, j], 0] + scale * Ug[[i, j], 0], P[[i, j], 1] + scale * Ug[[i, j], 1], color=SERIES[0], lw=1.6)
        ax.plot(P[[i, j], 0] + scale * Us[[i, j], 0], P[[i, j], 1] + scale * Us[[i, j], 1], "o", color=REFERENCE,
                mfc="none", ms=2.5)
    ax.plot([], [], color=SERIES[0], lw=1.6, label="GBT, all modes")
    ax.plot([], [], "o", color=REFERENCE, mfc="none", ms=3, label="shell")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_title(f"Tip section, no ribs (x{scale:.0f})", loc="left")
    ax.legend(fontsize=7.5, loc="lower right")
    ax = axes[2]
    keys = list(d["ribs"])
    xs = np.arange(len(keys))
    for off, which, label in ((-0.2, 0, "skin"), (0.2, 1, "webs")):
        ax.bar(xs + off - 0.09, [d["ribs"][k]["shell"][-1][which] / d["timoshenko"][-1] for k in keys], 0.18,
               color=REFERENCE, alpha=0.75 if which == 0 else 0.4, label=f"shell, {label}")
        ax.bar(xs + off + 0.09, [d["ribs"][k]["gbt"]["all"][-1][which] / d["timoshenko"][-1] for k in keys], 0.18,
               color=SERIES[0], alpha=1.0 if which == 0 else 0.55, label=f"GBT, {label}")
    ax.axhline(1.0, color=SERIES[1], ls="--", lw=1.0, label="Timoshenko")
    ax.set_xticks(xs, ["no ribs" if k == "none" else f"ribs every {k} m" for k in keys])
    ax.set_ylabel("tip u_y / Timoshenko")
    ax.set_title("Ribs stop the distortion", loc="left")
    ax.legend(fontsize=7.0, ncol=2)
    fig.text(0.0, -0.06, f"The same blade with {SUCTION / 1e3:g} kPa suction on the upper skin. The skin point is at "
             f"x = {d['skin_point']:.2f} m. Ribs: rigid in their plane, free to warp.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "blade_skin_suction.png")


def figure_axial(a):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    colours = {"GBT plane sections": SERIES[4], "GBT + shear lag": SERIES[2], "GBT + shear lag, extensible": SERIES[0]}
    ax = axes[0]
    d = np.array(a["distance"])
    ax.plot(d, np.array(a["te_share"]["shell"]) / a["te_far"]["shell"], "o", color=REFERENCE, mfc="none", ms=5,
            label="shell")
    for k, c in colours.items():
        ax.plot(d, np.array(a["te_share"][k]) / a["te_far"][k], color=c, label=k)
    ax.axhline(1.0, color=TEXT_SECONDARY, ls="--", lw=1.0, label="plane sections, far field")
    ax.set_xlabel("distance from the loaded end [m]")
    ax.set_ylabel("mean N_zz in the TE skin / its far-field value")
    ax.set_title("The caps' load spreading into the skins", loc="left")
    ax.legend(fontsize=7.5)
    for ax, key in zip(axes[1:], sorted(a["contour"], key=float)):
        c = a["contour"][key]
        ax.plot(c["shell"][0], np.array(c["shell"][1]) / 1e3, "o", color=REFERENCE, mfc="none", ms=3, label="shell")
        for k, col in colours.items():
            ax.plot(c[k][0], np.array(c[k][1]) / 1e3, color=col, lw=1.5, label=k)
        for s in a["boundaries"]:
            ax.axvline(s, color=TEXT_SECONDARY, lw=0.5, ls=":")
        ax.set_xlabel("contour position s [m] (dotted: region joints)")
        ax.set_ylabel("N_zz [kN/m]")
        ax.set_title(f"Round the section, {float(key):.2f} m from the load", loc="left")
        ax.legend(fontsize=7.5)
    fig.text(0.0, -0.04, f"{CAP_FORCE / 1e3:g} kN axial fed into the spar caps alone at the tip, as at a root insert or a cap "
             "termination. Each curve is scaled by its own far-field skin stress "
             f"(shell {a['te_far']['shell'] / 1e3:.2f} kN/m).", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "blade_axial_diffusion.png")


def report(r):
    b, d, a = r["bending"], r["suction"], r["axial"]
    print(f"Section: EI_flap {b['EI_flap']:.4g} N m2, GA_flap {b['GA_flap']:.4g} N (thin-wall), "
          f"{b['GA_flap_shell']:.4g} N (shell 6 x 6)")
    print(f"Flapwise tip force {TIP_FORCE:g} N, tip deflection / shell:")
    for k, v in b["tip"].items():
        print(f"  {k:<28} {1e3 * v:8.4f} mm  {v / b['tip']['shell']:.3f}")
    x, N = b["cap_stress"]["shell"]
    print(f"  upper cap N_zz at z = {b['cap_row_z']:.3f} m [kN/m]: shell {min(N) / 1e3:.1f} to {max(N) / 1e3:.1f}, "
          + ", ".join(f"{k} {min(v[1]) / 1e3:.1f} to {max(v[1]) / 1e3:.1f}" for k, v in b["cap_stress"].items()
                      if k != "shell"))
    print(f"Skin suction {SUCTION / 1e3:g} kPa, tip u_y (skin, webs) [mm]:")
    for key, v in d["ribs"].items():
        print(f"  ribs {key}: shell {1e3 * v['shell'][-1][0]:.3f}, {1e3 * v['shell'][-1][1]:.3f}; " + "; ".join(
            f"GBT {n}: {1e3 * w[-1][0]:.3f}, {1e3 * w[-1][1]:.3f}" for n, w in v["gbt"].items()))
    print(f"  Timoshenko {1e3 * d['timoshenko'][-1]:.3f}")
    print(f"Axial {CAP_FORCE / 1e3:g} kN into the caps, mean N_zz in the TE skin / its own far field "
          f"(far field [kN/m]: " + ", ".join(f"{k} {v / 1e3:.2f}" for k, v in a["te_far"].items()) + "):")
    for i in range(0, len(a["distance"]), max(1, len(a["distance"]) // 6)):
        print(f"  {a['distance'][i]:.3f} m: " + ", ".join(f"{k} {v[i] / a['te_far'][k]:.2f}"
                                                          for k, v in a["te_share"].items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", action="store_true", help="redraw from data/results.json")
    args = parser.parse_args()
    if args.reuse and DATA.exists():
        r = json.loads(DATA.read_text())
    else:
        blade = build(create_default_library())
        zs = np.asarray(blade.span_positions, float)
        sec = ThinWalledSection.from_blade_mesh(blade, z=LENGTH / 2)
        sections = {float(z): ThinWalledSection.from_blade_mesh(blade, z=float(z)) for z in zs[1:]}
        props = beam_section_properties(blade)
        r = {"bending": bending(blade, sec, sections, zs, props), "suction": suction(blade, sec, sections, zs),
             "axial": axial(blade, sec, sections, zs)}
        DATA.parent.mkdir(exist_ok=True)
        DATA.write_text(json.dumps(r, indent=1))
    report(r)
    figure_bending(r["bending"])
    figure_suction(r["suction"])
    figure_axial(r["axial"])


if __name__ == "__main__":
    main()
