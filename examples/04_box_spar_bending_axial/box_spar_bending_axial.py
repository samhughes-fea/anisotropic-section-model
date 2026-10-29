"""Bending and axial load beyond Euler-Bernoulli on a composite box spar.

Example 01 took torsion from Saint-Venant through restrained warping to distortion.
This example does the same for bending and axial load on the same carbon/glass box
spar, checking each layer of beam theory against a shell model of the box (flat
Quad8, FSDT):

Bending
1. Euler-Bernoulli: plane sections stay plane and normal to the axis (4 x 4 stiffness).
2. Timoshenko: the section also shears, with GA from the thin-wall shear flows
   (``ThinWalledSection.timoshenko_stiffness``, solved with ``TimoshenkoBeam``).
3. Shear lag (GBT warping modes): the caps' axial stress is not uniform across their
   width near a clamp, because the shear flow that feeds it in also shears the caps.
4. Distortion (GBT distortional modes): a pressure on one cap bends the cap and the
   webs across the section, which no rigid-section beam can show. Ribs stop it.

Axial load
1. Uniform extension: N / EA, the same stress everywhere in each wall.
2. Shear lag (GBT warping modes): a load at one corner spreads over the section only
   gradually along the span.
3. End effects (extensible GBT, with hoop strain): near a clamp the walls cannot
   contract sideways (Poisson), which raises hoop stress and lowers the axial stress;
   and a point load diffuses in the plane of the walls as a plane-stress problem.

Figures: bending (tip deflection by theory, deflection along the span, axial stress
across the top cap at the root); distortion under a cap pressure (cap and web
deflection, tip section shape, ribs); axial load (corner load diffusion, stress round
the section, Poisson end effect at the clamp).

Run with ``--reuse`` to redraw the figures from ``data/results.json``.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import REFERENCE, SERIES, TEXT_SECONDARY, plt, save
from _shared.torsion import add_rib, box_spar, contour_coordinate, loop_order, shell_section_motion
from anisotropic_section_model import GeneralizedBeam, ThinWalledSection, TimoshenkoBeam
from composite_laminate_model import (BoundaryConditions, Laminate, LoadCase, ResultsExtractor,
    StructuralGlobalSolver, create_default_library)

WIDTH, HEIGHT = 0.5, 0.25
N_WIDTH, N_HEIGHT = 8, 4
BEND_LENGTH, BEND_SPAN = 1.5, 24      # short cantilever: shear deflection and shear lag show
PRESSURE_LENGTH, PRESSURE_SPAN = 3.0, 32
AXIAL_LENGTH, AXIAL_SPAN = 2.0, 32
GBT_ELEMENTS = 40
TIP_FORCE = 1.0e3       # N, vertical, through a tip rib
PRESSURE = 2.0e3        # Pa, downward on the top cap
CORNER_FORCE = 10.0e3   # N, axial, at one tip corner
RIB_SPACINGS = [None, 1.0, 0.5]
DATA = Path(__file__).resolve().parent / "data" / "results.json"


def laminates(lib):
    carbon, glass = lib.get("T300/5208"), lib.get("E-glass/epoxy")
    cap = Laminate.from_angles(carbon, [0, 0, 45, -45, -45, 45, 0, 0], 0.5e-3, "cap")
    web = Laminate.from_angles(glass, [45, -45, -45, 45], 0.5e-3, "web")
    return {"cap_top": cap, "cap_bottom": cap, "web_le": web, "web_te": web}


def gbt(section, length, warping="full", distortional=None, extensible=False, ribs=()):
    z = GeneralizedBeam.graded_nodes(length, GBT_ELEMENTS, list(ribs))
    beam = GeneralizedBeam(section, length, warping=warping, distortional_modes=distortional, z_nodes=z,
                           extensible=extensible).clamp(0.0)
    for zr in ribs:
        beam.diaphragm(zr)
    return beam


def walls_named(section, prefix):
    return [k for k, w in enumerate(section.walls) if w.name.startswith(prefix)]


def gbt_wall_values(solution, z, walls, column=0):
    """(x, y, value) at the GBT wall points of ``walls`` at z (column 0: N_zz)."""
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


def gbt_contour(solution, z, order, s0):
    """Contour position and N_zz at every GBT wall point at z."""
    sec = solution.beam.section
    wr = solution.wall_resultants(z)
    frac = {}
    for p in solution.beam.modes.points_:
        frac.setdefault(p.wall, []).append(p.s_fraction)
    ss, nn = [], []
    for (k, fwd), s_start in zip(order, s0[:-1]):
        f = np.array(frac[k])
        ss.extend(s_start + (f if fwd else 1 - f) * sec.length[k])
        nn.extend(wr[k][:, 0])
    o = np.argsort(ss)
    return np.array(ss)[o], np.array(nn)[o]


def gauss_line(results, z_target, z_max):
    """The Gauss points on the Gauss line nearest z_target (below z_max)."""
    g = results.gauss_points()
    zg = np.unique(np.round(g.xyz[g.xyz[:, 2] < z_max, 2], 12))
    z_row = float(zg[np.argmin(np.abs(zg - z_target))])
    sel = np.abs(g.xyz[:, 2] - z_row) < 1e-9
    return z_row, g.xyz[sel], g.N[sel]


# ====================================================================== bending
def bending(lib):
    L = BEND_LENGTH
    box = box_spar(WIDTH, HEIGHT, L, laminates(lib), N_WIDTH, N_HEIGHT, BEND_SPAN)
    sec = ThinWalledSection.from_blade_mesh(box, z=L / 2)
    zs = box.span_positions
    K6 = sec.timoshenko_stiffness(sec.shear_centre)
    # shell: root clamped, tip rib through the shear centre takes the force
    mesh = box.mesh.copy()
    bcs = BoundaryConditions().clamp(box.nodes("root"))
    ref = add_rib(mesh, bcs, L, sec.shear_centre)
    case = LoadCase("tip shear").add_nodal_load(ref, 1, TIP_FORCE)
    sol = StructuralGlobalSolver().solve(mesh, bcs, case)
    sections = {float(z): ThinWalledSection.from_blade_mesh(box, z=float(z)) for z in zs[1:]}
    shell_uy = [0.0] + [float(shell_section_motion(sol, mesh, sections[float(z)], float(z)).displacement[:, 1].mean())
                        for z in zs[1:]]
    beams = {"Euler-Bernoulli": TimoshenkoBeam(K6, L, 4, shear=False), "Timoshenko": TimoshenkoBeam(K6, L, 4)}
    z_fine = np.linspace(0.0, L, 61)
    curves = {k: b.clamp(0.0).add_force(L, [0.0, TIP_FORCE, 0.0]).solve().displacement(z_fine)[:, 1].tolist()
              for k, b in beams.items()}
    gbt_runs = {"GBT plane sections": ("classical", 0), "GBT + 2 shear-lag modes": (2, 0),
                "GBT + all shear-lag modes": ("full", 0), "GBT + distortion": ("full", None)}
    gbt_sol = {}
    for name, (w, d) in gbt_runs.items():
        gbt_sol[name] = gbt(sec, L, w, d, ribs=[L]).add_section_force(L, (0.0, TIP_FORCE)).solve()
        curves[name] = [float(gbt_sol[name].section_displacement(float(z))[:, 1].mean()) for z in zs]
    tip = {k: v[-1] for k, v in curves.items()}
    tip["shell"] = shell_uy[-1]
    # axial stress across the top cap on the shell's first Gauss line
    res = ResultsExtractor(sol, case)
    z_row, xyz, N = gauss_line(res, 0.0, zs[1])
    top = np.abs(xyz[:, 1] - HEIGHT / 2) < 1e-6
    o = np.argsort(xyz[top, 0])
    caps = walls_named(sec, "cap_top")
    cap_stress = {"shell": [xyz[top, 0][o].tolist(), N[top, 0][o].tolist()]}
    for name in ("GBT plane sections", "GBT + all shear-lag modes", "GBT + distortion"):
        v = gbt_wall_values(gbt_sol[name], z_row, caps)
        o = np.argsort(v[:, 0])
        cap_stress[name] = [v[o, 0].tolist(), v[o, 2].tolist()]
    return {"length": L, "z_shell": zs.tolist(), "shell_uy": shell_uy, "z_fine": z_fine.tolist(), "z_gbt": zs.tolist(),
            "curves": curves, "tip": tip, "cap_row_z": z_row, "cap_stress": cap_stress,
            "GA_y": float(K6[1, 1]), "EI_x": float(K6[3, 3])}


# ====================================================================== distortion under a cap pressure
def distortion(lib):
    L = PRESSURE_LENGTH
    box = box_spar(WIDTH, HEIGHT, L, laminates(lib), N_WIDTH, N_HEIGHT, PRESSURE_SPAN)
    sec = ThinWalledSection.from_blade_mesh(box, z=L / 2)
    zs = box.span_positions
    sections = {float(z): ThinWalledSection.from_blade_mesh(box, z=float(z)) for z in zs[1:]}
    P = sec.points
    centre = int(np.argmin(np.hypot(P[:, 0], P[:, 1] - HEIGHT / 2)))
    webs = np.abs(np.abs(P[:, 0]) - WIDTH / 2) < 1e-9
    q = -PRESSURE * WIDTH  # N/m, downward
    caps = walls_named(sec, "cap_top")
    K6 = sec.timoshenko_stiffness(sec.shear_centre)

    def measures(disp):
        return float(disp[centre, 1]), float(disp[webs, 1].mean())

    out = {"z": zs.tolist(), "ribs": {}}
    for spacing in RIB_SPACINGS:
        ribs = [] if spacing is None else [float(z) for z in np.arange(spacing, L + 1e-9, spacing)]
        mesh = box.mesh.copy()
        bcs = BoundaryConditions().clamp(box.nodes("root"))
        for zr in ribs:
            add_rib(mesh, bcs, zr, sec.shear_centre)
        case = LoadCase("cap pressure").add_pressure(-PRESSURE, box.elements("cap_top"))
        sol = StructuralGlobalSolver().solve(mesh, bcs, case)
        shell = [(0.0, 0.0)] + [measures(shell_section_motion(sol, mesh, sections[float(z)], float(z)).displacement)
                                for z in zs[1:]]
        counts = (0, 2, 4, None) if spacing is None else (0, None)
        runs = {}
        for n in counts:
            s = gbt(sec, L, "full", n, ribs=ribs).add_distributed_wall_load(caps, (0.0, q)).solve()
            runs["all" if n is None else str(n)] = [measures(s.section_displacement(float(z))) for z in zs]
            if spacing is None and n is None:
                tip_gbt = s.section_displacement(L)
        key = "none" if spacing is None else f"{spacing:g}"
        out["ribs"][key] = {"shell": shell, "gbt": runs}
        if spacing is None:
            tip_shell = shell_section_motion(sol, mesh, sections[float(L)], L).displacement
    tim = TimoshenkoBeam(K6, L, 8).clamp(0.0).add_distributed([0.0, q, 0.0]).solve()
    eb = TimoshenkoBeam(K6, L, 8, shear=False).clamp(0.0).add_distributed([0.0, q, 0.0]).solve()
    out["timoshenko"] = tim.displacement(zs)[:, 1].tolist()
    out["euler_bernoulli"] = eb.displacement(zs)[:, 1].tolist()
    out["tip_section"] = {"points": P.tolist(), "start": list(map(int, sec.start)), "end": list(map(int, sec.end)),
                          "shell": tip_shell[:, :2].tolist(), "gbt": tip_gbt[:, :2].tolist()}
    return out


# ====================================================================== axial load
def axial(lib):
    L = AXIAL_LENGTH
    box = box_spar(WIDTH, HEIGHT, L, laminates(lib), N_WIDTH, N_HEIGHT, AXIAL_SPAN)
    sec = ThinWalledSection.from_blade_mesh(box, z=L / 2)
    zs = box.span_positions
    order, s0 = loop_order(sec, range(len(sec.walls)))
    out = {}
    # ---- a point load at one tip corner
    mesh = box.mesh.copy()
    bcs = BoundaryConditions().clamp(box.nodes("root"))
    corner = [n for n in box.nodes("tip") if np.allclose(mesh.node_coordinates(n)[:2], (WIDTH / 2, HEIGHT / 2))][0]
    case = LoadCase("corner").add_nodal_load(corner, 2, CORNER_FORCE)
    sol = StructuralGlobalSolver().solve(mesh, bcs, case)
    res = ResultsExtractor(sol, case)
    g = res.gauss_points()
    kc = int(np.argmin(np.linalg.norm(sec.points - np.array([WIDTH / 2, HEIGHT / 2]), axis=1)))
    runs = {"GBT plane sections": ("classical", False), "GBT + shear lag": ("full", False),
            "GBT + shear lag, extensible": ("full", True)}
    sols = {k: gbt(sec, L, w, None, ex).add_point_force(L, kc, (0.0, 0.0, CORNER_FORCE)).solve()
            for k, (w, ex) in runs.items()}
    z_rows = 0.5 * (zs[:-1] + zs[1:])
    peak = {"shell": [float(g.N[(g.xyz[:, 2] >= a) & (g.xyz[:, 2] <= b), 0].max()) for a, b in zip(zs[:-1], zs[1:])]}
    for k, s in sols.items():
        peak[k] = [float(max(r[:, 0].max() for r in s.wall_resultants(float(z)))) for z in z_rows]
    out["corner"] = {"distance": (L - z_rows).tolist(), "peak": peak, "uniform": CORNER_FORCE / float(np.sum(sec.length * sec.Et)) *
                     float(np.max(sec.Et))}
    contour = {}
    for d_target in (0.1, 0.5):
        z_row, xyz, N = gauss_line(res, L - d_target, L)
        s_shell = contour_coordinate(sec, order, s0, xyz[:, :2])
        o = np.argsort(s_shell)
        c = {"shell": [s_shell[o].tolist(), N[o, 0].tolist()]}
        for k, s in sols.items():
            ss, nn = gbt_contour(s, z_row, order, s0)
            c[k] = [ss.tolist(), nn.tolist()]
        contour[f"{L - z_row:.3f}"] = c
    out["contour"] = contour
    names = [sec.walls[k].name for k, _ in order]
    out["corners_s"] = [float(s0[i]) for i in range(1, len(order)) if names[i] != names[i - 1]]
    # ---- uniform extension: Poisson end effect at the clamp
    mesh = box.mesh.copy()
    bcs = BoundaryConditions().clamp(box.nodes("root")).fix(box.nodes("tip"), [2], 1e-3)
    sol = StructuralGlobalSolver().solve(mesh, bcs, LoadCase("stretch"))
    g = ResultsExtractor(sol, LoadCase("stretch")).gauss_points()
    line = (np.abs(g.xyz[:, 1] - HEIGHT / 2) < 1e-6) & (np.abs(g.xyz[:, 0]) < 0.04)
    o = np.argsort(g.xyz[line, 2])
    z_sh, Nzz, Nss = g.xyz[line, 2][o], g.N[line, 0][o], g.N[line, 1][o]
    far = float(np.interp(L / 2, z_sh, Nzz))
    caps = walls_named(sec, "cap_top")
    xm = [0.5 * (sec.points[sec.start[k], 0] + sec.points[sec.end[k], 0]) for k in caps]
    k_mid = caps[int(np.argmin(np.abs(xm)))]
    z_g = np.linspace(0.0, 0.6, 61)
    pois = {"z_shell": z_sh.tolist(), "shell_Nzz": (Nzz / far).tolist(), "shell_Nss": (Nss / far).tolist(), "z_gbt": z_g.tolist()}
    for name, ex in (("GBT", False), ("GBT extensible", True)):
        s = gbt(sec, L, "full", None, ex).add_axial_force(L, 1.0).solve()
        ref = s.wall_resultants(L / 2)[k_mid][:, 0].mean()
        r = np.array([s.wall_resultants(float(z))[k_mid].mean(axis=0) for z in z_g])
        pois[name + "_Nzz"] = (r[:, 0] / ref).tolist()
        if ex:
            pois[name + "_Nss"] = (r[:, 1] / ref).tolist()
    out["poisson"] = pois
    return out


# ====================================================================== figures
def figure_bending(b):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    colours = {"Euler-Bernoulli": SERIES[3], "Timoshenko": SERIES[1], "GBT plane sections": SERIES[4],
               "GBT + 2 shear-lag modes": SERIES[6], "GBT + all shear-lag modes": SERIES[2], "GBT + distortion": SERIES[0]}
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
    zf = np.array(b["z_fine"])
    for k in ("Euler-Bernoulli", "Timoshenko"):
        ax.plot(zf, 1e3 * np.array(b["curves"][k]), color=colours[k], label=k)
    ax.plot(b["z_gbt"], 1e3 * np.array(b["curves"]["GBT + distortion"]), color=colours["GBT + distortion"],
            ls="--", label="GBT + distortion")
    ax.plot(b["z_shell"], 1e3 * np.array(b["shell_uy"]), "o", color=REFERENCE, mfc="none", ms=5, label="shell")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("deflection u_y [mm]")
    ax.set_title(f"Cantilever, L/h = {b['length'] / HEIGHT:g}", loc="left")
    ax.legend(fontsize=7.5)
    ax = axes[2]
    x, N = b["cap_stress"]["shell"]
    ax.plot(x, np.array(N) / 1e3, "o", color=REFERENCE, mfc="none", ms=4, label="shell")
    for k in ("GBT plane sections", "GBT + all shear-lag modes", "GBT + distortion"):
        x, N = b["cap_stress"][k]
        ax.plot(x, np.array(N) / 1e3, color=colours[k], ls="--" if k == "GBT + distortion" else "-", label=k)
    ax.set_xlabel("x across the top cap [m]")
    ax.set_ylabel("N_zz [kN/m]")
    ax.set_title(f"Shear lag in the top cap at z = {b['cap_row_z']:.3f} m", loc="left")
    ax.legend(fontsize=7.5)
    fig.text(0.0, -0.04, f"Box {WIDTH:g} x {HEIGHT:g} m, {b['length']:g} m long, root clamped, {TIP_FORCE:g} N vertical at the tip "
             "through a tip rib. Caps [0₂/±45/∓45/0₂] T300/5208 (4 mm), webs [±45]s E-glass (2 mm). Shell: flat Quad8 FSDT, "
             f"{BEND_SPAN} elements along, {N_WIDTH} across each cap, {N_HEIGHT} up each web; GBT: {GBT_ELEMENTS} elements.",
             fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "box_spar_bending.png")


def figure_distortion(d):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1.2, 0.9, 1.0]})
    z = np.array(d["z"])
    none = d["ribs"]["none"]
    ax = axes[0]
    sh = np.array(none["shell"])
    ax.plot(z, 1e3 * np.array(d["euler_bernoulli"]), color=SERIES[3], lw=1.4, label="Euler-Bernoulli")
    ax.plot(z, 1e3 * np.array(d["timoshenko"]), color=SERIES[1], lw=1.4, label="Timoshenko")
    shades = {"0": SERIES[4], "2": plt.cm.Greens(0.45), "4": plt.cm.Greens(0.75), "all": SERIES[0]}
    for n, v in none["gbt"].items():
        v = np.array(v)
        label = "GBT, rigid section" if n == "0" else f"GBT, {n} distortional modes"
        ax.plot(z, 1e3 * v[:, 0], color=shades[n], lw=1.6, label=label)
        ax.plot(z, 1e3 * v[:, 1], color=shades[n], lw=1.0, ls="--")
    ax.plot(z, 1e3 * sh[:, 0], "o", color=REFERENCE, mfc="none", ms=5, label="shell")
    ax.plot(z, 1e3 * sh[:, 1], "s", color=REFERENCE, mfc="none", ms=4)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("u_y [mm]")
    ax.set_title("Top-cap centre (solid, ○) and webs (dashed, □)", loc="left")
    ax.legend(fontsize=7.0, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax = axes[1]
    t = d["tip_section"]
    P, Us, Ug = np.array(t["points"]), np.array(t["shell"]), np.array(t["gbt"])
    scale = 0.08 / np.max(np.abs(Ug))
    for i, j in zip(t["start"], t["end"]):
        ax.plot(P[[i, j], 0], P[[i, j], 1], color=TEXT_SECONDARY, lw=0.8)
        ax.plot(P[[i, j], 0] + scale * Ug[[i, j], 0], P[[i, j], 1] + scale * Ug[[i, j], 1], color=SERIES[0], lw=1.8)
        ax.plot(P[[i, j], 0] + scale * Us[[i, j], 0], P[[i, j], 1] + scale * Us[[i, j], 1], "o", color=REFERENCE,
                mfc="none", ms=3)
    ax.plot([], [], color=SERIES[0], lw=1.8, label="GBT, all modes")
    ax.plot([], [], "o", color=REFERENCE, mfc="none", ms=3, label="shell")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_title(f"Tip section, no ribs (x{scale:.0f})", loc="left")
    ax.legend(fontsize=7.5, loc="center")
    ax = axes[2]
    keys = list(d["ribs"])
    xs = np.arange(len(keys))
    for off, which, label in ((-0.2, 0, "top-cap centre"), (0.2, 1, "webs")):
        ax.bar(xs + off - 0.09, [d["ribs"][k]["shell"][-1][which] / d["timoshenko"][-1] for k in keys], 0.18,
               color=REFERENCE, alpha=0.75 if which == 0 else 0.4, label=f"shell, {label}")
        ax.bar(xs + off + 0.09, [d["ribs"][k]["gbt"]["all"][-1][which] / d["timoshenko"][-1] for k in keys], 0.18,
               color=SERIES[0], alpha=1.0 if which == 0 else 0.55, label=f"GBT, {label}")
    ax.axhline(1.0, color=SERIES[1], ls="--", lw=1.0, label="Timoshenko")
    ax.set_xticks(xs, ["no ribs" if k == "none" else f"ribs every {k} m" for k in keys])
    ax.set_ylabel("tip u_y / Timoshenko")
    ax.set_title("Ribs stop the distortion", loc="left")
    ax.legend(fontsize=7.0, ncol=2)
    fig.text(0.0, -0.06, f"Box {WIDTH:g} x {HEIGHT:g} m, {PRESSURE_LENGTH:g} m long, root clamped, {PRESSURE / 1e3:g} kPa down on the "
             f"top cap. Ribs: rigid in their plane, free to warp. Shell {PRESSURE_SPAN} elements along.", fontsize=7.5,
             color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "box_spar_cap_pressure.png")


def figure_axial(a):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    colours = {"GBT plane sections": SERIES[4], "GBT + shear lag": SERIES[2], "GBT + shear lag, extensible": SERIES[0]}
    c = a["corner"]
    ax = axes[0]
    d = np.array(c["distance"])
    ax.plot(d, np.array(c["peak"]["shell"]) / 1e3, "o", color=REFERENCE, mfc="none", ms=5, label="shell")
    for k, col in colours.items():
        ax.plot(d, np.array(c["peak"][k]) / 1e3, color=col, label=k)
    ax.set_yscale("log")
    ax.set_xlabel("distance from the loaded end [m]")
    ax.set_ylabel("peak N_zz [kN/m]")
    ax.set_title("A corner load spreading along the span", loc="left")
    ax.legend(fontsize=7.5)
    ax = axes[1]
    key = sorted(a["contour"], key=float)[0]
    cc = a["contour"][key]
    ax.plot(*[np.array(v) / (1e3 if i else 1) for i, v in enumerate(cc["shell"])], "o", color=REFERENCE, mfc="none",
            ms=3, label="shell")
    for k, col in colours.items():
        ax.plot(cc[k][0], np.array(cc[k][1]) / 1e3, color=col, lw=1.5, label=k)
    for s in a["corners_s"]:
        ax.axvline(s, color=TEXT_SECONDARY, lw=0.5, ls=":")
    ax.set_xlabel("contour position s [m] (dotted: corners)")
    ax.set_ylabel("N_zz [kN/m]")
    ax.set_title(f"Round the section, {float(key):.2f} m from the load", loc="left")
    ax.legend(fontsize=7.5)
    ax = axes[2]
    p = a["poisson"]
    ax.plot(p["z_shell"], p["shell_Nzz"], "o", color=REFERENCE, mfc="none", ms=4, label="shell, N_zz")
    ax.plot(p["z_shell"], p["shell_Nss"], "s", color=REFERENCE, mfc="none", ms=4, label="shell, N_ss")
    ax.plot(p["z_gbt"], p["GBT_Nzz"], color=SERIES[2], label="GBT inextensible, N_zz")
    ax.plot(p["z_gbt"], p["GBT extensible_Nzz"], color=SERIES[0], label="GBT extensible, N_zz")
    ax.plot(p["z_gbt"], p["GBT extensible_Nss"], color=SERIES[0], ls="--", label="GBT extensible, N_ss")
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("z from the clamp [m]")
    ax.set_ylabel("stress resultant / N_zz far from the ends")
    ax.set_title("Poisson end effect at the clamp, top-cap centre", loc="left")
    ax.legend(fontsize=7.5)
    fig.text(0.0, -0.04, f"Box {WIDTH:g} x {HEIGHT:g} m, {AXIAL_LENGTH:g} m long, root clamped. Left and middle: {CORNER_FORCE / 1e3:g} kN "
             "axial at the top trailing corner of the tip. Right: the tip pulled uniformly. Shell "
             f"{AXIAL_SPAN} elements along.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "box_spar_axial.png")


def report(r):
    b, d, a = r["bending"], r["distortion"], r["axial"]
    print(f"Bending, L = {b['length']:g} m, tip force {TIP_FORCE:g} N: tip deflection / shell")
    for k, v in b["tip"].items():
        print(f"  {k:<28} {1e3 * v:8.4f} mm  {v / b['tip']['shell']:.3f}")
    print("Cap pressure, tip u_y (cap centre, webs) [mm]:")
    for key, v in d["ribs"].items():
        print(f"  ribs {key}: shell {1e3 * v['shell'][-1][0]:.3f}, {1e3 * v['shell'][-1][1]:.3f}; " + ", ".join(
            f"GBT {n}: {1e3 * w[-1][0]:.3f}, {1e3 * w[-1][1]:.3f}" for n, w in v["gbt"].items()))
    print(f"  Timoshenko {1e3 * d['timoshenko'][-1]:.3f}, Euler-Bernoulli {1e3 * d['euler_bernoulli'][-1]:.3f}")
    c = a["corner"]
    print("Corner load, peak N_zz [kN/m] by distance from the tip:")
    order = np.argsort(c["distance"])
    for i in order[[0, 2, 5, 10, len(order) // 2]]:
        print(f"  {c['distance'][i]:.3f} m: " + ", ".join(f"{k} {v[i] / 1e3:.1f}" for k, v in c["peak"].items()))
    p = a["poisson"]
    print(f"Clamp, top-cap centre: shell N_zz {p['shell_Nzz'][0]:.3f}, N_ss {p['shell_Nss'][0]:.3f} (first Gauss point, "
          f"z = {p['z_shell'][0]:.4f}); GBT extensible at z = 0: N_zz {p['GBT extensible_Nzz'][0]:.3f}, "
          f"N_ss {p['GBT extensible_Nss'][0]:.3f}; inextensible {p['GBT_Nzz'][0]:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", action="store_true", help="redraw from data/results.json")
    args = parser.parse_args()
    if args.reuse and DATA.exists():
        r = json.loads(DATA.read_text())
    else:
        lib = create_default_library()
        r = {"bending": bending(lib), "distortion": distortion(lib), "axial": axial(lib)}
        DATA.parent.mkdir(exist_ok=True)
        DATA.write_text(json.dumps(r, indent=1))
    report(r)
    figure_bending(r["bending"])
    figure_distortion(r["distortion"])
    figure_axial(r["axial"])


if __name__ == "__main__":
    main()
