"""Strength check of blade sections: section forces, then ply stresses, then failure exposures.

The three-cell NACA 0024 blade of examples 02 and 05 (1 m chord, 3 m long: glass skins,
carbon spar caps, two glass shear webs) is checked along its span for two tip load cases,
a flapwise and an edgewise one, each with a torque:

1. **Section forces.** A cantilever is statically determinate, so the six section forces at
   every cut follow from statics alone (``cut_forces``). Forces from a beam solve, and the
   envelope over load cases along a beam, are 1d-fem's (its ``ply_strength.txt``).
2. **Ply stresses.** ``SectionStrength.from_thin_walled`` recovers the ply stresses of every
   ply of every wall at each station from the coupled Saint-Venant solution of the section
   (with the interlaminar stresses from 3D equilibrium).
3. **Failure exposures.** Maximum stress, Tsai-Wu, Hashin and Puck at every point
   (``ply_failure``), the critical ply and where it is, and the envelope over the two cases.

Each step is checked against a shell model of the same blade (curved Quad8 FSDT, skins on
their mid-surface, a tip rib through the shear centre): the ply stresses point by point at
mid-span, the largest exposure at every element row along the span, and the critical part,
ply and criterion. Then the same section meshed in 2D (``SectionStrength.from_solid``)
gives the same critical exposure away from the wall junctions.

Figures: the exposure along the span for both cases and their envelope; the ply stresses
against the shell at mid-span; the exposure over the root section with its critical point.

Run with ``--reuse`` to redraw the figures and re-run the checks from ``data/results.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

from _shared.checks import Checks  # noqa: E402
from _shared.plotting import REFERENCE, SEQUENTIAL, SERIES, TEXT_SECONDARY, plt, save  # noqa: E402
from _shared.torsion import add_rib  # noqa: E402

DATA = HERE / "data" / "results.json"
CHORD, LENGTH, N_SPAN = 1.0, 3.0, 16
#: tip loads [Fx, Fy, N, Mx, My, Mz] through the tip rib at the shear centre (x to the trailing
#: edge, y to the suction side): a flapwise and an edgewise case, each with a torque
CASES = {
    "flapwise": [3.0e3, 30.0e3, 0.0, 0.0, 0.0, 4.0e3],
    "edgewise": [-40.0e3, -6.0e3, 0.0, 0.0, 0.0, -3.0e3],
}
MID = 1.5            # station of the point-by-point comparison
VALID = (0.5, 2.6)   # rows away from the clamp and the tip rib, where Saint-Venant holds


def blade_model(lib):
    from shell_fem import Airfoil, BladeBuilder, FSDTModel, Laminate, ShearWeb, SkinRegion, Station

    glass, carbon = lib.get("E-glass/epoxy"), lib.get("T300/5208")
    skin = Laminate.from_angles(glass, [0, 45, -45, -45, 45, 0], 0.5e-3)
    web = Laminate.from_angles(glass, [45, -45, -45, 45], 0.5e-3)
    cap = Laminate.from_angles(carbon, [0] * 16, 0.5e-3)
    lams = {"le_panel": skin, "te_panel": skin, "web_fwd": web, "web_aft": web, "spar_cap": cap}
    regions = [SkinRegion("le_panel", 0.0, 0.18, 4), SkinRegion("spar_cap", 0.18, 0.48, 6),
               SkinRegion("te_panel", 0.48, 1.0, 8)]
    webs = [ShearWeb("web_fwd", 0.18, 4), ShearWeb("web_aft", 0.48, 4)]
    airfoil = Airfoil.naca4("0024")
    stations = [Station(0.0, CHORD, airfoil, pitch_axis=0.3), Station(LENGTH, CHORD, airfoil, pitch_axis=0.3)]
    return BladeBuilder(stations, regions, webs, n_span=N_SPAN, mesh_surface="mid").build(lams, FSDTModel())


def part(name):
    return name.replace("_ss", "").replace("_ps", "")


def cut_forces(z, tip, centre):
    """Section forces ``[Fx, Fy, N, Mx, My, Mz]`` about the section origin at the cut ``z`` of
    the cantilever, for the tip load ``tip`` applied at the point ``centre``: statics alone."""
    Fx, Fy, N, Mx, My, Mz = tip
    xs, ys = centre
    a = LENGTH - z
    return np.array([Fx, Fy, N, Mx - Fy * a + ys * N, My + Fx * a - xs * N, Mz + xs * Fy - ys * Fx])


# ============================================================================ the whole pipeline
def solve(lib):
    from anisotropic_section_model import ThinWalledSection
    from anisotropic_section_model.ply_failure import CRITERIA, PlyStrength, exposures
    from anisotropic_section_model.section_strength import SectionStrength
    from anisotropic_section_model.solid import SolidSectionAnalysis
    from anisotropic_section_model.solid.mesh import SectionMesh
    from shell_fem import BoundaryConditions, ExactStressRecoverer, LoadCase, StructuralGlobalSolver

    blade = blade_model(lib)
    sec = ThinWalledSection.from_blade_mesh(blade, z=MID)  # prismatic: one section for every station
    check = SectionStrength.from_thin_walled(sec, points_per_wall=3, points_per_ply=3)
    dz = LENGTH / N_SPAN
    rows = (np.arange(N_SPAN) + 0.5) * dz          # shell element rows: compare at their middles
    stations = np.concatenate([[0.0], rows, [LENGTH]])

    # the shell's sampling points (element centres, the middle of every ply) and the same
    # points of the section check: an element's mid-side node is a section point
    mesh0 = blade.mesh
    row_elements = {j: [] for j in range(N_SPAN)}
    for e in mesh0.shell_elements:
        zc = float(np.mean([mesh0.node_coordinates(n)[2] for n in e.node_ids]))
        row_elements[int(np.clip(np.floor(zc / dz), 0, N_SPAN - 1))].append(e)
    def points_of(e):
        """(ply, height, index of the same point in the section check) at an element's centre."""
        xy = e.global_coordinates(0.0, 0.0)[:2]
        d = np.linalg.norm(check.position - xy, axis=1)
        out_ = []
        for k, (zb, zt) in enumerate(e.laminate.ply_bounds()):
            j = np.where((d < 1e-9) & (check.ply == k) & np.isclose(check.n, 0.5 * (zb + zt)))[0]
            if j.size:
                out_.append((k, 0.5 * (zb + zt), int(j[0])))
        return out_

    sampled = {j: [(e, points_of(e)) for e in row_elements[j]] for j in range(N_SPAN)}
    strengths = {}
    rec = ExactStressRecoverer()
    out = {"stations": stations.tolist(), "rows": rows.tolist(), "cases": {}, "points": {}}
    for name, tip in CASES.items():
        forces = np.array([cut_forces(z, tip, sec.shear_centre) for z in stations])
        sections = [check.evaluate(f) for f in forces]
        largest = np.array([float(s_.exposure().max()) for s_ in sections])

        mesh = blade.mesh.copy()
        bcs = BoundaryConditions().clamp(blade.nodes("root"))
        ref = add_rib(mesh, bcs, LENGTH, sec.shear_centre)
        case = LoadCase(name)
        for dof in (0, 1, 5):
            case.add_nodal_load(ref, dof, tip[dof])
        sol = StructuralGlobalSolver().solve(mesh, bcs, case)

        shell_rows = []
        for j in range(N_SPAN):
            stress, st, idx_asm = [], [], []
            for e, pts in sampled[j]:
                u = sol.element_displacements(e.element_id)
                for k, n, i_asm in pts:
                    stress.append(rec.recover(e, u, n, 0.0, 0.0).material_stress_3d())
                    card = e.laminate.plies[k].material
                    st.append(strengths.setdefault(card.name, PlyStrength.from_strength_properties(card.strength)))
                    idx_asm.append(i_asm)
            stress, idx_asm = np.array(stress), np.array(idx_asm)
            ex = {c: np.zeros(len(stress)) for c in CRITERIA + ("exposure",)}
            for s_ in {id(v): v for v in st}.values():
                idx = np.array([i for i, v in enumerate(st) if v is s_])
                e_ = exposures(stress[idx], s_)
                for c in ex:
                    ex[c][idx] = e_[c]
            shell_rows.append((stress, ex, idx_asm))

        cmp = {"shell": {c: [] for c in CRITERIA + ("exposure",)}, "asm": {c: [] for c in CRITERIA + ("exposure",)},
               "asm_all": largest.tolist(), "shell_critical": [], "asm_critical": []}
        for j, (stress, ex, m) in enumerate(shell_rows):
            r = sections[j + 1]  # station j + 1 is the middle of row j
            for c in cmp["shell"]:
                cmp["shell"][c].append(float(ex[c].max()))
                cmp["asm"][c].append(float(r.exposure(c)[m].max()))
            i_s, i_a = int(np.argmax(ex["exposure"])), int(np.argmax(r.exposure()[m]))
            gov_s = max(CRITERIA, key=lambda c: ex[c][i_s])
            gov_a = max(CRITERIA, key=lambda c: r.exposure(c)[m][i_a])
            cmp["shell_critical"].append({"part": part(check.names[m[i_s]]), "ply": int(check.ply[m[i_s]]),
                                          "criterion": gov_s})
            cmp["asm_critical"].append({"part": part(check.names[m[i_a]]), "ply": int(check.ply[m[i_a]]),
                                        "criterion": gov_a})
        jm = int(np.argmin(np.abs(rows - MID)))
        stress_mid, _, m = shell_rows[jm]
        cmp["mid"] = {"z": float(rows[jm]), "part": [part(check.names[i]) for i in m], "shell": stress_mid.tolist(),
                      "asm": sections[jm + 1].stress[m].tolist()}
        cmp["forces"] = forces.tolist()
        jc = int(np.argmax(largest))
        cmp["summary"] = {"critical": {"z": float(stations[jc]), **sections[jc].critical().as_dict()}}
        root = sections[0]
        cmp["root"] = {"exposure": root.exposure().tolist(), "position": check.position.tolist(),
                       "critical": root.critical().as_dict()}
        by_name = [s_.by_name() for s_ in sections]
        cmp["by_part"] = {k: [float(b.get(k, np.nan)) for b in by_name] for k in dict.fromkeys(k for b in by_name for k in b)}
        out["cases"][name] = cmp

    # the same section meshed in 2D: critical exposure at mid-span, everywhere and away from
    # the wall junctions (where the strips of the solid mesh overlap)
    a = sec.analysis
    smesh = SectionMesh.from_thin_walled(a.section, a.laminates, elements_along=2)
    solid = SectionStrength.from_solid(SolidSectionAnalysis(smesh), SectionStrength.ply_strengths(sec))
    junctions = np.array([p for p in range(len(sec.points))
                          if sum(int(sec.start[k] == p) + int(sec.end[k] == p) for k in range(len(sec.walls))) > 2])
    jp = sec.points[junctions]

    def away(c):
        d = np.min(np.linalg.norm(c.position[:, None, :] - jp[None, :, :], axis=2), axis=1)
        return d > 0.03

    out["solid"] = {}
    for name in CASES:
        F = np.array(out["cases"][name]["forces"])[int(np.argmin(np.abs(stations - MID)))]
        t, s_ = check.evaluate(F), solid.evaluate(F)
        out["solid"][name] = {
            "thin": t.critical().as_dict(), "solid": s_.critical().as_dict(),
            "thin_away": t.critical(where=away(check)).as_dict(), "solid_away": s_.critical(where=away(solid)).as_dict(),
        }
    out["section"] = {"points": sec.points.tolist(), "start": list(map(int, sec.start)), "end": list(map(int, sec.end)),
                      "shear_centre": list(sec.shear_centre), "n_points": check.n_points, "matched": int(sum(len(p) for _, p in sampled[0])),
                      "solid_points": solid.n_points}
    return out


# ============================================================================ report and checks
def report(r):
    print(f"section check: {r['section']['n_points']} ply points ({r['section']['matched']} matched to the shell), "
          f"solid {r['section']['solid_points']}")
    for name, c in r["cases"].items():
        crit = c["summary"]["critical"]
        print(f"\n{name}: critical z = {crit['z']:.3f} m, {crit['name']} ply {crit['ply']}, {crit['criterion']} "
              f"{crit['mode']}, exposure {crit['exposure']:.3f} (RF {crit['reserve_factor']:.2f})")
        print("   z [m]   shell   ASM    (largest exposure of the row, same points)")
        for z, fs, fa in zip(r["rows"], c["shell"]["exposure"], c["asm"]["exposure"]):
            print(f"  {z:6.3f}  {fs:6.3f}  {fa:6.3f}")
    for name, s in r["solid"].items():
        print(f"\n{name} at z = {MID}: critical exposure thin-walled {s['thin']['exposure']:.3f}, "
              f"solid {s['solid']['exposure']:.3f}; away from junctions {s['thin_away']['exposure']:.3f} / "
              f"{s['solid_away']['exposure']:.3f}")


def check(r):
    c = Checks("14 blade section strength")
    rows = np.array(r["rows"])
    valid = (rows > VALID[0]) & (rows < VALID[1])
    for name, tip in CASES.items():
        cs = r["cases"][name]
        # the saved forces are the statics of the cantilever
        z = np.array(r["stations"])
        F = np.array(cs["forces"])
        expected = np.array([cut_forces(zz, tip, r["section"]["shear_centre"]) for zz in z])
        c.close(f"{name}: section forces from statics", float(np.abs(F - expected).max()), 0.0, 1e-6,
                ref="statics", scale=float(np.abs(expected).max()))
        # ply stresses point by point at mid-span
        sh, a = np.array(cs["mid"]["shell"]), np.array(cs["mid"]["asm"])
        parts = np.array(cs["mid"]["part"])
        for comp, label in ((0, "sigma_11"), (1, "sigma_22"), (5, "tau_12")):
            for p in ("spar_cap", "le_panel", "te_panel", "web_fwd", "web_aft"):
                if comp == 1 and p == "spar_cap":
                    continue  # the 0-degree caps carry almost no transverse stress (under 0.1 % of sigma_11)
                sel = parts == p
                scale = np.abs(sh[sel, comp]).max()
                err = np.sqrt(np.mean((sh[sel, comp] - a[sel, comp]) ** 2)) / scale
                c.close(f"{name}: {label} in {p}, RMS / max", err, 0.0, 0.05, ref="shell", scale=1.0)
        # exposures at every cut, away from the clamp and the tip rib
        for crit in ("max_stress", "tsai_wu", "hashin", "puck"):
            es, ea = np.array(cs["shell"][crit])[valid], np.array(cs["asm"][crit])[valid]
            c.close(f"{name}: worst {crit} exposure error along the span", np.max(np.abs(ea / es - 1.0)), 0.0, 0.05,
                    ref="shell", scale=1.0)
        # the critical point, row by row
        pairs = [(s, a) for s, a, v in zip(cs["shell_critical"], cs["asm_critical"], valid) if v]
        same = [s["part"] == a["part"] and s["criterion"] == a["criterion"] for s, a in pairs]
        c.true(f"{name}: same critical part and criterion as the shell at every valid row", all(same), ref="shell",
               value=float(np.mean(same)))
        # the skins' two middle plies are both -45: near the tip the shell and the check pick
        # one or the other of the pair, with exposures 0.1 % apart
        ply = [s["ply"] == a["ply"] for s, a in pairs]
        c.true(f"{name}: same critical ply at three quarters of the valid rows or more", np.mean(ply) >= 0.75,
               ref="shell", value=float(np.mean(ply)))
        # the root governs a cantilever
        c.true(f"{name}: critical cut is the root", r["cases"][name]["summary"]["critical"]["z"] == 0.0)
        s = r["solid"][name]
        c.close(f"{name}: solid section, critical exposure away from the junctions", s["solid_away"]["exposure"],
                s["thin_away"]["exposure"], 0.03, ref="thin-walled")
        c.true(f"{name}: solid section, same governing criterion", s["solid_away"]["criterion"] == s["thin_away"]["criterion"],
               ref="thin-walled")
    env = np.maximum(r["cases"]["flapwise"]["asm_all"], r["cases"]["edgewise"]["asm_all"])
    c.true("both cases give an exposure at every cut", bool(np.all(env > 0)))
    return c


# ============================================================================ figures
def figure_span(r):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)
    z = np.array(r["stations"])
    rows = np.array(r["rows"])
    env = np.zeros_like(z)
    for i, (name, cs) in enumerate(r["cases"].items()):
        f = np.array(cs["asm_all"])
        env = np.maximum(env, f)
        ax[0].plot(z, f, color=SERIES[i], label=f"{name}, section check")
        ax[0].plot(rows, cs["shell"]["exposure"], "o", color=SERIES[i], mfc="none", ms=5, label=f"{name}, shell")
    ax[0].plot(z, env, color=REFERENCE, ls="--", lw=1.2, label="larger of the two cases")
    ax[0].axvspan(0.0, VALID[0], color="#f0efec", zorder=0)
    ax[0].axvspan(VALID[1], LENGTH, color="#f0efec", zorder=0)
    ax[0].set(xlabel="z [m]", ylabel="largest exposure in the section", title="Largest exposure at each cut")
    ax[0].legend(fontsize=7.5)
    cs = r["cases"]["flapwise"]
    for i, crit in enumerate(("max_stress", "tsai_wu", "hashin", "puck")):
        ax[1].plot(rows, cs["asm"][crit], color=SERIES[i], label=crit.replace("_", " "))
        ax[1].plot(rows, cs["shell"][crit], "o", color=SERIES[i], mfc="none", ms=4)
    ax[1].set(xlabel="z [m]", ylabel="largest exposure", title="Flapwise case by criterion (markers: shell)")
    ax[1].legend(fontsize=7.5)
    for a in ax:
        a.set_xlim(0, LENGTH)
    save(fig, __file__, "exposure_at_each_cut.png")


def figure_stresses(r):
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True)
    cs = r["cases"]["flapwise"]
    sh, a = np.array(cs["mid"]["shell"]) / 1e6, np.array(cs["mid"]["asm"]) / 1e6
    parts = np.array(cs["mid"]["part"])
    colours = {"spar_cap": SERIES[0], "le_panel": SERIES[1], "te_panel": SERIES[2], "web_fwd": SERIES[3],
               "web_aft": SERIES[4]}
    for k, (comp, label) in enumerate(((0, "σ₁₁"), (1, "σ₂₂"), (5, "τ₁₂"))):
        for p, col in colours.items():
            sel = parts == p
            ax[k].plot(sh[sel, comp], a[sel, comp], "o", ms=3, color=col, label=p.replace("_", " ") if k == 0 else None)
        lim = np.abs(np.concatenate([sh[:, comp], a[:, comp]])).max() * 1.05
        ax[k].plot([-lim, lim], [-lim, lim], color=REFERENCE, ls="--", lw=1)
        ax[k].set(xlabel=f"shell {label} [MPa]", ylabel=f"section check {label} [MPa]", title=label)
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Ply stresses at z = {cs['mid']['z']:.2f} m, flapwise case, every ply of every element",
                 fontsize=9.5, color=TEXT_SECONDARY)
    save(fig, __file__, "ply_stresses_vs_shell.png")


def figure_root(r):
    fig, ax = plt.subplots(2, 1, figsize=(7.6, 5.4), constrained_layout=True)
    P = np.array(r["section"]["points"])
    for a, (name, cs) in zip(ax, r["cases"].items()):
        for s, e in zip(r["section"]["start"], r["section"]["end"]):
            a.plot(P[[s, e], 0], P[[s, e], 1], color="#c9c8c3", lw=0.8, zorder=1)
        pos = np.array(cs["root"]["position"])
        f = np.array(cs["root"]["exposure"])
        # the largest exposure through the thickness at each point of the section
        key = np.round(pos, 9)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        fmax = np.full(len(uniq), -np.inf)
        np.maximum.at(fmax, inv.ravel(), f)
        sc = a.scatter(uniq[:, 0], uniq[:, 1], c=fmax, s=10, cmap=SEQUENTIAL, zorder=2)
        crit = cs["root"]["critical"]
        a.plot(*crit["position"], marker="*", ms=14, color=SERIES[1], mec="white", zorder=3)
        a.annotate(f"{crit['name']}, ply {crit['ply']}, {crit['criterion']} {crit['mode']}\nexposure {crit['exposure']:.2f}",
                   crit["position"], xytext=(12, 10), textcoords="offset points", fontsize=7.5, color=TEXT_SECONDARY)
        a.set_aspect("equal")
        a.set(title=f"{name}: exposure over the root section", xlabel="x [m]", ylabel="y [m]")
        fig.colorbar(sc, ax=a, shrink=0.8, label="exposure")
    save(fig, __file__, "root_section_exposure.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", action="store_true", help="redraw from data/results.json")
    args = parser.parse_args()
    if args.reuse and DATA.exists():
        r = json.loads(DATA.read_text())
    else:
        from laminate_constitutive_model import create_default_library

        r = solve(create_default_library())
        DATA.parent.mkdir(exist_ok=True)
        DATA.write_text(json.dumps(r, indent=1))
    report(r)
    figure_span(r)
    figure_stresses(r)
    figure_root(r)
    check(r).report()


if __name__ == "__main__":
    main()
