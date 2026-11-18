"""Solid sections against BECAS and VABS: the published validation cases, term by term.

``SolidSectionAnalysis`` solves the cross-section's 2D elasticity problem by finite
elements, the method of BECAS and (in its Timoshenko form) VABS. This example runs it on
the sections those two tools publish results for, using the benchmark in
``benchmarks/vabs_becas`` (cases, published values and their provenance):

- the BECAS user guide's squares: isotropic (S1), two materials (S2) and orthotropic at
  0, 22.5 and 90 degrees (S3), on BECAS's own 10 x 10 bilinear mesh and converged;
- BECAS's tube (C1) and half tube (C2), with the shear and tension centres;
- VABS: Chen et al.'s thick aluminium tube, and Yu et al.'s two-ply composite pipe with
  extension-twist and shear-bending couplings.

Figures: the meshes; the difference from every published term; mesh convergence, which
explains the one place the published values and the converged answer part.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "benchmarks" / "vabs_becas"))

import numpy as np  # noqa: E402

from _shared.checks import Checks  # noqa: E402
from _shared.plotting import DIVERGING, REFERENCE, SERIES, TEXT_SECONDARY, plt, save  # noqa: E402
from anisotropic_section_model.solid import SolidSectionAnalysis  # noqa: E402

from cases import CASES  # noqa: E402  (benchmarks/vabs_becas)
from reference import BECAS_FROM_ASM, PUBLISHED, VABS_FROM_ASM  # noqa: E402
from run import exact, rounding, term  # noqa: E402

DATA = HERE / "data" / "results.json"
LEVELS = (1, 2, 4)
SQUARES = ("S1", "S2", "S3_0", "S3_22.5", "S3_90")
#: how close converged ASM must come to each published set (fraction). The squares carry
#: BECAS's coarse-mesh error (about 0.85 % on shear and torsion, see the README).
TOLERANCE = {"S1": 0.015, "S2": 0.015, "S3_0": 0.015, "S3_22.5": 0.015, "S3_90": 0.015,
             "C1": 2e-3, "C2": 2e-3, "tube": 5e-4, "pipe": 3e-3}
#: published terms reported but not gated against the source: VABS's tube GJ is 2 % above
#: the exact annulus value, which ASM matches (it is gated against the exact value instead)
NOT_GATED = {("tube", "K44")}


def stiffness(case, n, kind):
    a = SolidSectionAnalysis(CASES[case](n, kind))
    p = BECAS_FROM_ASM if PUBLISHED[case]["order"] == "becas" else VABS_FROM_ASM
    return a, a.stiffness[np.ix_(p, p)]


def run():
    out = {}
    for case in CASES:
        pub = PUBLISHED[case]
        rec = {"tool": pub["tool"], "source": pub["source"], "terms": {}, "exact": exact(case),
               "centres": pub.get("centres")}
        q8 = []
        for n in LEVELS:
            a, K = stiffness(case, n, "Q8")
            q8.append((a, K))
        a, K = q8[-1]
        rec["shear_centre"], rec["tension_centre"] = list(a.shear_centre), list(a.tension_centre)
        if case in SQUARES:
            q4 = stiffness(case, 1, "Q4")[1]
        for name, ref in pub["terms"].items():
            row = {"published": ref, "rounding": rounding(ref), "q8": [term(k, name) for _, k in q8]}
            if case in SQUARES:
                row["q4_source"] = term(q4, name)
            for alt, terms in pub.get("alternatives", {}).items():
                row.setdefault("alternatives", {})[alt] = terms[name]
            rec["terms"][name] = row
        out[case] = rec
        print(f"{case}: done", flush=True)
    # the S1 square's torsion stiffness on refined Q4 and Q8 meshes, against BECAS and the series
    conv = {"Q4": [], "Q8": []}
    for kind, levels in (("Q4", (1, 2, 4, 8)), ("Q8", (1, 2, 4))):
        for n in levels:
            conv[kind].append((10 * n, term(stiffness("S1", n, kind)[1], "K66")))
    out["S1_convergence"] = conv
    # meshes for the figure (node coordinates and corner connectivity, ply angle per element)
    draw = {}
    for case in ("S3_22.5", "C2", "tube", "pipe"):
        m = CASES[case](1, "Q8")
        draw[case] = {"nodes": m.nodes.tolist(), "elements": [list(map(int, e[:4])) for e in m.elements],
                      "theta": [float(t) for t in m.theta], "material": [int(k) for k in m.material_ids]}
    out["meshes"] = draw
    return out


# ----------------------------------------------------------------------------- checks
def check(r) -> Checks:
    c = Checks("11 solid sections against BECAS and VABS")
    for case in CASES:
        rec = r[case]
        tool = rec["tool"]
        for name, row in rec["terms"].items():
            conv = row["q8"][-1]
            if (case, name) not in NOT_GATED:
                c.close(f"{case} {name}, converged Q8 against {tool}", conv, row["published"], TOLERANCE[case], tool)
            if "q4_source" in row:
                c.close(f"{case} {name}, BECAS's own 10 x 10 Q4 mesh", row["q4_source"], row["published"],
                        row["rounding"] + 1e-4, "BECAS, printed digits")
            if name in rec["exact"]:
                c.close(f"{case} {name}, converged Q8 against the closed form", conv, rec["exact"][name], 1e-5,
                        "closed form")
        if rec["centres"]:
            for key in ("shear", "tension"):
                pub, asm = rec["centres"][key], rec[key + "_centre"]
                scale = 0.1  # the sections' size
                c.close(f"{case} {key} centre y", asm[0], pub[0], 2e-3, tool, scale=scale)
    tube = r["tube"]["terms"]["K44"]
    c.true(f"VABS tube GJ is {100 * (tube['published'] / r['tube']['exact']['K44'] - 1):.1f} % above the exact "
           "annulus (reported, not gated)", tube["published"] > r["tube"]["exact"]["K44"], "Chen et al. 2010")
    q4 = r["S1_convergence"]["Q4"]
    rate = math.log(abs(q4[-2][1] - r["S1"]["exact"]["K66"]) / abs(q4[-1][1] - r["S1"]["exact"]["K66"]), 2)
    c.true(f"BECAS's bilinear mesh refined converges on the exact S1 torsion at second order ({rate:.2f})",
           1.8 <= rate <= 2.2, "Saint-Venant series", rate)
    return c


# ----------------------------------------------------------------------------- figures
def figure_meshes(r):
    titles = {"S3_22.5": "BECAS S3: orthotropic square, 22.5°", "C2": "BECAS C2: half tube",
              "tube": "VABS: thick aluminium tube", "pipe": "VABS: two-ply composite pipe"}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1, 0.8, 1, 1.9]})
    for ax, (case, m) in zip(axes, r["meshes"].items()):
        P = np.array(m["nodes"])
        th = np.array(m["theta"])
        lim = max(np.max(np.abs(th)), 1.0)
        for e, t in zip(m["elements"], th):
            q = P[e + [e[0]]]
            ax.fill(q[:, 0], q[:, 1], color=DIVERGING(0.5 + 0.5 * t / 90.0) if lim > 1 else "#d8d7d2",
                    ec="white", lw=0.25)
        rec = r[case]
        ax.plot(*rec["shear_centre"], "o", color=SERIES[1], ms=6, label="shear centre")
        ax.plot(*rec["tension_centre"], "+", color=REFERENCE, ms=9, mew=1.5, label="tension centre")
        if rec["centres"]:
            ax.plot(*rec["centres"]["shear"], "x", color=REFERENCE, ms=8, mew=1.2, label=f"{rec['tool']}")
        ax.set_aspect("equal")
        ax.set_title(titles[case], loc="left", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.grid(False)
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("The benchmark sections, coloured by ply angle (blue −90°, red +90°)", x=0.0, ha="left",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.text(0.0, -0.03, "Q8 meshes at the base level. The pipe's flats are [90/0] and its arcs [45/−45], "
             "inner ply first; the tube and half tube are isotropic.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "benchmark_meshes.png")


def figure_differences(r):
    fig, ax = plt.subplots(figsize=(14, 4.4))
    x, ticks, labels, seen = 0, [], [], set()
    for ci, case in enumerate(CASES):
        rec = r[case]
        start = x
        for name, row in rec["terms"].items():
            d8 = 100 * abs(row["q8"][-1] / row["published"] - 1)
            ax.plot(x, max(d8, 1e-4), "o", color=SERIES[0], ms=6, label="converged Q8" if x == 0 else None)
            if "q4_source" in row:
                d4 = 100 * abs(row["q4_source"] / row["published"] - 1)
                ax.plot(x, max(d4, 1e-4), "s", color=SERIES[1], mfc="none", ms=6,
                        label="BECAS's own 10 x 10 Q4 mesh" if x == 0 else None)
            ax.plot([x - 0.35, x + 0.35], [100 * row["rounding"]] * 2, color=TEXT_SECONDARY, lw=1.5,
                    label="half a unit in the published last digit" if x == 0 else None)
            if name in rec["exact"]:
                de = 100 * abs(row["published"] / rec["exact"][name] - 1)
                ax.plot(x, max(de, 1e-4), "x", color=REFERENCE, ms=7, mew=1.5,
                        label=None if "x" in seen else "published against the closed form")
                seen.add("x")
            ticks.append(x)
            labels.append(name)
            x += 1
        if ci % 2:
            ax.axvspan(start - 0.5, x - 0.5, color="#f0efec", lw=0)
        ax.text(0.5 * (start + x - 1), 12, f"{case}\n{rec['tool']}", ha="center", va="bottom", fontsize=8)
        x += 1
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 60)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
    ax.set_ylabel("|ASM / published − 1|  [%]")
    ax.set_title("On BECAS's own mesh ASM matches every published term to its printed digits", loc="left")
    ax.legend(fontsize=7.5, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.2))
    ax.grid(axis="x", visible=False)
    fig.text(0.0, -0.08, "Converged: Q8, the finest of three levels. Points at 1e-4 % are at or below it. Published "
             "values: BECAS user guide (Blasques 2012) section 4.2; VABS: Chen et al. 2010, Yu et al. 2012 "
             "(see benchmarks/vabs_becas/README.md for provenance).", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "benchmark_differences.png")


def figure_convergence(r):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax = axes[0]
    ex, pub = r["S1"]["exact"]["K66"], r["S1"]["terms"]["K66"]["published"]
    for kind, col in (("Q4", SERIES[1]), ("Q8", SERIES[0])):
        n, k = np.array(r["S1_convergence"][kind]).T
        ax.plot(n, k / ex, "o-", color=col, label=f"ASM {kind}")
    ax.axhline(pub / ex, color=TEXT_SECONDARY, ls="--", lw=1.2, label="BECAS published (10 x 10 Q4)")
    ax.axhline(1.0, color=REFERENCE, ls=":", lw=1.2, label="Saint-Venant series")
    ax.set_xscale("log", base=2)
    ax.set_xticks([10, 20, 40, 80])
    ax.set_xticklabels(["10", "20", "40", "80"])
    ax.set_xlabel("elements along each side")
    ax.set_ylabel("S1 torsion stiffness / exact")
    ax.set_title("BECAS's square values carry their mesh's error", loc="left")
    ax.legend(fontsize=7.5)
    ax = axes[1]
    pipe = r["pipe"]["terms"]
    for (name, col) in zip(("K14", "K25", "K36", "K33"), SERIES):
        row = pipe[name]
        ax.plot(LEVELS, 100 * (np.array(row["q8"]) / row["published"] - 1), "o-", color=col, label=name)
    ax.axhline(0.0, color=REFERENCE, lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(LEVELS)
    ax.set_xticklabels([str(v) for v in LEVELS])
    ax.set_xlabel("mesh level (1: 4 elements through each wall)")
    ax.set_ylabel("ASM / VABS − 1 [%]")
    ax.set_title("Pipe couplings converge onto VABS", loc="left")
    ax.legend(fontsize=7.5)
    fig.text(0.0, -0.04, "Left: unit square S1, E = 100, ν = 0.2. Right: Yu et al. 2012 pipe, K14 extension-twist, "
             "K25 and K36 shear-bending, in VABS's order [N, V2, V3, T, M2, M3].", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "benchmark_convergence.png")


def report(r):
    for case in CASES:
        rec = r[case]
        parts = []
        for name, row in rec["terms"].items():
            s = f"{name} {100 * (row['q8'][-1] / row['published'] - 1):+.3f}"
            if "q4_source" in row:
                s += f" (Q4 {100 * (row['q4_source'] / row['published'] - 1):+.3f})"
            parts.append(s)
        print(f"{case:8} {rec['tool']:5} % against published: " + ", ".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", action="store_true", help="redraw from data/results.json")
    args = parser.parse_args()
    if args.reuse and DATA.exists():
        r = json.loads(DATA.read_text())
    else:
        r = run()
        DATA.parent.mkdir(exist_ok=True)
        DATA.write_text(json.dumps(r, indent=1, default=float))
    report(r)
    figure_meshes(r)
    figure_differences(r)
    figure_convergence(r)
    check(r).report()


if __name__ == "__main__":
    main()
