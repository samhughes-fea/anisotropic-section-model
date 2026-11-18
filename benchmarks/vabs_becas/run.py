"""Run ASM's solid section on the VABS and BECAS benchmark sections and write the comparison.

    python benchmarks/vabs_becas/run.py            # writes results.json and RESULTS.md here

For every case: the published terms, ASM on the source's own mesh where it is known (Q4,
BECAS user guide), and ASM converged on quadratic meshes (Q8 and T6, the element the
section-properties port would use) with its mesh convergence. Closed-form values are added
where they exist, so a difference between ASM and a published number can be put on the right
side.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from anisotropic_section_model.solid import SolidSectionAnalysis  # noqa: E402

from cases import CASES  # noqa: E402
from reference import BECAS_FROM_ASM, MATERIALS, PUBLISHED, VABS_FROM_ASM  # noqa: E402

#: cases whose source mesh is known (BECAS user guide 10 x 10 Q4; GXBeam's C1 / C2 meshes)
SOURCE_MESH = {"S1", "S2", "S3_0", "S3_22.5", "S3_90"}
LEVELS = (1, 2, 4)


def order(case: str):
    return BECAS_FROM_ASM if PUBLISHED[case]["order"] == "becas" else VABS_FROM_ASM


def term(K, name: str) -> float:
    return float(K[int(name[1]) - 1, int(name[2]) - 1])


def rounding(value: float) -> float:
    """Half a unit in the last significant digit of ``value`` as stored, relative: what the
    published rounding allows (a value printed as a round number, such as EA = 1.0, shows
    as coarse)."""
    s = f"{abs(value):.6e}".split("e")[0].rstrip("0").rstrip(".")
    digits = len(s.replace(".", ""))
    return 0.5 * 10.0 ** (1 - digits) / float(s)


def _square_torsion(G: float, a: float, terms: int = 200) -> float:
    s = sum(math.tanh(n * math.pi / 2) / n ** 5 for n in range(1, 2 * terms, 2))
    return G * a ** 4 / 3.0 * (1.0 - 192.0 / math.pi ** 5 * s)


def exact(case: str):
    """Closed-form terms (in the case's published order) for the sections that have them."""
    if case == "S1":
        E, G, a = 100.0, 41.667, 0.1
        return {"K33": E * a * a, "K44": E * a ** 4 / 12, "K55": E * a ** 4 / 12, "K66": _square_torsion(G, a)}
    if case in ("C1", "tube"):
        E, nu = (100.0, 0.2) if case == "C1" else (73e9, 0.33)
        G = MATERIALS["iso1"][3] if case == "C1" else E / (2 * (1 + nu))
        ri, ro = (0.09, 0.1) if case == "C1" else (0.1, 0.3)
        A, I = math.pi * (ro ** 2 - ri ** 2), math.pi / 4 * (ro ** 4 - ri ** 4)
        if case == "C1":
            return {"K33": E * A, "K44": E * I, "K55": E * I, "K66": G * 2 * I}
        return {"K11": E * A, "K44": G * 2 * I, "K55": E * I, "K66": E * I}
    return {}


def analyse(case: str, n: int, kind: str):
    t = time.perf_counter()
    mesh = CASES[case](n, kind)
    a = SolidSectionAnalysis(mesh)
    p = order(case)
    K = a.stiffness[np.ix_(p, p)]
    return dict(kind=kind, n=n, elements=len(mesh.elements), dof=a.n_dof, seconds=time.perf_counter() - t,
                K=K.tolist(), shear_centre=list(a.shear_centre), tension_centre=list(a.tension_centre),
                mass=a.mass_properties)


def observed_order(values, h):
    """Convergence order from three levels halving ``h``: ``log2(|f1 - f2| / |f2 - f4|)``."""
    d1, d2 = abs(values[0] - values[1]), abs(values[1] - values[2])
    if d2 <= 1e-9 * abs(values[2]) or d1 <= 1e-9 * abs(values[2]):
        return None
    return math.log(d1 / d2, 2.0)


def run():
    out = {}
    for case in CASES:
        pub = PUBLISHED[case]
        runs = {}
        if case in SOURCE_MESH:
            runs["Q4 source mesh"] = analyse(case, 1, "Q4")
        for kind in ("Q8", "T6"):
            runs[kind] = [analyse(case, n, kind) for n in LEVELS]
        rows = []
        ex = exact(case)
        for name, ref in pub["terms"].items():
            row = dict(term=name, published=ref, rounding=rounding(ref))
            if "Q4 source mesh" in runs:
                row["q4_source"] = term(np.array(runs["Q4 source mesh"]["K"]), name)
            for kind in ("Q8", "T6"):
                vals = [term(np.array(r["K"]), name) for r in runs[kind]]
                row[kind.lower()] = vals[-1]
                row[kind.lower() + "_order"] = observed_order(vals, None)
            if name in ex:
                row["exact"] = ex[name]
            for alt, terms in pub.get("alternatives", {}).items():
                row.setdefault("alternatives", {})[alt] = terms.get(name)
            rows.append(row)
        if case in SOURCE_MESH:
            runs["Q4 refined"] = [analyse(case, n, "Q4") for n in (2, 4, 8)]
        out[case] = dict(tool=pub["tool"], order=pub["order"], source=pub["source"], via=pub["via"],
                         rows=rows, centres=pub.get("centres"), published_mass=pub.get("mass"), runs=runs)
        print(f"{case}: done", flush=True)
    return out


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def report(res) -> str:
    lines = ["# ASM solid section against VABS and BECAS: results", "",
             "Generated by `run.py`. Differences are ASM / reference - 1, in %. `Q4 src` is ASM on the "
             "source's own mesh (BECAS user guide 10 x 10 Q4); `Q8` and `T6` are ASM on the finest "
             f"quadratic mesh (level {LEVELS[-1]}). `exact` is the closed form where one exists; "
             "`rnd` is half a unit in the published value's last digit.", ""]
    for case, r in res.items():
        lines += [f"## {case} ({r['tool']}, {r['order']} order)", "", f"Source: {r['source']} (via {r['via']}).", ""]
        has_q4 = any("q4_source" in row for row in r["rows"])
        has_ex = any("exact" in row for row in r["rows"])
        head = ["term", "published", "rnd %"] + (["Q4 src %"] if has_q4 else []) + ["Q8 %", "T6 %", "Q8 order"]
        if has_ex:
            head += ["exact", "published vs exact %", "Q8 vs exact %"]
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for row in r["rows"]:
            cells = [row["term"], f"{row['published']:.6g}", f"{100 * row['rounding']:.4f}"]
            if has_q4:
                cells.append(f"{pct(row['q4_source'], row['published']):+.3f}")
            cells += [f"{pct(row['q8'], row['published']):+.3f}", f"{pct(row['t6'], row['published']):+.3f}",
                      "-" if row["q8_order"] is None else f"{row['q8_order']:.1f}"]
            if has_ex:
                if "exact" in row:
                    cells += [f"{row['exact']:.5g}", f"{pct(row['published'], row['exact']):+.3f}",
                              f"{pct(row['q8'], row['exact']):+.3f}"]
                else:
                    cells += ["", "", ""]
            lines.append("| " + " | ".join(cells) + " |")
        if any(row.get("alternatives") for row in r["rows"]):
            alts = list(r["rows"][0]["alternatives"])
            lines += ["", "ASM Q8 against the other published VABS sets:", "",
                      "| term | " + " | ".join(alts) + " |", "|---|" + "---|" * len(alts)]
            for row in r["rows"]:
                lines.append(f"| {row['term']} | " + " | ".join(
                    f"{pct(row['q8'], row['alternatives'][a]):+.3f}" for a in alts) + " |")
        if r["centres"]:
            fin = r["runs"]["Q8"][-1]
            lines += ["", "| centre | published | ASM Q8 |", "|---|---|---|"]
            for key in ("shear", "tension"):
                pub = r["centres"][key]
                asm = fin[key + "_centre"]
                lines.append(f"| {key} | ({pub[0]:.4g}, {pub[1]:.4g}) | ({asm[0]:.4g}, {asm[1]:.4g}) |")
        if r["published_mass"]:
            fin = r["runs"]["Q8"][-1]["mass"]
            lines += ["", "| mass term | published | ASM Q8 | % |", "|---|---|---|---|"]
            for key, v in r["published_mass"].items():
                lines.append(f"| {key} | {v:.4g} | {fin[key]:.4g} | {pct(fin[key], v):+.3f} |")
        if "Q4 refined" in r["runs"]:
            q4 = [r["runs"]["Q4 source mesh"]] + r["runs"]["Q4 refined"]
            q8 = np.array(r["runs"]["Q8"][-1]["K"])
            names = [row["term"] for row in r["rows"]]
            lines += ["", "BECAS's Q4 mesh refined, against ASM Q8 level 4 (%): the published offset is "
                      "the coarse bilinear mesh, and it closes as the Q4 mesh is refined.", "",
                      "| Q4 mesh | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
            for x in q4:
                K = np.array(x["K"])
                lines.append(f"| {10 * x['n']} x {10 * x['n']} | " + " | ".join(
                    f"{pct(term(K, t), term(q8, t)):+.3f}" if abs(term(q8, t)) > 1e-12 * abs(q8).max() else "0"
                    for t in names) + " |")
        meshes = []
        for kind in ("Q8", "T6"):
            meshes += [f"{kind} level {x['n']}: {x['elements']} elements, {x['dof']} dof, {x['seconds']:.1f} s"
                       for x in r["runs"][kind]]
        lines += ["", "Meshes: " + "; ".join(meshes) + ".", ""]
    return "\n".join(lines)


def main():
    res = run()
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (HERE / "RESULTS.md").write_text(report(res))


if __name__ == "__main__":
    main()
