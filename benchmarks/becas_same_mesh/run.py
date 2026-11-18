"""ASM's solid section against BECAS on the same meshes: the full 6 x 6 and both centres.

    python benchmarks/becas_same_mesh/run.py            # ASM; BECAS too where it is available
    python benchmarks/becas_same_mesh/run.py --becas    # fail if BECAS cannot be run

Cases (``becas_cases.py``): BECAS's own validation meshes (``EXAMPLES``, needs ``BECAS_HOME``) and
the composite sections built here (``BUILT``), written to BECAS input with ``becas_io``.

BECAS is licensed and never part of this repository. With ``BECAS_HOME`` set and MATLAB or
GNU Octave on ``PATH`` (or ``BECAS_RUNTIME`` naming one), each case is run through BECAS with
``run_becas.m`` and the resulting numbers are stored in ``becas_reference.json`` (numbers
only, no BECAS files). Without BECAS the stored numbers are used, so ``RESULTS.md`` can be
regenerated and ``tests/test_benchmark_becas_same_mesh.py`` can gate the built cases anywhere.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "vabs_becas"))

from anisotropic_section_model.solid import SolidSectionAnalysis  # noqa: E402

from becas_io import becas_home, read_becas_results, read_vabs_point, read_vabs_timoshenko, write_becas  # noqa: E402
from becas_cases import BUILT, EXAMPLES, example  # noqa: E402
from becas_published import USER_GUIDE  # noqa: E402
from reference import BECAS_FROM_ASM, PUBLISHED  # noqa: E402

REFERENCE = HERE / "becas_reference.json"
WORK = HERE / "_work"                     # generated BECAS input and output (git-ignored)
#: BECAS order -> VABS Timoshenko order: VABS [F1 F2 F3 M1 M2 M3] = BECAS [N Vx Vy Mz Mx My]
VABS_FROM_BECAS = [2, 0, 1, 5, 3, 4]
#: a term counts as a coupling worth comparing when |K_ij| / sqrt(K_ii K_jj) exceeds this
SIGNIFICANT = 1e-6
#: The ``BECASresults.txt`` shipped in some example folders belongs to another folder's input.
#: Case -> folder whose shipped results were made on this case's input, found by running ASM
#: on every example mesh (agreement to 1e-11 against the right one, tens of % against the
#: folder the file sits in): S2_1's file is the E1/E2 = 10 square (S2_10), and the 22.5 and 45
#: deg squares' files are the 0 deg square's.
SHIPPED_RESULTS = {"S1": "S1", "S2_10": "S2_1", "S3_0": "S3_0", "C1": "C1", "C2": "C2", "C4": "C4"}
NOTES = {
    "S1": "S1's shipped result was made with G = 41.667; its MATPROPS.in now holds G = E / 2.4 = "
          "41.6667. The shear and torsion terms (proportional to G) differ by -8.0e-6 = "
          "41.6667 / 41.667 - 1; the terms that depend only on E agree to 1e-9.",
    "S2_10": "Compared with the BECASresults.txt in BECAS's S2_1 folder: that file is the E1/E2 = 10 "
             "result (S2_1's two materials are identical).",
    "S3_22.5": "The BECASresults.txt in this folder is the 0 deg square's result, so it is not used.",
    "S3_45": "The BECASresults.txt in this folder is the 0 deg square's result, so it is not used.",
    "T1": "User-guide table: the rows labelled K26 and K35 are swapped (becas_published.py).",
    "T2": "BECAS's T2 MATPROPS.in swaps G13 / G23 and nu13 / nu23 of orthotropic #1 relative to "
          "T3 and the user guide's table; ASM uses the file as given.",
    "C3_1": "User-guide table: columns relabelled (becas_published.py); VABS on the same mesh is "
            "the full-precision check.",
}


def runtime() -> Optional[str]:
    r = os.environ.get("BECAS_RUNTIME")
    if r:
        return r
    for exe in ("matlab", "octave-cli", "octave"):
        p = shutil.which(exe)
        if p:
            return p
    return None


def run_becas(folder: Path, out: Path) -> Dict:
    """BECAS (via ``run_becas.m``) on the input in ``folder``."""
    home, exe = becas_home(), runtime()
    if home is None or exe is None:
        raise RuntimeError("BECAS needs BECAS_HOME and MATLAB or Octave")
    src = (home / "src" / "matlab").as_posix()
    call = f"run_becas('{src}', '{folder.resolve().as_posix()}', '{out.resolve().as_posix()}')"
    name = Path(exe).name.lower()
    if name.startswith("matlab"):
        cmd = [exe, "-batch", f"addpath('{HERE.as_posix()}'); {call}"]
    else:
        cmd = [exe, "--no-gui", "--quiet", "--eval", f"addpath('{HERE.as_posix()}'); {call}"]
    t = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=str(HERE), timeout=3600)
    res = read_becas_results(out)
    res.update(seconds=time.perf_counter() - t, runtime=name.split(".")[0])
    return res


def mesh_for(case: str):
    return example(case) if case in EXAMPLES else BUILT[case](1)


def becas_input(case: str, mesh) -> Path:
    if case in EXAMPLES:
        return becas_home() / "examples" / case
    return write_becas(mesh, WORK / case)


def asm(mesh) -> Dict:
    t = time.perf_counter()
    a = SolidSectionAnalysis(mesh)
    p = BECAS_FROM_ASM
    return dict(K=a.stiffness[np.ix_(p, p)].tolist(), shear_centre=list(a.shear_centre),
                tension_centre=list(a.tension_centre), seconds=time.perf_counter() - t,
                elements=len(mesh.elements), nodes=int(mesh.nodes.shape[0]), dof=a.n_dof,
                kinds=sorted(set(mesh.kinds)))


def compare(K, R) -> Dict:
    """Term by term ``K / R - 1`` on the significant terms of ``R``, and the largest
    difference scaled by the diagonal, ``|K_ij - R_ij| / sqrt(R_ii R_jj)``, over all 36."""
    K, R = np.asarray(K, float), np.asarray(R, float)
    d = np.sqrt(np.abs(np.outer(np.diag(R), np.diag(R))))
    scaled = np.abs(K - R) / d
    terms = {}
    for i in range(6):
        for j in range(i, 6):
            if abs(R[i, j]) / d[i, j] > SIGNIFICANT:
                terms[f"K{i + 1}{j + 1}"] = dict(asm=K[i, j], ref=R[i, j], rel=K[i, j] / R[i, j] - 1.0)
    return dict(terms=terms, max_scaled=float(scaled.max()),
                max_rel=float(max(abs(t["rel"]) for t in terms.values())))


def centre_diff(a, b, size: float) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]) / size)


def size_of(mesh) -> float:
    lo, hi = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)
    return float(np.max(hi - lo))


def run(require_becas: bool = False):
    stored = json.loads(REFERENCE.read_text()) if REFERENCE.exists() else {}
    can_run = becas_home() is not None and runtime() is not None
    if require_becas and not can_run:
        raise SystemExit("BECAS cannot be run here: set BECAS_HOME and put MATLAB or Octave on PATH")
    out = {}
    cases = (list(EXAMPLES) if becas_home() else []) + list(BUILT)
    for case in cases:
        mesh = mesh_for(case)
        row = dict(case=case, description=EXAMPLES.get(case) or (BUILT[case].__doc__ or "").split("\n")[0],
                   source="BECAS example mesh" if case in EXAMPLES else "built here", asm=asm(mesh))
        size = size_of(mesh)
        if can_run:
            folder = becas_input(case, mesh)
            WORK.mkdir(exist_ok=True)
            stored[case] = run_becas(folder, WORK / f"{case}_becas_out.txt")
            stored[case].update(elements=len(mesh.elements), nodes=int(mesh.nodes.shape[0]))
        refs = {}
        if case in stored:
            b = stored[case]
            refs["BECAS, same mesh"] = dict(K=b["K"], shear_centre=b["shear_centre"],
                                            tension_centre=b["tension_centre"], runtime=b.get("runtime"))
        shipped = (becas_home() / "examples" / SHIPPED_RESULTS[case] / "BECASresults.txt"
                   if case in SHIPPED_RESULTS else None)
        if shipped is not None and shipped.exists():
            b = read_becas_results(shipped)
            refs["BECAS output shipped with the example"] = dict(K=b["K"], shear_centre=b["shear_centre"],
                                                                 tension_centre=b["tension_centre"])
        if case in USER_GUIDE:
            refs["BECAS user guide"] = dict(terms=USER_GUIDE[case]["terms"], centres=USER_GUIDE[case].get("centres"))
        if case.startswith("C3_") and becas_home():
            kfile = becas_home() / "examples" / case / f"{case}.vabs.K"
            Kv = read_vabs_timoshenko(kfile)
            if Kv is not None:
                inv = np.argsort(VABS_FROM_BECAS)
                refs["VABS, same mesh"] = dict(K=Kv[np.ix_(inv, inv)].tolist(),
                                               shear_centre=read_vabs_point(kfile, "Shear Center"),
                                               tension_centre=read_vabs_point(kfile, "Tension Center"))
        if case == "W3":
            alt = PUBLISHED["pipe"]["alternatives"]["PreVABS docs, YU2005"]
            refs["VABS (Yu 2005), pipe"] = dict(vabs_terms=alt)
        row["comparisons"] = {}
        A = row["asm"]
        for name, ref in refs.items():
            c: Dict = {}
            if "K" in ref:
                c.update(compare(A["K"], ref["K"]))
                for which in ("shear_centre", "tension_centre"):
                    if ref.get(which) and None not in ref[which]:
                        c[which] = dict(asm=A[which], ref=list(ref[which]),
                                        diff_over_size=centre_diff(A[which], ref[which], size))
            if "terms" in ref:
                K = np.asarray(A["K"])
                c["terms"] = {t: dict(asm=float(K[int(t[1]) - 1, int(t[2]) - 1]), ref=v,
                                      rel=float(K[int(t[1]) - 1, int(t[2]) - 1] / v - 1.0) if v else None)
                              for t, v in ref["terms"].items()}
                for which, key in (("shear_centre", "shear"), ("tension_centre", "tension")):
                    if ref.get("centres"):
                        c[which] = dict(asm=A[which], ref=list(ref["centres"][key]),
                                        diff_over_size=centre_diff(A[which], ref["centres"][key], size))
            if "vabs_terms" in ref:
                Kv = np.asarray(A["K"])[np.ix_(VABS_FROM_BECAS, VABS_FROM_BECAS)]
                c["terms"] = {t: dict(asm=float(Kv[int(t[1]) - 1, int(t[2]) - 1]), ref=v,
                                      rel=float(Kv[int(t[1]) - 1, int(t[2]) - 1] / v - 1.0))
                              for t, v in ref["vabs_terms"].items()}
            row["comparisons"][name] = c
        out[case] = row
        print(f"{case}: done", flush=True)
    if can_run:
        REFERENCE.write_text(json.dumps(stored, indent=1))
    (HERE / "results.json").write_text(json.dumps(out, indent=1))
    (HERE / "RESULTS.md").write_text(report(out, can_run))
    return out


def _fmt(v: Optional[float], spec: str = ".4g") -> str:
    return "" if v is None else format(v, spec)


def report(out: Dict, ran: bool) -> str:
    L = ["# ASM solid section against BECAS on the same meshes: results", "",
         "Generated by `run.py`. `rel` is ASM / reference - 1 in %. `max scaled` is the largest "
         "`|K_ij(ASM) - K_ij(ref)| / sqrt(K_ii K_jj)` over all 36 terms, so a coupling is judged "
         "against the diagonal terms it couples. Matrices are in BECAS's order "
         "`[Vx, Vy, N, Mx, My, Mz]`; centres in the mesh's own coordinates and their difference "
         "as a fraction of the section's size.", ""]
    if not any("BECAS, same mesh" in r["comparisons"] for r in out.values()):
        L += ["**BECAS has not been run on these meshes yet** (no MATLAB or Octave where this was "
              "generated). The comparisons below are against BECAS's published values and VABS.", ""]
    L += ["## Summary", "", "| case | mesh | elements | reference | max scaled | max term % | shear centre / size | tension centre / size |",
          "|---|---|---|---|---|---|---|---|"]
    for case, r in out.items():
        if not r["comparisons"]:
            L.append(f"| {case} | {r['source']} | {r['asm']['elements']} {'/'.join(r['asm']['kinds'])} | "
                     "BECAS run pending | | | | |")
        for name, c in r["comparisons"].items():
            terms = c.get("terms", {})
            mx = c.get("max_rel")
            if mx is None and terms:
                mx = max(abs(t["rel"]) for t in terms.values() if t["rel"] is not None)
            L.append(f"| {case} | {r['source']} | {r['asm']['elements']} {'/'.join(r['asm']['kinds'])} | {name} | "
                     f"{_fmt(c.get('max_scaled'), '.1e')} | {_fmt(None if mx is None else 100 * mx, '.3g')} | "
                     f"{_fmt(c.get('shear_centre', {}).get('diff_over_size'), '.1e')} | "
                     f"{_fmt(c.get('tension_centre', {}).get('diff_over_size'), '.1e')} |")
    for case, r in out.items():
        L += ["", f"## {case}", "", r["description"], ""]
        if case in NOTES:
            L += [f"Note: {NOTES[case]}", ""]
        A = r["asm"]
        L += [f"ASM: {A['elements']} {'/'.join(A['kinds'])} elements, {A['dof']} dof, {A['seconds']:.1f} s. "
              f"Shear centre ({A['shear_centre'][0]:.6g}, {A['shear_centre'][1]:.6g}), "
              f"tension centre ({A['tension_centre'][0]:.6g}, {A['tension_centre'][1]:.6g}).", ""]
        for name, c in r["comparisons"].items():
            L += [f"Against {name}:", "", "| term | ASM | reference | rel % |", "|---|---|---|---|"]
            for t, v in c.get("terms", {}).items():
                L.append(f"| {t} | {v['asm']:.6g} | {v['ref']:.6g} | {_fmt(None if v['rel'] is None else 100 * v['rel'], '+.4f')} |")
            for which in ("shear_centre", "tension_centre"):
                if which in c:
                    v = c[which]
                    L.append(f"| {which.replace('_', ' ')} | ({v['asm'][0]:.6g}, {v['asm'][1]:.3g}) | "
                             f"({v['ref'][0]:.6g}, {v['ref'][1]:.3g}) | {v['diff_over_size']:.1e} of size |")
            L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    run(require_becas="--becas" in sys.argv)
