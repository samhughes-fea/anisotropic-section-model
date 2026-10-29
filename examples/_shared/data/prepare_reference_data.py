"""Build the IEA 15 MW reference data (composite-laminate-model's example 24) from the published IEA 15 MW blade definition.

This script is run once; its output, ``iea15mw_sections.json`` next to it, is committed so the
example itself needs neither WISDEM nor a network connection. To regenerate it::

    pip install wisdem pyyaml
    python examples/_shared/data/prepare_reference_data.py

It does three things:

1. Downloads ``IEA-15-240-RWT.yaml`` (the windIO definition of the IEA Wind 15 MW
   reference turbine) from the IEAWindTask37/IEA-15-240-RWT repository.
2. Runs WISDEM's blade model on it, which turns the windIO layup (layers placed by
   non-dimensional arc length, webs, anchors) into PreComp sections: airfoil shape,
   sector boundaries on each surface, laminae per sector, web positions. The inputs
   PreComp receives are captured at the selected stations.
3. Re-runs PreComp (WISDEM's Python port of NREL's code) on each captured section with
   zero twist and zero twist rate, so its properties are in the chord frame and
   describe a prismatic segment, like the shell model in the example. The published
   6x6 stiffness and mass data in the same file are interpolated to each station.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/IEAWindTask37/IEA-15-240-RWT/master/WT_Ontology/IEA-15-240-RWT.yaml"
HERE = Path(__file__).resolve().parent
OUT = HERE / "iea15mw_sections.json"
# Span fractions to compare at: root tube, max chord (flatback), mid-span, outboard.
TARGETS = [0.0, 0.207, 0.345, 0.517, 0.69, 0.862]

MODELING = """
General:
    verbosity: False
WISDEM:
    RotorSE:
        flag: True
        spar_cap_ss: Spar_Cap_SS
        spar_cap_ps: Spar_Cap_PS
        te_ss: TE_reinforcement_SS
        te_ps: TE_reinforcement_PS
    DriveSE: {flag: False}
    TowerSE: {flag: False}
    FixedBottomSE: {flag: False}
    BOS: {flag: False}
    LCOE: {flag: True}
"""
ANALYSIS = """
general:
    folder_output: outputs
    fname_output: refturb_output
"""


def capture_precomp_inputs(yaml_path: Path):
    """Run WISDEM and keep the PreComp object it builds for the blade."""
    import wisdem.rotorse.rotor_elasticity as elasticity
    from wisdem import run_wisdem
    from wisdem.precomp import PreComp

    captured = []

    class Capture(PreComp):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.append(self)

    elasticity.PreComp = Capture
    with tempfile.TemporaryDirectory() as tmp:
        mo, ao = Path(tmp) / "modeling.yaml", Path(tmp) / "analysis.yaml"
        mo.write_text(MODELING)
        ao.write_text(ANALYSIS.replace("outputs", str(Path(tmp) / "outputs")))
        run_wisdem(str(yaml_path), str(mo), str(ao))
    return captured[-1]


def rerun_precomp(pc, i):
    """PreComp properties of section ``i`` with zero twist and twist rate (chord frame)."""
    from wisdem.precomp import properties

    xnode, ynode = pc.profile[i]._preCompFormat()
    U, L, W = (cs[i]._preCompFormat() for cs in (pc.upperCS, pc.lowerCS, pc.websCS))
    nweb = len(W[0])
    if nweb == 0:
        W = ([0], [0], [0], [0], [0], [0])
    m = pc.materials
    out = properties.properties(
        pc.chord[i], 0.0, 0.0, pc.leLoc[i], xnode, ynode,
        [x.E1 for x in m], [x.E2 for x in m], [x.G12 for x in m], [x.nu12 for x in m], [x.rho for x in m],
        *U, *L, nweb, *W)
    (eif, eil, gj, ea, eifl, sf, sl, sft, slt, sat, x_sc, y_sc, x_tc, y_tc, mass, area, iflap, ilag, tw,
     x_cg, y_cg) = out
    return {"EA": ea, "EI_flap": eif, "EI_edge": eil, "EI_flap_edge": eifl, "GJ": gj,
            "EA_flap": sf, "EA_edge": sl, "flap_GJ": sft, "edge_GJ": slt, "EA_GJ": sat,
            "x_sc": x_sc, "y_sc": y_sc, "x_tc": x_tc, "y_tc": y_tc, "x_cg": x_cg, "y_cg": y_cg,
            "mass": mass, "area": area}


def section(cs, i, names):
    c = cs[i]
    laminae = []
    for n, t, th, mi in zip(c.n_plies, c.t, c.theta, c.mat_idx):
        laminae.append([[names[int(k)], float(nn * tt), float(a)] for nn, tt, a, k in zip(n, t, th, mi)])
    return {"loc": [float(v) for v in c.loc], "laminae": laminae}


def published(blade_yaml, s):
    """The published 6x6 stiffness and mass data in the yaml, interpolated to span fraction s."""
    ep = blade_yaml["structure"]["elastic_properties"]
    K = ep["stiffness_matrix"]
    g = np.asarray(K["grid"])
    k6 = np.zeros((6, 6))
    for a in range(6):
        for b in range(a, 6):
            k6[a, b] = k6[b, a] = np.interp(s, g, K[f"K{a + 1}{b + 1}"])
    mi = ep["inertia_matrix"]
    return {"K": k6.tolist(), "mass": float(np.interp(s, mi["grid"], mi["mass"]))}


def main():
    import yaml

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "IEA-15-240-RWT.yaml"
        urllib.request.urlretrieve(URL, path)
        raw = path.read_bytes()
        turbine = yaml.safe_load(raw)
        pc = capture_precomp_inputs(path)
    r = np.asarray(pc.r)
    s_all = (r - r[0]) / (r[-1] - r[0])
    names = [m.name for m in pc.materials]
    used = set()
    stations = []
    for target in TARGETS:
        i = int(np.argmin(np.abs(s_all - target)))
        prof = pc.profile[i]
        entry = {
            "index": i, "span_fraction": float(s_all[i]), "r": float(r[i]),
            "chord": float(pc.chord[i]), "twist_deg": float(pc.theta[i]), "pitch_axis": float(pc.leLoc[i]),
            "profile": {"x": prof.x.tolist(), "y_upper": prof.yu.tolist(), "y_lower": prof.yl.tolist()},
            "upper": section(pc.upperCS, i, names), "lower": section(pc.lowerCS, i, names),
            "webs": section(pc.websCS, i, names),
            "precomp": rerun_precomp(pc, i),
            "published": published(turbine["components"]["blade"], float(s_all[i])),
        }
        for part in ("upper", "lower", "webs"):
            used.update(m for lam in entry[part]["laminae"] for m, _, _ in lam)
        stations.append(entry)
        print(f"station {i}: s = {s_all[i]:.3f}, chord {pc.chord[i]:.2f} m")
    materials = {m.name: {"E1": m.E1, "E2": m.E2, "G12": m.G12, "nu12": m.nu12, "rho": m.rho}
                 for m in pc.materials if m.name in used}
    data = {
        "source": URL, "source_sha256": hashlib.sha256(raw).hexdigest(),
        "turbine": turbine["name"],
        "notes": "PreComp inputs captured from WISDEM; PreComp re-run with zero twist (chord frame). "
                 "PreComp axes: x flapwise, y chordwise, about the pitch axis. "
                 "'published' is the stiffness_matrix / inertia_matrix block of the yaml.",
        "materials": materials, "stations": stations,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
