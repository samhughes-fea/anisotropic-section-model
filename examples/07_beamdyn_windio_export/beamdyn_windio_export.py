"""Blade beam properties for OpenFAST BeamDyn and windIO: a spanwise sweep, checked.

``BeamDynBlade.from_geometry`` cuts a ``BladeGeometry`` at each station, builds the
``ThinWalledSection`` of every cut and writes a complete BeamDyn deck (primary and blade
files) and the windIO 2.0 blade fields. The example checks it three ways:

1. **Against the IEA 15 MW reference files.** The turbine's windIO file, converted by
   ``BeamDynBlade.from_windio``, is written as a BeamDyn blade file and compared with
   the official ``IEA-15-240-RWT_BeamDyn_blade.dat`` entry by entry; the official deck
   is read and rewritten for OpenFAST 4 and 5 and read back. The two files are
   downloaded (public, Apache-2.0) from the IEA-15-240-RWT repository at a fixed commit
   into ``data/reference/`` (git-ignored); without a network that part is skipped.
2. **In BeamDyn itself**, when ``BEAMDYN_DRIVER`` points at OpenFAST's
   ``beamdyn_driver``: the official and rewritten IEA 15 MW decks under the same tip
   load, and a twisted, coupled, offset prismatic blade swept from the geometry against
   ``TimoshenkoBeam`` on its section, under all six tip loads and self-weight.
3. **A tapered, twisted, prebent blade** with 15° carbon caps, swept at 11 stations;
   its deck and windIO file are written to ``data/`` and drawn along the span.

Figure: the tapered blade's stiffness, couplings and mass along the span.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from _shared.plotting import SERIES, TEXT_SECONDARY, plt, save
from anisotropic_section_model import BeamDynBlade, TimoshenkoBeam
from anisotropic_section_model.beamdyn import MIRROR
from implicit_geometry_kernel import Airfoil, BladeGeometry, Station
from shell_fem import Laminate, create_default_library

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = DATA / "results.json"
REFERENCE = DATA / "reference"
IEA_COMMIT = "28ae083f6e4d2e2e4ea35ac9100fcab104940484"
IEA_URL = f"https://raw.githubusercontent.com/IEAWindTask37/IEA-15-240-RWT/{IEA_COMMIT}/"
IEA_FILES = {"IEA-15-240-RWT.yaml": "WT_Ontology/IEA-15-240-RWT.yaml",
             "IEA-15-240-RWT_BeamDyn.dat": "OpenFAST/IEA-15-240-RWT/IEA-15-240-RWT_BeamDyn.dat",
             "IEA-15-240-RWT_BeamDyn_blade.dat": "OpenFAST/IEA-15-240-RWT/IEA-15-240-RWT_BeamDyn_blade.dat"}
TIP = ("TipTDxr", "TipTDyr", "TipTDzr", "TipRDxr", "TipRDyr", "TipRDzr")
LENGTH = 20.0


def laminates(lib, cap_angle=0.0, cap_plies=16):
    glass, carbon = lib.get("E-glass/epoxy"), lib.get("T300/5208")
    return {"skin": Laminate.from_angles(glass, [0, 45, -45, -45, 45, 0], 0.5e-3),
            "web": Laminate.from_angles(glass, [45, -45, -45, 45], 0.5e-3),
            "cap": Laminate.from_angles(carbon, [cap_angle] * cap_plies, 0.5e-3)}


# ---------------------------------------------------------------- BeamDyn driver
def driver_version(driver):
    banner = subprocess.run([driver, "-v"], capture_output=True, text=True).stdout
    return int(re.search(r"OpenFAST - v(\d+)", banner).group(1))


def run_beamdyn(driver, primary, tip_load=(0.0,) * 6, gravity=(0.0,) * 3):
    """Static BeamDyn solve of ``primary`` (root clamped); tip displacement and rotation."""
    primary = Path(primary).resolve()
    lines = ["--- BeamDyn driver", "static", "--- SIMULATION CONTROL", "False DynamicSolve", "0 t_initial",
             "0 t_final", "0.002 dt", "--- GRAVITY", *[f"{g:.12e} G" for g in gravity], "--- FRAME", "0 GlbPos1",
             "0 GlbPos2", "0 GlbPos3", "--- DCM", "--- DCM", "1 0 0", "0 1 0", "0 0 1", "T GlbRotBladeT0",
             "--- ROOT VELOCITY", "0 RootVel4", "0 RootVel5", "0 RootVel6", "--- APPLIED FORCE", *["0 DistrLoad"] * 6,
             *[f"{v:.12e} TipLoad" for v in tip_load], "0 NumPointLoads", "eta F M", "(-) (N) (N-m)",
             "--- PRIMARY INPUT FILE", f"\"{primary.name}\" InputFile", "--- Outputs", "0 WrVTK", "15 VTK_fps"]
    drv = primary.with_name(primary.stem + "_driver.inp")
    drv.write_text("\n".join(lines) + "\n")
    subprocess.run([driver, drv.name], cwd=primary.parent, check=True, capture_output=True)
    out = drv.with_suffix(".out").read_text().splitlines()
    names = next(line.split() for line in out if line.split()[:1] == ["Time"])
    values = dict(zip(names, map(float, out[-1].split())))
    return [values[n] for n in TIP]


# ---------------------------------------------------------------- 1. IEA 15 MW
def fetch_reference():
    REFERENCE.mkdir(parents=True, exist_ok=True)
    for name, rel in IEA_FILES.items():
        path = REFERENCE / name
        if not path.exists():
            with urllib.request.urlopen(IEA_URL + rel, timeout=60) as r:
                path.write_bytes(r.read())
    return REFERENCE


def iea15(driver, version, work):
    try:
        ref = fetch_reference()
    except OSError as exc:
        print(f"IEA 15 MW check skipped (no reference files: {exc})")
        return None
    official = BeamDynBlade.read(ref / "IEA-15-240-RWT_BeamDyn.dat")
    windio = BeamDynBlade.from_windio(ref / "IEA-15-240-RWT.yaml", n_key_points=len(official.key_points))
    out = {"stations": len(official.eta), "key_points": len(official.key_points),
           "K_max_rel_diff": float(np.max(np.abs(windio.stiffness - official.stiffness)) /
                                   np.max(np.abs(official.stiffness))),
           "M_max_rel_diff": float(np.max(np.abs(windio.mass - official.mass)) / np.max(np.abs(official.mass))),
           "eta_max_diff": float(np.max(np.abs(windio.eta - official.eta))),
           "kp_max_diff_m": float(np.max(np.abs(windio.key_points - official.key_points))),
           "twist_max_diff_deg": float(np.max(np.abs(windio.twist - official.twist)))}
    exact = True
    for v in (4, 5):
        back = BeamDynBlade.read(official.write(work / f"iea15_v{v}.dat", damping=official.damping, version=v))
        exact &= all(np.array_equal(getattr(back, a), getattr(official, a))
                     for a in ("eta", "stiffness", "mass", "key_points", "twist"))
    out["rewrite_exact"] = bool(exact)
    print(f"IEA 15 MW: windIO -> BeamDyn blade file against the official one: K {out['K_max_rel_diff']:.1e}, "
          f"M {out['M_max_rel_diff']:.1e} (largest difference over the largest entry); rewritten deck read back "
          f"{'exactly' if exact else 'NOT exactly'}")
    if driver:
        load = (1e5, 0, 0, 0, 0, 0)  # 100 kN flapwise at the tip
        options = dict(damping=official.damping, version=version, quadrature=2, refine=2, order_elem=10)
        if version >= 5:  # the official deck is in OpenFAST 4's format
            official.write(work / "iea15_official.dat", **options)
            primary = work / "iea15_official.dat"
        else:
            primary = ref / "IEA-15-240-RWT_BeamDyn.dat"
        out["beamdyn_tip_official"] = run_beamdyn(driver, primary, load)
        out["beamdyn_tip_from_windio"] = run_beamdyn(driver, windio.write(work / "iea15_windio.dat", **options), load)
        print(f"  BeamDyn, 100 kN flapwise tip load: tip deflection {out['beamdyn_tip_official'][0]:.4f} m "
              f"(official deck), {out['beamdyn_tip_from_windio'][0]:.4f} m (from windIO)")
    return out


# ---------------------------------------------------------------- 2. BeamDyn vs the section beam
def prismatic_check(lib, driver, version, work):
    stations = [Station(z, 1.0, Airfoil.naca4("0024"), twist=12.0, pitch_axis=0.3, offset=(0.05, -0.03))
                for z in (0.0, LENGTH)]
    geo = BladeGeometry(stations, "2D-CN-LC-0R", 0.003, pivot_fraction=0.3, spar_cap_thickness=0.008,
                        shear_web_thickness=0.002, web_positions=(0.18, 0.48))
    blade = BeamDynBlade.from_geometry(geo, laminates(lib, cap_angle=20.0), z=np.linspace(0.0, LENGTH, 5))
    primary = blade.write(work / "prismatic.dat", quadrature=1, version=version)
    cut = geo.section_at(LENGTH / 2)
    sec = blade.sections[2]
    K, M = sec.timoshenko_stiffness(cut.offset), sec.mass_matrix(cut.offset)
    rows = []
    for k, (name, scale) in enumerate(zip(("Fx", "Fy", "Fz", "Mx", "My", "Mz"), (1.0, 1.0, 10.0, 1.0, 1.0, 1.0))):
        f = np.zeros(6)
        f[k] = scale
        asm = MIRROR @ TimoshenkoBeam(K, LENGTH, 4).clamp(0.0).add_force(LENGTH, f).solve().node_displacement(LENGTH)
        bd = np.array(run_beamdyn(driver, primary, tip_load=MIRROR @ f))
        rows.append({"load": f"tip {name} = {scale:g}", "asm": asm.tolist(), "beamdyn": bd.tolist(),
                     "max_diff": float(np.max(np.abs(bd - asm)) / np.max(np.abs(asm)))})
    for k, g in enumerate(np.eye(3) * 1e-3):
        q = M @ np.concatenate([MIRROR[:3, :3] @ g, np.zeros(3)])
        asm = MIRROR @ TimoshenkoBeam(K, LENGTH, 4).clamp(0.0).add_distributed(q).solve().node_displacement(LENGTH)
        bd = np.array(run_beamdyn(driver, primary, gravity=g))
        rows.append({"load": f"gravity {'XYZ'[k]} = 1e-3", "asm": asm.tolist(), "beamdyn": bd.tolist(),
                     "max_diff": float(np.max(np.abs(bd - asm)) / np.max(np.abs(asm)))})
    print("BeamDyn against TimoshenkoBeam on the section (twisted 12°, offset, 20° caps): largest tip difference "
          f"{max(r['max_diff'] for r in rows):.1e} of the largest tip motion")
    return rows


# ---------------------------------------------------------------- 3. a tapered blade
def tapered_blade(lib):
    af_root, af_tip = Airfoil.naca4("0030"), Airfoil.naca4("0018")
    stations = [Station(0.0, 2.0, af_root, twist=10.0, pitch_axis=0.3),
                Station(LENGTH, 1.0, af_tip, twist=0.0, pitch_axis=0.3, offset=(0.0, -0.4))]
    cap_t = lambda z: 0.012 - 0.00025 * z  # noqa: E731 - 12 mm to 7 mm
    geo = BladeGeometry(stations, "2D-CN-LC-0R", 0.003, pivot_fraction=0.3, spar_cap_thickness=cap_t,
                        shear_web_thickness=0.002, web_positions=(0.18, 0.48),
                        offset=([0.0, 10.0, LENGTH], [[0.0, 0.0], [0.0, -0.1], [0.0, -0.4]]))

    def lams(z):
        return laminates(lib, cap_angle=15.0, cap_plies=int(round(cap_t(z) / 0.5e-3)))

    z = np.linspace(0.0, LENGTH, 11)
    return BeamDynBlade.from_geometry(geo, lams, z=z, name="Tapered 20 m blade, 2D-CN layout"), z


def solve():
    driver = os.environ.get("BEAMDYN_DRIVER")
    version = driver_version(driver) if driver else 5
    tmp = tempfile.TemporaryDirectory()
    work = Path(tmp.name)
    lib = create_default_library()
    out = {"openfast_version": version if driver else None, "iea15": iea15(driver, version, work)}
    if driver:
        out["prismatic"] = prismatic_check(lib, driver, version, work)
    blade, z = tapered_blade(lib)
    blade.write(DATA / "tapered_BeamDyn.dat", version=5)
    blade.write_windio(DATA / "tapered_windio.yaml")
    print("wrote data/tapered_BeamDyn.dat, data/tapered_BeamDyn_blade.dat (OpenFAST 5) and data/tapered_windio.yaml")
    if driver:
        p = blade.write(work / "tapered.dat", version=version)
        out["tapered_beamdyn_tip_1kN_flap"] = run_beamdyn(driver, p, (1e3, 0, 0, 0, 0, 0))
    v = blade.station_values()
    K = blade.stiffness
    out["tapered"] = {"z": z.tolist(), "eta": blade.eta.tolist(), "twist": blade.twist.tolist(),
                      "prebend": blade.key_points[:, 0].tolist(),
                      **{k: v[k].tolist() for k in ("mass", "EA", "EI_flap", "EI_edge", "GJ", "cm_y", "i_plr")},
                      "c_flap_twist": (K[:, 4, 5] / np.sqrt(K[:, 4, 4] * K[:, 5, 5])).tolist(),
                      "c_edge_twist": (K[:, 3, 5] / np.sqrt(K[:, 3, 3] * K[:, 5, 5])).tolist(),
                      "c_ext_twist": (K[:, 2, 5] / np.sqrt(K[:, 2, 2] * K[:, 5, 5])).tolist()}
    tmp.cleanup()
    return out


def plot(res):
    t = res["tapered"]
    z = np.array(t["z"])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    ax = axes[0]
    for name, label, c in (("EA", "EA (N)", SERIES[0]), ("EI_edge", "EI edge, K44 (N m²)", SERIES[1]),
                           ("EI_flap", "EI flap, K55 (N m²)", SERIES[2]), ("GJ", "GJ, K66 (N m²)", SERIES[3])):
        ax.semilogy(z, t[name], "o-", color=c, ms=4, label=label)
    ax.set(xlabel="span z (m)", title="Stiffness along the span")
    ax.legend()
    ax = axes[1]
    for name, label, c in (("c_ext_twist", "extension-twist K36", SERIES[0]),
                           ("c_edge_twist", "edge-twist K46", SERIES[1]),
                           ("c_flap_twist", "flap-twist K56", SERIES[2])):
        ax.plot(z, t[name], "o-", color=c, ms=4, label=label)
    ax.set(xlabel="span z (m)", ylabel="K_ij / √(K_ii K_jj)", title="Couplings about the pitch axis")
    ax.legend()
    ax = axes[2]
    ax.plot(z, t["mass"], "o-", color=SERIES[0], ms=4, label="mass (kg/m)")
    ax.set(xlabel="span z (m)", ylabel="mass per length (kg/m)", title="Mass and mass centre")
    ax2 = ax.twinx()
    ax2.plot(z, np.array(t["cm_y"]) * 1e3, "s--", color=SERIES[1], ms=4, label="mass centre aft of pitch axis")
    ax2.set_ylabel("mass centre aft of the pitch axis (mm)", color=TEXT_SECONDARY)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2)
    fig.tight_layout()
    save(fig, __file__, "tapered_blade_properties.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", action="store_true", help="redraw from data/results.json")
    args = parser.parse_args()
    if args.reuse and RESULTS.exists():
        res = json.loads(RESULTS.read_text())
    else:
        res = solve()
        RESULTS.write_text(json.dumps(res, indent=1))
    plot(res)


if __name__ == "__main__":
    main()
