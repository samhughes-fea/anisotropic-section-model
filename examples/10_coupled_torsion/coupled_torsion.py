"""Torsion of a coupled composite box: GJ, GJ_free and restrained warping.

A box whose walls are all wound the same way round (CUS, example 09) stretches when it is
twisted. That changes what "torsional stiffness" means:

- GJ, the 6 x 6 diagonal term, is the torque per unit twist rate with every other strain
  held at zero: the box is not allowed to stretch.
- GJ_free = 1 / C66, from the compliance, is the torque per unit twist rate with every
  other section force zero: the box stretches as it wants. With the coupling this is the
  softer one, and it is what a free cantilever feels.

For a [30]4 carbon box GJ_free is 0.44 GJ. ``RestrainedTorsionBeam.from_section`` takes
GJ_free by default (``other_strains="free"``) and GJ with ``other_strains="held"``.

The example sweeps the ply angle (thin-wall and solid section), then runs a shell
cantilever twisted two ways: with its tip free to stretch, and with the tip held. Each
matches its own stiffness. Last, the twist rate near the clamp, where the clamp stops the
section warping: Saint-Venant, Vlasov, Benscoter and GBT against the shell.

Figures: GJ and GJ_free against the ply angle; the cantilever's twist for the two tip
conditions; the twist rate and the stretch near the clamp.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

from _shared import composite_box as box  # noqa: E402
from _shared.checks import Checks  # noqa: E402
from _shared.plotting import REFERENCE, SERIES, TEXT_SECONDARY, plt, save  # noqa: E402
from anisotropic_section_model import (GeneralizedBeam, RestrainedTorsionBeam, ThinWalledSection,  # noqa: E402
                                       TimoshenkoBeam)
from anisotropic_section_model.solid import SolidSectionAnalysis  # noqa: E402
from anisotropic_section_model.thin_wall import BEAMDYN_FROM_ONEDFEM  # noqa: E402
from laminate_constitutive_model import create_default_library  # noqa: E402

DATA = HERE / "data" / "results.json"
B = box.SPAR
ANGLES = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
THETA = 30.0
LENGTH = 3.2
TORQUE = 100.0      # N m
N_SPAN = 64


def layup(theta):
    return box.cus(theta, B.plies)


# ----------------------------------------------------------------------------- section sweep
def sweep(lib):
    p = BEAMDYN_FROM_ONEDFEM
    out = {"theta": ANGLES, "GJ": [], "GJ_free": [], "solid_GJ": [], "solid_GJ_free": [], "EGamma": [],
           "decay_length": []}
    for th in ANGLES:
        s = box.thin(lib, layup(th), b=B)
        Ks = SolidSectionAnalysis(box.solid(lib, layup(th), b=B)).stiffness[np.ix_(p, p)]
        out["GJ"].append(s.GJ)
        out["GJ_free"].append(s.GJ_free)
        out["solid_GJ"].append(float(Ks[5, 5]))
        out["solid_GJ_free"].append(float(1.0 / np.linalg.inv(Ks)[5, 5]))
        out["EGamma"].append(s.EGamma)
        out["decay_length"].append(s.decay_length)
    return out


# ----------------------------------------------------------------------------- shell cantilever
def shell_cantilever(lib, tip):
    """Root clamped, torque on a tip rib. ``tip="free"``: the rib holds the section's shape
    only, so the tip warps and stretches freely. ``tip="held"``: the tip is welded to a
    rigid plate that cannot move along the beam (no stretch and no warping there)."""
    from shell_fem import BoundaryConditions, LoadCase, StructuralGlobalSolver, add_reference_node, rigid_link

    b = box.shell(lib, layup(THETA), LENGTH, n_width=4, n_depth=2, n_span=N_SPAN, b=B)
    mesh = b.mesh
    xyz = {n: np.asarray(mesh.node_coordinates(n)) for n in mesh.node_ids}
    root = [n for n, p in xyz.items() if abs(p[2]) < 1e-12]
    tipn = [n for n, p in xyz.items() if abs(p[2] - LENGTH) < 1e-12]
    bcs = BoundaryConditions().clamp(root)
    ref = add_reference_node(mesh, (0.0, 0.0, LENGTH))
    rigid_link(mesh, bcs, ref, tipn, dofs=(0, 1) if tip == "free" else (0, 1, 2, 3, 4, 5))
    bcs.fix([ref], [2, 3, 4])
    case = LoadCase(tip)
    case.add_nodal_load(ref, 5, TORQUE)
    sol = StructuralGlobalSolver().solve(mesh, bcs, case)

    rows = {}
    for z in b.span_positions:
        nodes = [n for n, p in xyz.items() if abs(p[2] - z) < 1e-9 and n != ref]
        P = np.array([xyz[n][:2] for n in nodes])
        d = np.array([sol.node_displacement(n)[:3] for n in nodes])
        r, ut = P - P.mean(0), d[:, :2] - d[:, :2].mean(0)
        rows[float(z)] = (float(np.sum(r[:, 0] * ut[:, 1] - r[:, 1] * ut[:, 0]) / np.sum(r * r)), float(d[:, 2].mean()))
    zs = sorted(rows)
    return {"z": zs, "twist": [rows[z][0] for z in zs], "stretch": [rows[z][1] for z in zs],
            "section": ThinWalledSection.from_blade_mesh(b, z=LENGTH)}


def cantilevers(lib):
    out = {}
    sec = None
    for tip in ("free", "held"):
        s = shell_cantilever(lib, tip)
        sec = s.pop("section")
        out[tip] = s
    z = np.linspace(0.0, LENGTH, 321)
    beams = {}
    for strains in ("free", "held"):
        rtb = RestrainedTorsionBeam.from_section(sec, LENGTH, other_strains=strains)
        for tip in ("free", "held"):
            r = rtb.cantilever_tip_torque(TORQUE, z, tip_warping="free" if tip == "free" else "restrained")
            beams[f"{strains} strains, {tip} tip"] = {"twist": r.twist.tolist(), "rate": r.twist_rate.tolist()}
        beams[f"{strains} strains, Saint-Venant"] = {"twist": rtb.saint_venant(TORQUE, z).twist.tolist(),
                                                     "rate": rtb.saint_venant(TORQUE, z).twist_rate.tolist()}
    vlasov = RestrainedTorsionBeam.from_section(sec, LENGTH, vlasov=True).cantilever_tip_torque(TORQUE, z)
    beams["Vlasov, free strains"] = {"twist": vlasov.twist.tolist(), "rate": vlasov.twist_rate.tolist()}
    # GBT: the section's warping and distortion as degrees of freedom, a diaphragm at the tip
    zn = GeneralizedBeam.graded_nodes(LENGTH, 48, [LENGTH])
    g = GeneralizedBeam(sec, LENGTH, warping="full", distortional_modes=None, z_nodes=zn).clamp(0.0)
    g = g.diaphragm(LENGTH)
    g.add_torque(LENGTH, TORQUE)
    gs = g.solve()
    beams["GBT"] = {"twist": np.asarray(gs.mean_twist(z)).tolist()}
    # the stretch of the free box: TimoshenkoBeam on the full 6 x 6
    K = sec.timoshenko_stiffness(sec.shear_centre)
    tb = TimoshenkoBeam(K, LENGTH, n_elements=8).clamp(0.0).add_force(LENGTH, [0, 0, 0, 0, 0, TORQUE]).solve()
    d = tb.displacement(z)
    beams["Timoshenko 6 x 6"] = {"twist": d[:, 5].tolist(), "stretch": d[:, 2].tolist()}
    return {"shell": out, "beam_z": z.tolist(), "beams": beams,
            "section": {"GJ": sec.GJ, "GJ_free": sec.GJ_free, "EGamma": sec.EGamma, "S": sec.S_warping,
                        "decay_length": sec.decay_length}}


# ----------------------------------------------------------------------------- checks
def check(r) -> Checks:
    c = Checks("10 coupled torsion")
    s = r["sweep"]
    k = ANGLES.index(THETA)
    c.close(f"[{THETA:g}]4 GJ_free / GJ, thin-wall", s["GJ_free"][k] / s["GJ"][k], 0.444, 0.01,
            "PR 22: 0.44 for a [30]4 box")
    c.close(f"[{THETA:g}]4 GJ_free, thin against solid", s["GJ_free"][k], s["solid_GJ_free"][k], 0.03, "solid section")
    c.close(f"[{THETA:g}]4 GJ (held), thin against solid", s["GJ"][k], s["solid_GJ"][k], 0.03, "solid section")
    for i in (0, len(ANGLES) - 1):
        c.close(f"GJ_free = GJ for orthotropic walls ({ANGLES[i]:g} deg)", s["GJ_free"][i], s["GJ"][i], 1e-6, "no coupling")
    sh, bm, z = r["cantilevers"]["shell"], r["cantilevers"]["beams"], np.array(r["cantilevers"]["beam_z"])
    sec = r["cantilevers"]["section"]

    def mid_rate(zs, tw):
        zs, tw = np.asarray(zs), np.asarray(tw)
        i1, i2 = np.argmin(abs(zs - 0.25 * LENGTH)), np.argmin(abs(zs - 0.75 * LENGTH))
        return (tw[i2] - tw[i1]) / (zs[i2] - zs[i1])

    c.close("free tip: shell twist rate = T / GJ_free", mid_rate(sh["free"]["z"], sh["free"]["twist"]),
            TORQUE / sec["GJ_free"], 0.03, "shell model")
    c.close("held tip: shell twist rate = T / GJ", mid_rate(sh["held"]["z"], sh["held"]["twist"]),
            TORQUE / sec["GJ"], 0.03, "shell model")
    c.close("free tip: Benscoter tip twist against shell", bm["free strains, free tip"]["twist"][-1],
            sh["free"]["twist"][-1], 0.02, "shell model")
    c.close("held tip: Benscoter (held strains) tip twist against shell", bm["held strains, held tip"]["twist"][-1],
            sh["held"]["twist"][-1], 0.03, "shell model")
    c.close("free tip: stretch under torque, TimoshenkoBeam against shell", bm["Timoshenko 6 x 6"]["stretch"][-1],
            sh["free"]["stretch"][-1], 0.03, "shell model")
    c.close("GBT tip twist against shell", bm["GBT"]["twist"][-1], sh["free"]["twist"][-1], 0.03, "shell model")
    c.true("the clamp stiffens: Benscoter twist at 2 decay lengths below Saint-Venant",
           np.interp(2 * sec["decay_length"], z, bm["free strains, free tip"]["twist"])
           < np.interp(2 * sec["decay_length"], z, bm["free strains, Saint-Venant"]["twist"]), "warping restraint")
    return c


# ----------------------------------------------------------------------------- figures
def figure_sweep(r):
    s = r["sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax = axes[0]
    ax.plot(s["theta"], np.array(s["GJ"]) / 1e6, color=SERIES[0], label="GJ (other strains held)")
    ax.plot(s["theta"], np.array(s["GJ_free"]) / 1e6, color=SERIES[1], label="GJ_free (other forces zero)")
    ax.plot(s["theta"], np.array(s["solid_GJ"]) / 1e6, "o", color=REFERENCE, mfc="none", ms=5, label="solid section")
    ax.plot(s["theta"], np.array(s["solid_GJ_free"]) / 1e6, "s", color=REFERENCE, mfc="none", ms=5)
    ax.set_xlabel("ply angle θ [deg]")
    ax.set_ylabel("torsional stiffness [MN m²]")
    ax.set_title("Letting the box stretch removes up to 57 % of GJ", loc="left")
    ax.set_xticks(ANGLES)
    ax.legend()
    ax = axes[1]
    ax.plot(s["theta"], np.array(s["GJ_free"]) / np.array(s["GJ"]), color=SERIES[1], label="thin-wall")
    ax.plot(s["theta"], np.array(s["solid_GJ_free"]) / np.array(s["solid_GJ"]), "s", color=REFERENCE, mfc="none",
            ms=5, label="solid section")
    ax.set_xlabel("ply angle θ [deg]")
    ax.set_ylabel("GJ_free / GJ")
    ax.set_title("GJ_free / GJ: lowest between 20 and 30 degrees", loc="left")
    ax.set_xticks(ANGLES)
    ax.legend()
    fig.text(0.0, -0.04, f"CUS box {B.width - B.wall:g} x {B.depth - B.wall:g} m on the mid-line, walls [θ]{B.plies} "
             f"of {B.ply * 1e3:g} mm {B.material} plies in every wall.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "gj_free.png")


def figure_cantilever(r):
    c = r["cantilevers"]
    sh, bm, z = c["shell"], c["beams"], np.array(c["beam_z"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1))
    ax = axes[0]
    ax.plot(sh["free"]["z"], 1e3 * np.array(sh["free"]["twist"]), "o", color=REFERENCE, mfc="none", ms=4,
            label="shell, tip free to stretch")
    ax.plot(sh["held"]["z"], 1e3 * np.array(sh["held"]["twist"]), "s", color=REFERENCE, mfc="none", ms=4,
            label="shell, tip held")
    ax.plot(z, 1e3 * np.array(bm["free strains, free tip"]["twist"]), color=SERIES[1], label="Benscoter, GJ_free")
    ax.plot(z, 1e3 * np.array(bm["held strains, held tip"]["twist"]), color=SERIES[0], label="Benscoter, GJ")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("twist [mrad]")
    ax.set_title("Each end condition has its own stiffness", loc="left")
    ax.legend(fontsize=7.5)
    ax = axes[1]
    zz = np.array(sh["free"]["z"])
    rate = np.diff(sh["free"]["twist"]) / np.diff(zz)
    zm = 0.5 * (zz[1:] + zz[:-1])
    norm = TORQUE / c["section"]["GJ_free"]
    ax.plot(zm, rate / norm, "o", color=REFERENCE, mfc="none", ms=4, label="shell")
    ax.plot(z, np.array(bm["free strains, Saint-Venant"]["rate"]) / norm, color=TEXT_SECONDARY, ls="--", lw=1.2,
            label="Saint-Venant (free warping)")
    ax.plot(z, np.array(bm["Vlasov, free strains"]["rate"]) / norm, color=SERIES[2], label="Vlasov")
    ax.plot(z, np.array(bm["free strains, free tip"]["rate"]) / norm, color=SERIES[1], label="Benscoter")
    g = np.array(bm["GBT"]["twist"])
    ax.plot(0.5 * (z[1:] + z[:-1]), np.diff(g) / np.diff(z) / norm, color=SERIES[3], ls="-.", label="GBT")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("z from the clamp [m]")
    ax.set_ylabel("twist rate / (T / GJ_free)")
    ax.set_title("Near the clamp: warping held, GBT follows the shell", loc="left")
    ax.legend(fontsize=7.5)
    ax = axes[2]
    ax.plot(sh["free"]["z"], 1e6 * np.array(sh["free"]["stretch"]), "o", color=REFERENCE, mfc="none", ms=4,
            label="shell, mean axial displacement")
    ax.plot(z, 1e6 * np.array(bm["Timoshenko 6 x 6"]["stretch"]), color=SERIES[0], label="TimoshenkoBeam, 6 x 6")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("axial displacement [µm]")
    ax.set_title("Twist it and it stretches", loc="left")
    ax.legend(fontsize=7.5)
    fig.text(0.0, -0.04, f"[{THETA:g}]{B.plies} CUS box, {LENGTH:g} m cantilever, {TORQUE:g} N m at the tip. Shell: "
             f"Quad8 FSDT, 4 + 2 elements per cap and web, {N_SPAN} along. 'Tip held': the tip welded to a rigid "
             "plate that cannot move along the beam.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "cus_cantilever.png")


def report(r):
    s = r["sweep"]
    print("theta  GJ [MN m2]  GJ_free  ratio | solid GJ  GJ_free")
    for i, th in enumerate(s["theta"]):
        print(f"{th:5g}  {s['GJ'][i] / 1e6:9.4f}  {s['GJ_free'][i] / 1e6:7.4f}  {s['GJ_free'][i] / s['GJ'][i]:.3f} | "
              f"{s['solid_GJ'][i] / 1e6:8.4f}  {s['solid_GJ_free'][i] / 1e6:7.4f}")
    c = r["cantilevers"]
    sh, bm = c["shell"], c["beams"]
    for tip in ("free", "held"):
        print(f"tip {tip}: shell tip twist {1e3 * sh[tip]['twist'][-1]:.4f} mrad; Benscoter with GJ_free "
              f"{1e3 * bm[f'free strains, {tip} tip']['twist'][-1]:.4f}, with GJ {1e3 * bm[f'held strains, {tip} tip']['twist'][-1]:.4f}")
    print(f"GBT tip twist {1e3 * bm['GBT']['twist'][-1]:.4f} mrad; stretch shell {1e6 * sh['free']['stretch'][-1]:.3f} um, "
          f"beam {1e6 * bm['Timoshenko 6 x 6']['stretch'][-1]:.3f} um")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", action="store_true", help="redraw from data/results.json")
    args = parser.parse_args()
    if args.reuse and DATA.exists():
        r = json.loads(DATA.read_text())
    else:
        lib = create_default_library()
        r = {"sweep": sweep(lib), "cantilevers": cantilevers(lib)}
        DATA.parent.mkdir(exist_ok=True)
        DATA.write_text(json.dumps(r, indent=1))
    report(r)
    figure_sweep(r)
    figure_cantilever(r)
    check(r).report()


if __name__ == "__main__":
    main()
