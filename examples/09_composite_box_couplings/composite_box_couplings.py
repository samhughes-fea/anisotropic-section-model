"""Coupled composite box beams: extension-twist (CUS) and bending-twist (CAS).

Wind the plies of a box beam at an angle and it no longer stretches, bends and twists
independently. Two layups give the two classic couplings:

- CUS (circumferentially uniform stiffness): every wall at +theta in its own axes. The
  fibres run round the box as one helix, so pulling the box twists it.
- CAS (circumferentially asymmetric stiffness): the top and bottom walls at the same
  fibre direction seen from outside, so the box bends and twists together, which is
  how a wind turbine blade is tailored to twist as it bends.

The section stiffness is computed by thin-wall theory (``ThinWalledSection``) and by the
solid section (``SolidSectionAnalysis``, every ply meshed), for ply angles from 0 to 90
degrees. Then a shell model of a 0.76 m cantilever checks what the couplings do: the CUS
box under tension and the CAS box under a flapwise tip force, against ``TimoshenkoBeam``
on each 6 x 6.

Figures: the couplings against the ply angle; the shell cantilevers' twist along the
span against the beam.
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
from anisotropic_section_model import ThinWalledSection, TimoshenkoBeam  # noqa: E402
from anisotropic_section_model.solid import SolidSectionAnalysis  # noqa: E402
from anisotropic_section_model.thin_wall import BEAMDYN_FROM_ONEDFEM  # noqa: E402
from laminate_constitutive_model import create_default_library  # noqa: E402

DATA = HERE / "data" / "results.json"
ANGLES = [0.0, 7.5, 15.0, 22.5, 30.0, 37.5, 45.0, 52.5, 60.0, 67.5, 75.0, 82.5, 90.0]
THETA = 15.0                 # the layup of the cantilevers (Chandra et al. tested 15 degrees)
LENGTH = 0.762               # m (30 in)
AXIAL = 2.0e3                # N, CUS tip tension
FLAP = 20.0                  # N, CAS tip force along y (the box depth: flapwise)
TERMS = ["Fx", "Fy", "N", "Mx", "My", "T"]
#: the coupling each layup is built for: (row, column) in [gamma_x, gamma_y, eps, kappa_x, kappa_y, kappa_z]
COUPLING = {"CUS": (2, 5), "CAS": (3, 5)}
LAYUPS = {"CUS": box.cus, "CAS": box.cas}


def normalised(K, i, j):
    return float(K[i, j] / np.sqrt(K[i, i] * K[j, j]))


def solid_k6(lib, layup):
    """The solid section's 6 x 6 about the box centre, in the top-level (BeamDyn) order."""
    p = BEAMDYN_FROM_ONEDFEM
    return SolidSectionAnalysis(box.solid(lib, layup)).stiffness[np.ix_(p, p)]


def sweep(lib):
    out = {}
    for name, layup in LAYUPS.items():
        i, j = COUPLING[name]
        rows = {"theta": ANGLES, "thin": [], "solid": [], "thin_diag": [], "solid_diag": []}
        for th in ANGLES:
            Kt = box.thin(lib, layup(th)).timoshenko_stiffness((0.0, 0.0))
            Ks = solid_k6(lib, layup(th))
            rows["thin"].append(normalised(Kt, i, j))
            rows["solid"].append(normalised(Ks, i, j))
            rows["thin_diag"].append(np.diag(Kt).tolist())
            rows["solid_diag"].append(np.diag(Ks).tolist())
        out[name] = rows
    return out


def matrices(lib):
    """Both 6 x 6 at THETA and at -THETA (the coupling must change sign)."""
    out = {}
    for name, layup in LAYUPS.items():
        out[name] = {"thin": box.thin(lib, layup(THETA)).timoshenko_stiffness((0.0, 0.0)).tolist(),
                     "solid": solid_k6(lib, layup(THETA)).tolist(),
                     "thin_flipped": box.thin(lib, layup(-THETA)).timoshenko_stiffness((0.0, 0.0)).tolist(),
                     "solid_refined": solid_k6_refined(lib, layup(THETA)).tolist()}
    return out


def solid_k6_refined(lib, layup):
    p = BEAMDYN_FROM_ONEDFEM
    return SolidSectionAnalysis(box.solid(lib, layup, refine=2)).stiffness[np.ix_(p, p)]


# ----------------------------------------------------------------------------- shell cantilevers
def shell_twist_line(lib, name):
    from shell_fem import BoundaryConditions, LoadCase, StructuralGlobalSolver, add_reference_node, rigid_link

    layup = LAYUPS[name](THETA)
    b = box.shell(lib, layup, LENGTH)
    mesh = b.mesh
    sec = ThinWalledSection.from_blade_mesh(b, z=LENGTH)
    xyz = {n: np.asarray(mesh.node_coordinates(n)) for n in mesh.node_ids}
    root = [n for n, p in xyz.items() if abs(p[2]) < 1e-12]
    tip = [n for n, p in xyz.items() if abs(p[2] - LENGTH) < 1e-12]
    bcs = BoundaryConditions().clamp(root)
    ref = add_reference_node(mesh, (0.0, 0.0, LENGTH))
    rigid_link(mesh, bcs, ref, tip, dofs=(0, 1))   # a rib: rigid in its plane, free to warp and stretch
    bcs.fix([ref], [2, 3, 4])
    case = LoadCase(name)
    if name == "CUS":
        perimeter = 2 * (box.WIDTH + box.DEPTH - 2 * box.WALL)
        case.add_line_load(tip, (0.0, 0.0, AXIAL / perimeter))
        force = [0, 0, AXIAL, 0, 0, 0]
    else:
        case.add_nodal_load(ref, 1, FLAP)
        force = [0, FLAP, 0, 0, 0, 0]
    sol = StructuralGlobalSolver().solve(mesh, bcs, case)

    def twist(z):
        nodes = [n for n, p in xyz.items() if abs(p[2] - z) < 1e-9 and n != ref]
        P = np.array([xyz[n][:2] for n in nodes])
        d = np.array([sol.node_displacement(n)[:2] for n in nodes])
        r, ut = P - P.mean(0), d - d.mean(0)
        return float(np.sum(r[:, 0] * ut[:, 1] - r[:, 1] * ut[:, 0]) / np.sum(r * r))

    zs = [float(z) for z in b.span_positions]
    return {"z": zs, "twist": [twist(z) for z in zs], "force": force,
            "section_K": sec.timoshenko_stiffness((0.0, 0.0)).tolist()}


def beam_twist(K, force, z):
    beam = TimoshenkoBeam(np.asarray(K), LENGTH, n_elements=8).clamp(0.0).add_force(LENGTH, force)
    return beam.solve().displacement(z)[:, 5].tolist()


def cantilevers(lib, mats):
    out = {}
    for name in LAYUPS:
        s = shell_twist_line(lib, name)
        z = np.linspace(0.0, LENGTH, 61)
        s["beam_z"] = z.tolist()
        s["beam_thin"] = beam_twist(mats[name]["thin"], s["force"], z)
        s["beam_solid"] = beam_twist(mats[name]["solid"], s["force"], z)
        s["beam_tip_thin"] = beam_twist(mats[name]["thin"], s["force"], [LENGTH])[0]
        s["beam_tip_solid"] = beam_twist(mats[name]["solid"], s["force"], [LENGTH])[0]
        out[name] = s
    return out


# ----------------------------------------------------------------------------- checks
def check(r) -> Checks:
    c = Checks("09 composite box couplings")
    for name, (i, j) in COUPLING.items():
        m = r["matrices"][name]
        Kt, Ks, Kf, Kr = (np.array(m[k]) for k in ("thin", "solid", "thin_flipped", "solid_refined"))
        term = f"K{i + 1}{j + 1}"
        c.close(f"{name} {term} / sqrt(K{i + 1}{i + 1} K{j + 1}{j + 1}), thin against solid",
                normalised(Kt, i, j), normalised(Ks, i, j), 0.03, "solid section")
        c.close(f"{name} {term} flips sign with the ply angles", normalised(Kf, i, j), -normalised(Kt, i, j), 1e-9,
                "symmetry")
        for k, lab in enumerate(TERMS):
            c.close(f"{name} K{k + 1}{k + 1} ({lab}), thin against solid", Kt[k, k], Ks[k, k], 0.03, "solid section")
        c.close(f"{name} solid {term} converged (mesh x 2)", Ks[i, j], Kr[i, j], 2e-3, "refined solid")
        cl = r["cantilevers"][name]
        tip_shell = cl["twist"][-1]
        c.close(f"{name} cantilever tip twist, beam (thin 6 x 6) against shell", cl["beam_tip_thin"], tip_shell, 0.02,
                "shell model")
        c.close(f"{name} shell section from the mesh = thin-wall section ({term})",
                normalised(np.array(cl["section_K"]), i, j), normalised(Kt, i, j), 1e-6, "same walls")
    s = r["sweep"]
    c.true("CUS extension-twist vanishes at 0 and 90 degrees", abs(s["CUS"]["thin"][0]) < 1e-9
           and abs(s["CUS"]["thin"][-1]) < 1e-9, "orthotropic walls")
    k = int(np.argmax(np.abs(s["CAS"]["solid"])))
    c.true(f"CAS bending-twist peaks between 15 and 40 degrees ({ANGLES[k]:g})", 15.0 <= ANGLES[k] <= 40.0,
           "solid section", ANGLES[k])
    return c


# ----------------------------------------------------------------------------- figures
def figure_couplings(r):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    labels = {"CUS": "CUS extension-twist K36", "CAS": "CAS bending-twist K46"}
    for ax, name, col in zip(axes[:2], ("CUS", "CAS"), SERIES):
        s = r["sweep"][name]
        ax.plot(s["theta"], s["solid"], "o", color=REFERENCE, mfc="none", label="solid section")
        ax.plot(s["theta"], s["thin"], color=col, label="thin-wall")
        ax.axhline(0.0, color=TEXT_SECONDARY, lw=0.6)
        ax.set_xlabel("ply angle θ [deg]")
        ax.set_ylabel("coupling / √(product of the diagonals)")
        ax.set_title(labels[name], loc="left")
        ax.set_xticks(range(0, 91, 15))
        ax.legend()
    ax = axes[2]
    for name, col, mk in (("CUS", SERIES[0], "o"), ("CAS", SERIES[1], "s")):
        s = r["sweep"][name]
        ratio = np.array(s["thin_diag"]) / np.array(s["solid_diag"])
        for k, ls in ((2, "-"), (3, "--"), (5, ":")):
            ax.plot(s["theta"], ratio[:, k], ls=ls, color=col, marker=mk, ms=3,
                    label=f"{name} {['GA_x', 'GA_y', 'EA', 'EI_x', 'EI_y', 'GJ'][k]}")
    ax.axhline(1.0, color=REFERENCE, lw=1, ls=":")
    ax.set_xlabel("ply angle θ [deg]")
    ax.set_ylabel("thin-wall / solid")
    ax.set_title("Diagonal terms: thin-wall within 3 %", loc="left")
    ax.set_xticks(range(0, 91, 15))
    ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Thin-wall theory reproduces the solid section's couplings at every ply angle", x=0.0, ha="left",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.text(0.0, -0.04, f"Box {box.WIDTH * 1e3:.2f} x {box.DEPTH * 1e3:.2f} mm outside, wall {box.WALL * 1e3:.3f} mm "
             f"of {box.PLIES} plies of {box.MATERIAL}. CUS: [θ]6 in every wall; CAS: top [θ]6, bottom [−θ]6, "
             "sides [θ/−θ]3, angles in each wall's own axes. Solid: one Q8 row per ply.",
             fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "box_couplings.png")


def figure_cantilevers(r):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    titles = {"CUS": f"CUS [{THETA:g}]6 under {AXIAL / 1e3:g} kN tension: it twists",
              "CAS": f"CAS [{THETA:g}]6 under a {FLAP:g} N flapwise tip force: it twists"}
    for ax, name in zip(axes, ("CUS", "CAS")):
        cl = r["cantilevers"][name]
        ax.plot(cl["z"], np.degrees(cl["twist"]), "o", color=REFERENCE, mfc="none", ms=4, label="shell model")
        ax.plot(cl["beam_z"], np.degrees(cl["beam_thin"]), color=SERIES[0], label="TimoshenkoBeam, thin-wall 6 x 6")
        ax.plot(cl["beam_z"], np.degrees(cl["beam_solid"]), color=SERIES[1], ls="--", label="TimoshenkoBeam, solid 6 x 6")
        ax.set_xlabel("z [m]")
        ax.set_ylabel("twist [deg]")
        ax.set_title(titles[name], loc="left")
        ax.legend()
    fig.text(0.0, -0.04, f"{LENGTH:g} m cantilever, root clamped, tip rib rigid in its plane and free to warp. Shell: "
             "Quad8 FSDT, 8 x 4 elements round the box, 40 along. The beam has no warping restraint, so the shell "
             "departs from it only in the clamp's boundary layer.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "box_cantilevers.png")


def report(r):
    for name, (i, j) in COUPLING.items():
        m = r["matrices"][name]
        Kt, Ks = np.array(m["thin"]), np.array(m["solid"])
        print(f"{name} [{THETA:g}]: K{i + 1}{j + 1} normalised thin {normalised(Kt, i, j):+.4f}, "
              f"solid {normalised(Ks, i, j):+.4f}; diagonals thin / solid "
              + " ".join(f"{a / b:.3f}" for a, b in zip(np.diag(Kt), np.diag(Ks))))
        cl = r["cantilevers"][name]
        print(f"  cantilever tip twist: shell {np.degrees(cl['twist'][-1]):.4f} deg, beam thin "
              f"{np.degrees(cl['beam_tip_thin']):.4f}, beam solid {np.degrees(cl['beam_tip_solid']):.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", action="store_true", help="redraw from data/results.json")
    args = parser.parse_args()
    if args.reuse and DATA.exists():
        r = json.loads(DATA.read_text())
    else:
        lib = create_default_library()
        r = {"sweep": sweep(lib), "matrices": matrices(lib)}
        r["cantilevers"] = cantilevers(lib, r["matrices"])
        DATA.parent.mkdir(exist_ok=True)
        DATA.write_text(json.dumps(r, indent=1))
    report(r)
    figure_couplings(r)
    figure_cantilevers(r)
    check(r).report()


if __name__ == "__main__":
    main()
