"""A tapered composite box beam: the Resal effect, and ply-thickness sensitivities for sizing it.

A carbon/glass box spar whose depth halves from root to tip, as a cantilever with a tip
shear force. Two things a beam model of it has to get right:

1. **Taper.** Each section along the beam is a cut of the box, but the beam is not a row
   of prismatic pieces. The caps are inclined to the axis, so they carry part of the shear
   force (the Resal effect): where the depth shrinks towards the tip, the webs carry
   ``V - M h'/h``, about half of ``V`` at the root here, and the beam is much stiffer in
   shear than its cuts say. ``thin_walled.taper.section_taper`` adds that coupling to
   each section's 6 x 6, and ``TimoshenkoBeam`` takes a stiffness per node and integrates
   the compliance as it varies along each element. Against a tapered shell model, the
   beam with the coupling is as close as the prismatic beam is to its shell (2 %); the
   cuts on their own are 29 % too soft however fine the mesh.
2. **Ply-thickness sensitivities.** For sizing the layup: the exact derivative of the tip
   deflection with respect to each ply's thickness, through the section stiffness
   (``section_derivative``), its taper coupling (``taper_derivative``) and the beam
   (``BeamSolution.displacement_derivative``). Checked against finite differences of the
   whole chain and of the shell model, and the 2D-meshed (solid) section's thickness
   derivatives against the thin-walled ones. Divided by the mass each ply adds, they rank
   where material stiffens the beam most.

Figures: tip deflection against the number of elements for the three beam models and the
shell; the deflection along the span; the ply sensitivities and their efficiency per kg.

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
from _shared.plotting import REFERENCE, SERIES, TEXT_SECONDARY, plt, save  # noqa: E402
from anisotropic_section_model.beam import TimoshenkoBeam, from_section_order  # noqa: E402
from anisotropic_section_model.solid import SectionMesh, SolidSectionAnalysis, solid_section_derivative  # noqa: E402
from anisotropic_section_model.thin_walled import ThinWalledSection, Wall, WallLaminate  # noqa: E402
from anisotropic_section_model.thin_walled.layup_sensitivity import LayupParameter, section_derivative  # noqa: E402
from anisotropic_section_model.thin_walled.section_analysis import ThinWalledSectionAnalysis  # noqa: E402
from anisotropic_section_model.thin_walled.taper import section_taper  # noqa: E402

DATA = HERE / "data" / "results.json"
WIDTH, ROOT_DEPTH, TIP_DEPTH, LENGTH = 0.4, 0.3, 0.15, 1.5     # m, on the wall mid-lines
PLY = 0.5e-3
CAP = [0, 0, 45, -45, -45, 45, 0, 0]           # T300/5208, outer surface first
WEB = [45, -45, -45, 45]                       # E-glass/epoxy
CAP_ID, WEB_ID = 1, 2
TIP_FORCE = 1.0e3                              # N, along the depth
ELEMENTS = [1, 2, 4, 8, 16]
SHELL_SPAN = 30                                # shell elements along the span
TAPER_STEP = 1e-3                              # m, between the two cuts of each taper coupling
PARAMETERS = [(CAP_ID, 0), (CAP_ID, 2), (CAP_ID, 3), (WEB_ID, 0), (WEB_ID, 1)]
SHELL_FD = [(CAP_ID, 0), (WEB_ID, 0)]          # the plies also checked against the shell
SHELL_STEP = 0.05e-3                           # m: a tenth of a ply, for the shell's differences
RESIZE = 0.1                                   # +10 % on a ply, to test the linear prediction


def depth(z):
    return ROOT_DEPTH + (TIP_DEPTH - ROOT_DEPTH) * z / LENGTH


# ----------------------------------------------------------------------------- models
def laminates(lib, change=None, dt=0.0):
    """``{CAP_ID: cap, WEB_ID: web}`` as ``WallLaminate`` s, ply ``change = (id, ply)`` thicker by ``dt``."""
    from laminate_constitutive_model import Laminate, Ply

    out = {}
    for lid, mat, angles in ((CAP_ID, "T300/5208", CAP), (WEB_ID, "E-glass/epoxy", WEB)):
        m = lib.get(mat)
        plies = [Ply(m, PLY + (dt if (lid, k) == change else 0.0), a) for k, a in enumerate(angles)]
        out[lid] = WallLaminate.from_laminate(Laminate(plies))
    return out


def section(h):
    """The box on its mid-lines in the section's (y, z): y across the width, z up; counter-
    clockwise, so every wall's normal points into the box and ply 1 is on the outside."""
    pts = {1: (-WIDTH / 2, -h / 2), 2: (WIDTH / 2, -h / 2), 3: (WIDTH / 2, h / 2), 4: (-WIDTH / 2, h / 2)}
    return ThinWalledSection(pts, [Wall(1, 1, 2, CAP_ID), Wall(2, 2, 3, WEB_ID), Wall(3, 3, 4, CAP_ID),
                                   Wall(4, 4, 1, WEB_ID)])


def station(lams, z, resal=True, prismatic=False):
    """(analysis, taper or None, right-handed 6 x 6) at span z."""
    h = ROOT_DEPTH if prismatic else depth(z)
    an = ThinWalledSectionAnalysis(section(h), lams, n_sub=4)
    K = an.stiffness
    taper = None
    if resal and not prismatic:
        taper = section_taper(section(depth(z - TAPER_STEP / 2)), section(depth(z + TAPER_STEP / 2)), lams, TAPER_STEP)
        K = K + taper.matrix()
    return an, taper, K


def tip(stiffness, z_nodes, shear=True):
    """Tip deflection along the depth (the beam's y) of the cantilever, and its solution."""
    s = TimoshenkoBeam(stiffness, z_nodes=z_nodes, shear=shear).clamp(0.0).add_force(LENGTH, [0.0, TIP_FORCE, 0.0]).solve()
    return float(s.node_displacement(LENGTH)[1]), s


def beams(lib):
    lams = laminates(lib)
    out = {"elements": ELEMENTS}
    for name, resal in (("tapered, Resal", True), ("tapered, cuts only", False)):
        out[name] = [tip(from_section_order([station(lams, v, resal)[2] for v in np.linspace(0, LENGTH, n + 1)]),
                         np.linspace(0, LENGTH, n + 1))[0] for n in ELEMENTS]
    stepped = []
    for n in ELEMENTS:
        z = np.linspace(0, LENGTH, n + 1)
        Km = [from_section_order(station(lams, v, False)[2]) for v in 0.5 * (z[:-1] + z[1:])]
        stepped.append(tip(lambda v, z=z, Km=Km: Km[min(int(np.searchsorted(z, v, side="right")) - 1, n - 1)], z)[0])
    out["stepped cuts"] = stepped
    z = np.linspace(0, LENGTH, 17)
    out["tapered, Euler-Bernoulli"] = tip(from_section_order([station(lams, v, False)[2] for v in z]), z, False)[0]
    out["prismatic"] = tip(from_section_order(station(lams, 0.0, prismatic=True)[2]), np.array([0.0, LENGTH]))[0]
    # the deflection along the span, 8 elements
    zf = np.linspace(0, LENGTH, 61)
    z8 = np.linspace(0, LENGTH, 9)
    out["z"] = zf.tolist()
    for name, resal in (("tapered, Resal", True), ("tapered, cuts only", False)):
        _, s = tip(from_section_order([station(lams, v, resal)[2] for v in z8]), z8)
        out[f"{name} deflection"] = s.displacement(zf)[:, 1].tolist()
    return out


# ----------------------------------------------------------------------------- shell
def shell_tip(lib, prismatic=False, change=None, dt=0.0, n_span=SHELL_SPAN):
    """Quad8 FSDT shell of the box, root clamped, the tip force on a rib: tip deflection and
    the deflection at the span stations. The walls are planar (the depth is linear in z)."""
    from laminate_constitutive_model import Laminate, Ply
    from shell_fem import BoundaryConditions, FEAMesh, FSDTModel, LoadCase, StructuralGlobalSolver

    from _shared.torsion import add_rib

    def shell_laminate(lid, mat, angles):
        # shell-fem's normals point out of the box: plies from the inside out, angles negated
        m = lib.get(mat)
        t = [PLY + (dt if (lid, k) == change else 0.0) for k in range(len(angles))]
        return Laminate([Ply(m, tk, -a) for a, tk in reversed(list(zip(angles, t)))])

    lam = {"cap": shell_laminate(CAP_ID, "T300/5208", CAP), "web": shell_laminate(WEB_ID, "E-glass/epoxy", WEB)}
    kin = FSDTModel()
    unit = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    counts, names = [8, 4, 8, 4], ["cap", "web", "cap", "web"]
    ring, wall = [], []
    for k in range(4):
        p, q = np.array(unit[k]), np.array(unit[(k + 1) % 4])
        for i in range(2 * counts[k]):
            ring.append(p + (q - p) * i / (2 * counts[k]))
            wall.append(k)
    nr = len(ring)
    zs = np.linspace(0.0, LENGTH, 2 * n_span + 1)
    mesh, nid, n = FEAMesh(), {}, 1
    for j, z in enumerate(zs):
        h = ROOT_DEPTH if prismatic else depth(z)
        for i in range(nr):
            if i % 2 and j % 2:
                continue
            mesh.add_node(n, (ring[i][0] * WIDTH, ring[i][1] * h, z))
            nid[(i, j)] = n
            n += 1
    eid = 1
    for jj in range(n_span):
        j = 2 * jj
        for ii in range(0, nr, 2):
            i2 = (ii + 2) % nr
            conn = [nid[(ii, j)], nid[(i2, j)], nid[(i2, j + 2)], nid[(ii, j + 2)], nid[(ii + 1, j)],
                    nid[(i2, j + 1)], nid[(ii + 1, j + 2)], nid[(ii, j + 1)]]
            mesh.add_element(eid, conn, lam[names[wall[ii]]], kin, element_type="quad8", material_axis=(0.0, 0.0, 1.0))
            eid += 1
    root = [m for m in mesh.node_ids if abs(mesh.node_coordinates(m)[2]) < 1e-12]
    bcs = BoundaryConditions().clamp(root)
    ref = add_rib(mesh, bcs, LENGTH, (0.0, 0.0))
    sol = StructuralGlobalSolver().solve(mesh, bcs, LoadCase("tip shear").add_nodal_load(ref, 1, TIP_FORCE))
    xyz = {m: np.asarray(mesh.node_coordinates(m)) for m in mesh.node_ids if m != ref}
    span = []
    for z in zs[::2]:
        nodes = [m for m, p in xyz.items() if abs(p[2] - z) < 1e-9]
        span.append(float(np.mean([sol.node_displacement(m)[1] for m in nodes])))
    return float(sol.node_displacement(ref)[1]), zs[::2].tolist(), span


def shells(lib):
    tip_t, z, span = shell_tip(lib)
    tip_fine = shell_tip(lib, n_span=2 * SHELL_SPAN)[0]
    tip_p = shell_tip(lib, prismatic=True)[0]
    fd = {}
    for p in SHELL_FD:
        a, b = (shell_tip(lib, change=p, dt=s)[0] for s in (SHELL_STEP, -SHELL_STEP))
        fd[f"{p[0]}.{p[1]}"] = (a - b) / (2 * SHELL_STEP)
    return {"tapered": tip_t, "tapered, fine": tip_fine, "prismatic": tip_p, "z": z, "deflection": span,
            "thickness FD": fd}


# ----------------------------------------------------------------------------- sensitivities
def chain(lib, n_elements=8):
    """Tip deflection and its exact ply-thickness derivatives, with the mass each ply adds."""
    lams = laminates(lib)
    z = np.linspace(0, LENGTH, n_elements + 1)
    st = [station(lams, v) for v in z]
    u, sol = tip(from_section_order([s[2] for s in st]), z)
    out = {"tip": u, "parameters": [], "exact": [], "fd": [], "mass rate": [], "resize": []}
    for lid, k in PARAMETERS:
        p = LayupParameter(lid, k, "thickness")
        dK, dm = [], []
        for an, taper, _ in st:
            d = section_derivative(an, p, taper=taper)
            dK.append(from_section_order(d.stiffness + d.taper.matrix()))
            dm.append(d.mass["m"])
        du = float(sol.displacement_derivative(np.array(dK))[-1, 1])
        h = 1e-6
        fd = (tip(from_section_order([station(laminates(lib, (lid, k), h), v)[2] for v in z]), z)[0]
              - tip(from_section_order([station(laminates(lib, (lid, k), -h), v)[2] for v in z]), z)[0]) / (2 * h)
        # the linear prediction of a 10 % thicker ply against the re-solved beam
        dt = RESIZE * PLY
        resized = tip(from_section_order([station(laminates(lib, (lid, k), dt), v)[2] for v in z]), z)[0]
        out["parameters"].append(f"{lid}.{k}")
        out["exact"].append(du)
        out["fd"].append(fd)
        out["mass rate"].append(float(np.trapezoid(dm, z)))
        out["resize"].append([u + du * dt, resized])
    return out


def solid_check(lib):
    """The root section meshed in 2D: its thickness derivatives against the thin-walled ones."""
    lams = laminates(lib)
    sec = section(ROOT_DEPTH)
    an = SolidSectionAnalysis(SectionMesh.from_thin_walled(sec, lams, elements_along=12))
    tw = ThinWalledSectionAnalysis(sec, lams, n_sub=4)
    out = {}
    for lid, k in ((CAP_ID, 0), (CAP_ID, 3), (WEB_ID, 0)):
        p = LayupParameter(lid, k, "thickness")
        ds, dt = solid_section_derivative(an, p), section_derivative(tw, p)
        out[f"{lid}.{k}"] = {"solid": [float(ds.stiffness[i, i]) for i in range(6)],
                             "thin": [float(dt.stiffness[i, i]) for i in range(6)],
                             "solid mass": ds.mass["m"], "thin mass": dt.mass["m"]}
    return out


# ----------------------------------------------------------------------------- report and figures
LABELS = {f"{CAP_ID}.{k}": f"cap ply {k + 1} ({CAP[k]:+d}°)" for k in range(len(CAP))}
LABELS.update({f"{WEB_ID}.{k}": f"web ply {k + 1} ({WEB[k]:+d}°)" for k in range(len(WEB))})


def report(r):
    b, s = r["beams"], r["shell"]
    print(f"Tip deflection under {TIP_FORCE:g} N (mm); shell {1e3 * s['tapered']:.4f} "
          f"({1e3 * s['tapered, fine']:.4f} with twice the elements)")
    print("elements  tapered+Resal  cuts only  stepped cuts")
    for i, n in enumerate(b["elements"]):
        print(f"{n:8d}  {1e3 * b['tapered, Resal'][i]:13.4f}  {1e3 * b['tapered, cuts only'][i]:9.4f}  "
              f"{1e3 * b['stepped cuts'][i]:12.4f}")
    print(f"prismatic (root section): beam {1e3 * b['prismatic']:.4f}, shell {1e3 * s['prismatic']:.4f}")
    print("root section, d / d t_ply: solid against thin-wall (EA, EI_y, EI_z, GJ, mass)")
    for p, d in r["solid"].items():
        ratios = [d["solid"][i] / d["thin"][i] for i in (0, 1, 2, 5)] + [d["solid mass"] / d["thin mass"]]
        print(f"  {LABELS[p]:<18} " + "  ".join(f"{v:.3f}" for v in ratios))
    c = r["chain"]
    print("ply            d tip / d t [mm/mm]  finite diff.  per kg added [mm/kg]")
    for i, p in enumerate(c["parameters"]):
        print(f"{LABELS[p]:<18} {c['exact'][i]:12.5f}  {c['fd'][i]:12.5f}  {1e3 * c['exact'][i] / c['mass rate'][i]:10.4f}")


def figure_taper(r):
    b, s = r["beams"], r["shell"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.8))
    n = b["elements"]
    for k, name in enumerate(("tapered, Resal", "tapered, cuts only", "stepped cuts")):
        ax.plot(n, 1e3 * np.array(b[name]), "o-", color=SERIES[k], label=name)
    ax.axhline(1e3 * s["tapered"], color=REFERENCE, ls="--", label="shell")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("beam elements")
    ax.set_ylabel("tip deflection [mm]")
    ax.set_title("Tapered box cantilever, tip shear")
    ax.legend(frameon=False)
    z = np.array(b["z"])
    for k, name in enumerate(("tapered, Resal", "tapered, cuts only")):
        bx.plot(z, 1e3 * np.array(b[f"{name} deflection"]), color=SERIES[k], label=f"beam, {name}")
    bx.plot(s["z"], 1e3 * np.array(s["deflection"]), "o", ms=3, color=REFERENCE, label="shell")
    bx.set_xlabel("span z [m]")
    bx.set_ylabel("deflection [mm]")
    bx.set_title("Deflection along the span (8 elements)")
    bx.legend(frameon=False)
    fig.tight_layout()
    save(fig, __file__, "tapered_cantilever.png")


def figure_sensitivities(r):
    c = r["chain"]
    labels = [LABELS[p] for p in c["parameters"]]
    x = np.arange(len(labels))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax.bar(x - 0.2, c["exact"], 0.4, color=SERIES[0], label="exact")
    ax.bar(x + 0.2, c["fd"], 0.4, color=SERIES[1], label="finite difference")
    for p, v in r["shell"]["thickness FD"].items():
        i = c["parameters"].index(p)
        ax.plot(i, v, "kD", ms=5, label="shell, finite difference" if i == 0 else None)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("d tip / d t_ply [mm per mm]")
    ax.set_title("Tip deflection sensitivity to ply thickness")
    ax.legend(frameon=False)
    eff = 1e3 * np.array(c["exact"]) / np.array(c["mass rate"])
    bx.barh(x, eff, color=SERIES[2])
    bx.set_yticks(x, labels)
    bx.invert_yaxis()
    bx.set_xlabel("deflection change per kg added [mm/kg]")
    bx.set_title("Where material stiffens the beam most")
    bx.axvline(0.0, color=TEXT_SECONDARY, lw=0.8)
    fig.tight_layout()
    save(fig, __file__, "ply_sensitivities.png")


# ----------------------------------------------------------------------------- checks
def check(r) -> Checks:
    c = Checks("12 tapered beams and layup sensitivities")
    b, s = r["beams"], r["shell"]
    i8 = b["elements"].index(8)
    c.close("shell tip converged (twice the elements)", s["tapered"], s["tapered, fine"], 0.002, "shell")
    c.close("prismatic beam against its shell (baseline)", b["prismatic"], s["prismatic"], 0.03, "shell")
    c.close("tapered beam with Resal coupling against shell", b["tapered, Resal"][i8], s["tapered"], 0.03, "shell")
    err_cuts = abs(b["tapered, cuts only"][i8] / s["tapered"] - 1)
    c.true("cuts on their own: over 20 % too soft", err_cuts > 0.2, "shell", err_cuts)
    c.close("stepped cuts converge to the cuts-only tapered beam", b["stepped cuts"][-1], b["tapered, cuts only"][-1],
            0.002, "16 elements")
    c.close("tapered elements converged at 8 elements", b["tapered, Resal"][i8], b["tapered, Resal"][-1], 0.002,
            "16 elements")
    c.close("tapered, Euler-Bernoulli below every shear model", min(1.0, b["tapered, Euler-Bernoulli"]
                                                                     / b["tapered, Resal"][i8]), 1.0, 0.5, "")
    ch = r["chain"]
    for i, p in enumerate(ch["parameters"]):
        c.close(f"d tip / d t, {LABELS[p]}: exact against finite difference", ch["exact"][i], ch["fd"][i], 1e-5,
                "beam chain")
        c.close(f"+10 % {LABELS[p]}: linear prediction", ch["resize"][i][0], ch["resize"][i][1], 2e-3, "re-solved")
    for p, v in s["thickness FD"].items():
        i = ch["parameters"].index(p)
        c.close(f"d tip / d t, {LABELS[p]}: beam against shell", ch["exact"][i], v, 0.08, "shell, finite difference")
    # the cap plies' rates agree with thin-wall theory; the web plies' do not, because in the
    # 2D mesh a thicker web also lengthens the caps that run over it at the butt-jointed corners
    for p, d in r["solid"].items():
        if p.startswith(f"{CAP_ID}."):
            for i, name in ((0, "EA"), (1, "EI_y"), (2, "EI_z")):
                c.close(f"root d {name} / d t, {LABELS[p]}: solid against thin-wall", d["solid"][i], d["thin"][i],
                        0.03, "thin-walled section")
            c.close(f"root d m / d t, {LABELS[p]}: solid against thin-wall", d["solid mass"], d["thin mass"], 0.01,
                    "thin-walled section")
    web = r["solid"][f"{WEB_ID}.0"]
    c.true("web ply: the 2D corners make d EI_y / d t larger than thin-wall's", web["solid"][1] > 1.1 * web["thin"][1],
           "butt-jointed corners", web["solid"][1] / web["thin"][1])
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", action="store_true", help="redraw from data/results.json")
    args = parser.parse_args()
    if args.reuse and DATA.exists():
        r = json.loads(DATA.read_text())
    else:
        from laminate_constitutive_model import create_default_library

        lib = create_default_library()
        r = {"beams": beams(lib), "shell": shells(lib), "chain": chain(lib), "solid": solid_check(lib)}
        DATA.parent.mkdir(exist_ok=True)
        DATA.write_text(json.dumps(r, indent=1))
    report(r)
    figure_taper(r)
    figure_sensitivities(r)
    check(r).report()


if __name__ == "__main__":
    main()
