"""Beam section stiffness beyond the classical 4 x 4: Timoshenko 6 x 6 and Timoshenko-Vlasov 7 x 7.

The classical stiffness of a blade section relates [N, Mx, My, Mz] to [eps, kappa_x,
kappa_y, kappa_z]: extension, two bending curvatures and the twist rate. It leaves out
two kinds of deformation that the torsion examples (27, 28) already showed matter:

- transverse shear: the Timoshenko 6 x 6 adds the two shear strains, so a beam code
  (BeamDyn, HAWC2, ...) can carry shear deflection and the couplings of shear with
  twist (the shear centre) and with bending;
- restrained warping: the Timoshenko-Vlasov 7 x 7 adds the warping amplitude, whose
  gradient is the seventh strain and whose force is the bimoment, with EΓ on the
  diagonal (Benscoter's shear stiffness S of the warping goes with it for closed
  sections).

Each matrix comes two ways that share no code: thin-wall section theory
(``ThinWalledSection.timoshenko_stiffness`` / ``vlasov_stiffness``) and the shell model
loaded as a cantilever (``beam_section_properties``). They are compared on a composite
box spar, with the shear terms checked against the tip deflection of short and long
cantilevers, and on six stations of the IEA 15 MW blade against the published 6 x 6.

Run with ``--reuse`` to redraw the figures from ``data/results.json`` without solving.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.iea15mw import DATA, build_segment
from _shared.plotting import DIVERGING, REFERENCE, SERIES, TEXT_SECONDARY, plt, save
from _shared.torsion import add_rib, box_spar, section_nodes
from anisotropic_section_model import RestrainedTorsionBeam, ThinWalledSection
from composite_laminate_model import (BoundaryConditions, Laminate, LoadCase,
    StructuralGlobalSolver, add_reference_node, beam_section_properties, create_default_library,
    rigid_link, transform_beam_stiffness)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "data" / "results.json"
WIDTH, HEIGHT, LENGTH = 0.5, 0.25, 4.0
LENGTHS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]  # cantilevers for the shear deflection check
LABELS = ["γx", "γy", "ε", "κx", "κy", "κz"]
FORCES = ["Fx", "Fy", "N", "Mx", "My", "Mz"]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def laminates(lib, cap_angle=0.0):
    """Example 01's box: carbon caps, glass ±45 webs. ``cap_angle`` turns the caps' 0° plies."""
    carbon, glass = lib.get("T300/5208"), lib.get("E-glass/epoxy")
    a = cap_angle
    cap = Laminate.from_angles(carbon, [a, a, 45, -45, -45, 45, a, a], 0.5e-3, "cap")
    web = Laminate.from_angles(glass, [45, -45, -45, 45], 0.5e-3, "web")
    return {"cap_top": cap, "cap_bottom": cap, "web_le": web, "web_te": web}


def make_box(lib, length=LENGTH, cap_angle=0.0, per_metre=8):
    box = box_spar(WIDTH, HEIGHT, length, laminates(lib, cap_angle), 6, 3, max(12, int(round(per_metre * length))))
    box.section_nodes = lambda z: section_nodes(box.mesh, z)
    return box


def beam_tip_displacement(C6, length, force):
    """Tip displacement of a cantilever (root clamped) under a tip force/moment ``force``
    (6 values, BeamDyn order, at the reference axis), from a 6 x 6 compliance.

    Integrates the beam strains e(z) = C f(z): theta' = kappa, u_x' = gamma_x + theta_y,
    u_y' = gamma_y - theta_x, u_z' = eps (an Euler-Bernoulli beam is the compliance with
    zero shear rows and columns).
    """
    z = np.linspace(0.0, length, 4001)
    F = np.asarray(force, dtype=float)
    f = np.tile(F, (len(z), 1))
    f[:, 3] = F[3] - (length - z) * F[1]
    f[:, 4] = F[4] + (length - z) * F[0]
    e = f @ C6.T
    theta = np.vstack([np.zeros(3), np.cumsum(0.5 * (e[1:, 3:] + e[:-1, 3:]) * np.diff(z)[:, None], axis=0)])
    du = np.column_stack([e[:, 0] + theta[:, 1], e[:, 1] - theta[:, 0], e[:, 2]])
    return np.trapezoid(du, z, axis=0) if hasattr(np, "trapezoid") else np.trapz(du, z, axis=0)


def shell_tip_displacement(box, length, force):
    """Tip displacement of the shell cantilever: root clamped, load through a rigid tip diaphragm."""
    mesh = box.mesh.copy()
    bcs = BoundaryConditions().clamp(box.nodes("root"))
    tip = add_reference_node(mesh, (0.0, 0.0, length))
    rigid_link(mesh, bcs, tip, box.nodes("tip"), dofs=(0, 1, 2))
    case = LoadCase("tip")
    for dof, value in enumerate(force):
        if value:
            case.add_nodal_load(tip, dof, float(value))
    sol = StructuralGlobalSolver().solve(mesh, bcs, case)
    return sol.node_displacement(tip)[:3]


def rigid_section_twist(box, centre, torque=1.0):
    """Twist along the shell box with every section rigid in its plane (free to warp), root clamped."""
    mesh = box.mesh.copy()
    bcs = BoundaryConditions().clamp(box.nodes("root"))
    zs = box.span_positions
    refs = [add_rib(mesh, bcs, float(z), centre) for z in zs[1:]]
    sol = StructuralGlobalSolver().solve(mesh, bcs, LoadCase("T").add_nodal_load(refs[-1], 5, torque))
    return np.array([0.0] + [sol.node_displacement(r)[5] for r in refs])


def publish_frame(K):
    """The published 6 x 6 in the shell model's axes.

    The windIO / BeamDyn data has x flapwise and y chordwise aft, both from the pitch
    axis, a mirror image of the shell axes (x chordwise aft, y flapwise): swap x and y,
    and as moments are pseudo-vectors, flip their sign under the reflection.
    """
    P = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    Q = np.zeros((6, 6))
    Q[:3, :3] = P
    Q[3:, 3:] = -P
    return Q @ np.asarray(K) @ Q.T


def shear_centre(K):
    C = np.linalg.inv(K)
    return -C[5, 1] / C[5, 5], C[5, 0] / C[5, 5]


def tension_centre(K):
    C4 = np.linalg.inv(K)[2:, 2:]
    return tuple(np.linalg.solve(np.column_stack([-C4[1:3, 2], C4[1:3, 1]]), -C4[1:3, 0]))


def at_centres(K):
    """Frame-free numbers: shear terms at the shear centre, axial and bending terms at the
    tension centre (classical, shear forces free), GJ with the shear free."""
    xs, ys = shear_centre(K)
    Ks = transform_beam_stiffness(K, (xs, ys))
    C = np.linalg.inv(K)
    S4 = np.linalg.inv(C[2:, 2:])
    B = S4[1:3, 1:3] - np.outer(S4[1:3, 0], S4[0, 1:3]) / S4[0, 0]
    return {"GA_x": Ks[0, 0], "GA_y": Ks[1, 1], "GA_xy": Ks[0, 1], "EA": S4[0, 0], "EI_x": B[0, 0], "EI_y": B[1, 1],
            "GJ": 1.0 / C[5, 5], "sc": (xs, ys), "tc": tension_centre(K)}


def normalised(K):
    d = np.sqrt(np.abs(np.diag(K)))
    return K / np.outer(d, d)


# ---------------------------------------------------------------------------
# Solves
# ---------------------------------------------------------------------------
def solve_box(lib):
    out = {}
    box = make_box(lib)
    sec = ThinWalledSection.from_blade_mesh(box, z=LENGTH / 2)
    t = time.perf_counter()
    props = beam_section_properties(box, warping=True, end_ribs=0.2)
    bare = beam_section_properties(box)
    print(f"box spar shell 6 x 6 and 7 x 7: {time.perf_counter() - t:.0f} s")
    out["shell_K6"] = props.stiffness_6x6.tolist()
    out["shell_K6_bare"] = bare.stiffness_6x6.tolist()
    out["shell_EGamma"], out["shell_S"] = props.EGamma, props.warping_shear_stiffness
    out["thin_K6"] = sec.timoshenko_stiffness().tolist()
    out["thin_EGamma"], out["thin_S"], out["thin_GJ"] = sec.EGamma, sec.S_warping, sec.GJ
    # a tailored box: caps turned 20° couple bending, twist and shear
    tailored = make_box(lib, cap_angle=20.0)
    out["tailored_shell_K6"] = beam_section_properties(tailored, end_ribs=0.2).stiffness_6x6.tolist()
    out["tailored_thin_K6"] = ThinWalledSection.from_blade_mesh(tailored, z=LENGTH / 2).timoshenko_stiffness().tolist()
    # restrained warping with rigid sections
    zs = box.span_positions
    out["warp_z"] = list(map(float, zs))
    out["warp_shell"] = rigid_section_twist(box, sec.shear_centre).tolist()
    # shear deflection of short and long cantilevers
    defl = []
    for L in LENGTHS:
        b = make_box(lib, L)
        defl.append([float(v) for v in shell_tip_displacement(b, L, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0))])
        defl[-1] += [float(v) for v in shell_tip_displacement(b, L, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0))]
        print(f"  cantilever L = {L:g} m: tip deflection {defl[-1][1]:.4e} m/N (Fy), {defl[-1][3]:.4e} m/N (Fx)")
    out["defl"] = defl
    return out


def solve_iea():
    rows = []
    for st in DATA["stations"]:
        t = time.perf_counter()
        blade = build_segment(st, 8 * st["chord"], 32, 0.015, "quad8_curved", mesh_surface="mid")
        tube = st["index"] == DATA["stations"][0]["index"]
        props = beam_section_properties(blade, end_ribs=0.2, warping=not tube)
        sec = ThinWalledSection.from_blade_mesh(blade)
        rows.append({"r": st["r"], "span": st["span_fraction"], "chord": st["chord"], "pitch_axis": st["pitch_axis"],
                     "shell_K6": props.stiffness_6x6.tolist(), "thin_K6": sec.timoshenko_stiffness().tolist(),
                     "pub_K6": publish_frame(st["published"]["K"]).tolist(),
                     "shell_EGamma": props.EGamma, "shell_S": props.warping_shear_stiffness,
                     "thin_EGamma": sec.EGamma, "thin_S": sec.S_warping, "thin_mu": sec.benscoter_mu,
                     "elements": len(blade.mesh.shell_elements)})
        print(f"IEA 15 MW r = {st['r']:.1f} m: {len(blade.mesh.shell_elements)} elements, {time.perf_counter() - t:.0f} s")
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def heatmap(ax, K, title, labels=LABELS):
    N = normalised(np.asarray(K))
    im = ax.imshow(N, cmap=DIVERGING, vmin=-1, vmax=1)
    for i in range(N.shape[0]):
        for j in range(N.shape[1]):
            v = N[i, j]
            if i == j:
                txt = "1"
            elif abs(v) < 5e-4:
                txt = "·"
            else:
                txt = f"{v:+.2f}" if abs(v) >= 0.01 else f"{v:+.0e}".replace("e-0", "e-")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.8, color="white" if abs(v) > 0.6 else "#0b0b0b")
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_yticks(range(len(labels)), FORCES[:len(labels)], fontsize=8)
    ax.grid(False)
    ax.set_title(title, fontsize=9.5)
    return im


def figure_box(box):
    lib = create_default_library()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    im = heatmap(axes[0], box["thin_K6"], "Box spar: thin-wall theory")
    heatmap(axes[1], box["shell_K6"], "Box spar: shell model")
    heatmap(axes[2], box["tailored_thin_K6"], "20° caps: thin-wall theory")
    heatmap(axes[3], box["tailored_shell_K6"], "20° caps: shell model")
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, label="K_ij / sqrt(K_ii K_jj)")
    fig.suptitle("The 6 × 6 of a box spar about its centre: thin-wall theory and the shell agree; "
                 "only the shell sees the couplings of off-axis caps", x=0.0, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.0, -0.05, "Box 0.5 × 0.25 m, carbon caps [0₂/±45/∓45/0₂] (or 20° in place of 0°), glass [±45]s webs; "
             "flat Quad8 FSDT, 4 m segment, middle fifth, ribs over the end fifths. Rows are the forces, columns the strains.",
             fontsize=7.5, color=TEXT_SECONDARY)
    save(fig, __file__, "box_spar_6x6.png")
    del lib


def figure_deflection(box):
    K_shell, K_thin = np.asarray(box["shell_K6"]), np.asarray(box["thin_K6"])
    L = np.array(LENGTHS)
    d = np.array(box["defl"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for ax, (col, force, name) in zip(axes, ((1, (0, 1, 0, 0, 0, 0), "flapwise, Fy (shear through the webs)"),
                                               (0, (1, 0, 0, 0, 0, 0), "chordwise, Fx (shear through the caps)"))):
        shell = d[:, 1] if col == 1 else d[:, 3]
        for K, label, colour, ls in ((K_shell, "shell 6 × 6", SERIES[0], "-"), (K_thin, "thin-wall 6 × 6", SERIES[1], "--")):
            C = np.linalg.inv(K)
            tim = np.array([beam_tip_displacement(C, x, force)[col] for x in L])
            ax.plot(L / HEIGHT if col == 1 else L / WIDTH, tim / shell, ls, color=colour, marker="o", ms=4,
                    label=f"Timoshenko, {label}")
        C_eb = np.zeros((6, 6))
        C_eb[2:, 2:] = np.linalg.inv(K_shell)[2:, 2:]
        eb = np.array([beam_tip_displacement(C_eb, x, force)[col] for x in L])
        ax.plot(L / HEIGHT if col == 1 else L / WIDTH, eb / shell, ":", color=REFERENCE, marker="s", ms=4, mfc="none",
                label="Euler-Bernoulli (classical 4 × 4)")
        ax.axhline(1.0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("length / section depth in the load direction")
        ax.set_ylabel("beam tip deflection / shell tip deflection")
        ax.set_title(name, fontsize=9.5)
        ax.legend(fontsize=8)
    fig.suptitle("Shear deflection: the Timoshenko 6 × 6 follows the shell down to stubby cantilevers; "
                 "the classical beam misses the shear", x=0.0, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.0, -0.04, "Cantilevers of the box spar, root clamped, unit tip force through a rigid tip diaphragm. "
             "The remaining gap at short lengths is the clamp's restraint of the shear warping (shear lag), "
             "which a beam theory does not hold.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "box_spar_shear_deflection.png")


def figure_warping(box):
    z = np.asarray(box["warp_z"])
    shell = np.asarray(box["warp_shell"])
    GJ = box["thin_GJ"]
    zf = np.linspace(0.0, LENGTH, 401)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    curves = [
        ("Saint-Venant (classical 4 × 4)", RestrainedTorsionBeam(GJ, 1.0, LENGTH).saint_venant(1.0, zf).twist, REFERENCE, ":"),
        ("Vlasov, thin-wall 7 × 7 (EΓ, S = ∞)", RestrainedTorsionBeam(GJ, box["thin_EGamma"], LENGTH)
         .cantilever_tip_torque(1.0, zf).twist, SERIES[3], "-."),
        ("Benscoter, thin-wall EΓ and S", RestrainedTorsionBeam(GJ, box["thin_EGamma"], LENGTH, box["thin_S"])
         .cantilever_tip_torque(1.0, zf).twist, SERIES[1], "--"),
        ("Benscoter, shell EΓ and S", RestrainedTorsionBeam(GJ, box["shell_EGamma"], LENGTH, box["shell_S"])
         .cantilever_tip_torque(1.0, zf).twist, SERIES[0], "-"),
    ]
    sv = zf / GJ
    for label, tw, colour, ls in curves:
        axes[0].plot(zf, tw * GJ, ls, color=colour, label=label, lw=1.6)
        if "Saint" not in label:
            axes[1].plot(zf, (tw - sv) * GJ * 1e3, ls, color=colour, label=label, lw=1.6)
    axes[0].plot(z, shell * GJ, "o", color=REFERENCE, mfc="none", ms=4, label="shell, sections rigid in their plane")
    axes[1].plot(z, (shell - z / GJ) * GJ * 1e3, "o", color=REFERENCE, mfc="none", ms=4, label="shell")
    axes[0].set_ylabel("GJ θ / T [m]")
    axes[1].set_ylabel("twist lost to the warping restraint, GJ (θ − θ_SV) / T [mm]")
    for ax in axes:
        ax.set_xlabel("z [m]")
        ax.legend(fontsize=8)
    axes[1].set_xlim(0, 1.5)
    axes[0].set_title("Twist under a tip torque", fontsize=9.5)
    axes[1].set_title("Near the clamp", fontsize=9.5)
    fig.suptitle(f"Warping: the 7th DOF. Shell EΓ = {box['shell_EGamma']:.3g} N m⁴ against thin-wall "
                 f"{box['thin_EGamma']:.3g} ({box['shell_EGamma'] / box['thin_EGamma'] - 1:+.1%}); "
                 "a closed box needs Benscoter's warping shear", x=0.0, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.0, -0.04, "Box spar clamped at the root, unit tip torque; every shell section tied rigid in its plane "
             "(free to warp) so that only warping, not distortion, acts. The shell's EΓ and S are fitted to its twist "
             "by beam_section_properties(warping=True).", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "box_spar_warping.png")


def figure_iea(rows):
    s = np.array([r["span"] for r in rows])
    summ = {k: [at_centres(np.asarray(r[f"{k}_K6"])) for r in rows] for k in ("shell", "thin", "pub")}
    style = {"shell": ("o-", SERIES[0], "shell model (this package)"), "thin": ("s--", SERIES[1], "thin-wall theory (this package)"),
             "pub": ("D:", REFERENCE, "published IEA 15 MW 6 × 6")}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8))
    panels = [("GA_y", "flapwise shear stiffness GA [N]"), ("GA_x", "edgewise shear stiffness GA [N]"),
              ("GJ", "GJ [N m²]"), ("EA", "EA [N]"), ("sc", "shear centre, x / chord from the pitch axis")]
    for ax, (key, label) in zip(axes.ravel(), panels):
        for k, (fmt, colour, name) in style.items():
            if key == "sc":
                y = [v["sc"][0] / r["chord"] for v, r in zip(summ[k], rows)]
                ax.plot(s, y, fmt, color=colour, mfc="none" if k != "shell" else colour, label=name, ms=5)
            else:
                ax.plot(s, [v[key] for v in summ[k]], fmt, color=colour, mfc="none" if k != "shell" else colour,
                        label=name, ms=5)
        if key != "sc":
            ax.set_yscale("log")
        ax.set_xlabel("span fraction")
        ax.set_title(label, fontsize=9.5)
    ax = axes[1, 2]
    sh = [(r["span"], r["shell_EGamma"]) for r in rows if r["shell_EGamma"]]
    ax.plot([a for a, _ in sh], [b for _, b in sh], "o-", color=SERIES[0], label="shell (rigid-section fit)")
    ax.plot(s[1:], [r["thin_EGamma"] for r in rows[1:]], "s--", color=SERIES[1], mfc="none", label="thin-wall")
    ax.set_yscale("log")
    ax.set_xlabel("span fraction")
    ax.set_title("warping stiffness EΓ [N m⁴] (7 × 7; not published)", fontsize=9.5)
    ax.legend(fontsize=7.5)
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle("IEA 15 MW blade: the 6 × 6 shear, torsion and axial terms against the published data, "
                 "and the warping stiffness of the 7 × 7", x=0.0, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.0, -0.03, "Six stations as prismatic segments 8 chords long; curved Quad8 FSDT on the mid-surface; "
             "middle fifth measured with ribs over the end fifths. Shear terms at each source's own shear centre, "
             "EA with the shear free. The root tube (span 0.03) has no meaningful warping.", fontsize=7.5,
             color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "iea15mw_6x6_terms.png")

    k = 3
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, (key, name) in zip(axes, (("shell_K6", "shell model"), ("thin_K6", "thin-wall theory"),
                                      ("pub_K6", "published"))):
        K = np.asarray(rows[k][key])
        im = heatmap(ax, transform_beam_stiffness(K, at_centres(np.asarray(rows[k]["pub_K6"]))["tc"]), name)
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label="K_ij / sqrt(K_ii K_jj)")
    fig.suptitle(f"IEA 15 MW at r = {rows[k]['r']:.1f} m: the normalised 6 × 6 about the published tension centre",
                 x=0.0, ha="left", fontweight="bold", fontsize=11)
    save(fig, __file__, "iea15mw_6x6_couplings.png")


def report(box, rows):
    Ks, Kt = np.asarray(box["shell_K6"]), np.asarray(box["thin_K6"])
    Cs, Ct = np.linalg.inv(Ks), np.linalg.inv(Kt)
    print("\nBox spar 6 x 6 about the box centre (compliance diagonals as stiffnesses):")
    for i, name in enumerate(["GA_x", "GA_y", "EA", "EI_x", "EI_y", "GJ"]):
        print(f"  {name:<5} shell {1 / Cs[i, i]:.4e}  thin-wall {1 / Ct[i, i]:.4e}  ({Ct[i, i] / Cs[i, i] - 1:+.2%})")
    Cb = np.linalg.inv(np.asarray(box["shell_K6_bare"]))
    print(f"  GJ without end ribs {1 / Cb[5, 5]:.4e} ({Cs[5, 5] / Cb[5, 5] - 1:+.2%} on with ribs); thin-wall GJ {box['thin_GJ']:.4e}")
    print(f"  EGamma shell {box['shell_EGamma']:.4e}, thin-wall {box['thin_EGamma']:.4e} "
          f"({box['shell_EGamma'] / box['thin_EGamma'] - 1:+.1%}); S shell {box['shell_S']:.4e}, thin-wall {box['thin_S']:.4e}")
    for name in ("tailored_shell_K6", "tailored_thin_K6"):
        N = normalised(np.asarray(box[name]))
        print(f"  {name}: bend-twist {N[3, 5]:+.3f} {N[4, 5]:+.3f}, extension-bending {N[2, 3]:+.3f} {N[2, 4]:+.3f}, "
              f"shear-bending {N[0, 4]:+.3f} {N[1, 3]:+.3f}, extension-twist {N[2, 5]:+.3f}")
    d = np.array(box["defl"])
    print("Cantilever tip deflection, beam / shell (Fy): L, Timoshenko shell 6x6, thin 6x6, Euler-Bernoulli")
    C_eb = np.zeros((6, 6))
    C_eb[2:, 2:] = Cs[2:, 2:]
    for L, row in zip(LENGTHS, d):
        f = (0, 1, 0, 0, 0, 0)
        print(f"  {L:4.2f} m  {beam_tip_displacement(Cs, L, f)[1] / row[1]:.4f}  {beam_tip_displacement(Ct, L, f)[1] / row[1]:.4f}"
              f"  {beam_tip_displacement(C_eb, L, f)[1] / row[1]:.4f}")
    print("Cantilever tip deflection, beam / shell (Fx):")
    for L, row in zip(LENGTHS, d):
        f = (1, 0, 0, 0, 0, 0)
        print(f"  {L:4.2f} m  {beam_tip_displacement(Cs, L, f)[0] / row[3]:.4f}  {beam_tip_displacement(Ct, L, f)[0] / row[3]:.4f}"
              f"  {beam_tip_displacement(C_eb, L, f)[0] / row[3]:.4f}")
    print("\nIEA 15 MW (shell / thin-wall / published):")
    for r in rows:
        v = {k: at_centres(np.asarray(r[f"{k}_K6"])) for k in ("shell", "thin", "pub")}
        line = f"  r = {r['r']:5.1f} m"
        for key in ("GA_y", "GA_x", "GJ", "EA"):
            line += f" | {key} {v['shell'][key]:.3e} {v['thin'][key] / v['shell'][key] - 1:+.1%} {v['pub'][key] / v['shell'][key] - 1:+.1%}"
        line += f" | sc/c {v['shell']['sc'][0] / r['chord']:+.4f} {v['thin']['sc'][0] / r['chord']:+.4f} {v['pub']['sc'][0] / r['chord']:+.4f}"
        if r["shell_EGamma"]:
            line += f" | EGamma {r['shell_EGamma']:.3e} thin {r['thin_EGamma'] / r['shell_EGamma'] - 1:+.1%}"
        print(line)


def main():
    if "--reuse" in sys.argv and RESULTS.exists():
        data = json.loads(RESULTS.read_text())
    else:
        lib = create_default_library()
        data = {"box": solve_box(lib), "iea": solve_iea()}
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text(json.dumps(data, indent=1))
    report(data["box"], data["iea"])
    figure_box(data["box"])
    figure_deflection(data["box"])
    figure_warping(data["box"])
    figure_iea(data["iea"])


if __name__ == "__main__":
    main()
