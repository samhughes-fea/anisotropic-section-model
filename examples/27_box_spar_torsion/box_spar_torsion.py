"""Torsion of a composite box spar: Saint-Venant, restrained warping and distortion.

A carbon/glass box spar (the load-carrying box of a blade or wing) is clamped at the
root and twisted at the tip. The torsion theories are compared layer by layer with a
shell model of the same box (curved-free flat Quad8, FSDT):

1. Saint-Venant: free warping, twist rate T / GJ (Bredt-Batho for the closed box).
2. Restrained warping: the clamped root stops the section warping, which creates a
   bimoment and axial warping stresses. Vlasov's theory (no shear strain from warping)
   and Benscoter's (warping amplitude of its own, with shear deformation) are solved in
   closed form; the GBT beam with the section rigid in its plane reproduces Benscoter.
3. Distortion (GBT): the box section can also distort. Near the clamp the distortion
   relieves the warping restraint, and a torque introduced as forces on the webs
   rather than through a rib mostly distorts the box. Ribs stop it.

Figures: the section (shear flows, warping function, distortional modes); the twist,
bimoment and root warping stress under a tip torque through a tip rib; the
distortional response to a web force couple against the number of GBT modes and the
rib spacing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import DIVERGING, REFERENCE, SERIES, TEXT_SECONDARY, plt, save
from _shared.torsion import (add_rib, box_spar, contour_coordinate, loop_order, section_nodes,
                             shell_bimoment, shell_section_motion)
from laminate_modelling_package import (BoundaryConditions, GeneralizedBeam, Laminate, LoadCase,
                                        ResultsExtractor, RestrainedTorsionBeam, StructuralGlobalSolver,
                                        ThinWalledSection, create_default_library)

WIDTH, HEIGHT, LENGTH = 0.5, 0.25, 4.0
N_WIDTH, N_HEIGHT, N_SPAN = 6, 3, 32
TORQUE = 1.0e3  # N m
RIB_SPACINGS = [None, 2.0, 1.0, 0.5]


def laminates(lib):
    carbon, glass = lib.get("T300/5208"), lib.get("E-glass/epoxy")
    cap = Laminate.from_angles(carbon, [0, 0, 45, -45, -45, 45, 0, 0], 0.5e-3, "cap")
    web = Laminate.from_angles(glass, [45, -45, -45, 45], 0.5e-3, "web")
    return {"cap_top": cap, "cap_bottom": cap, "web_le": web, "web_te": web}


def shell_case(box, section, ribs, load):
    """Solve the shell box with the root clamped, ribs at ``ribs`` and a tip load."""
    mesh = box.mesh.copy()
    bcs = BoundaryConditions().clamp(box.nodes("root"))
    case = LoadCase("tip")
    refs = {z: add_rib(mesh, bcs, z, section.shear_centre) for z in ribs}
    if load == "rib":
        case.add_nodal_load(refs[LENGTH], 5, TORQUE)
    else:  # vertical force couple spread over the two webs at the tip, no rib
        F = TORQUE / WIDTH
        for sign, x in ((1.0, 0.5 * WIDTH), (-1.0, -0.5 * WIDTH)):
            nodes = [n for n in section_nodes(mesh, LENGTH) if abs(mesh.node_coordinates(n)[0] - x) < 1e-9]
            case.add_line_load(nodes, (0.0, sign * F / HEIGHT, 0.0))
    sol = StructuralGlobalSolver().solve(mesh, bcs, case)
    return mesh, case, sol


def gbt_case(section, ribs, load, warping="full", distortional=None):
    z_nodes = GeneralizedBeam.graded_nodes(LENGTH, 32, ribs)
    beam = GeneralizedBeam(section, LENGTH, warping=warping, distortional_modes=distortional, z_nodes=z_nodes).clamp(0.0)
    for z in ribs:
        beam.diaphragm(z)
    if load == "rib":
        beam.add_torque(LENGTH, TORQUE)
    else:
        F = TORQUE / WIDTH
        beam.add_wall_load(LENGTH, beam.walls_named("web_te"), (0.0, F))
        beam.add_wall_load(LENGTH, beam.walls_named("web_le"), (0.0, -F))
    return beam.solve()


def ribs_at(spacing):
    if spacing is None:
        return []
    return [float(z) for z in np.arange(spacing, LENGTH + 1e-9, spacing)]


def main():
    lib = create_default_library()
    box = box_spar(WIDTH, HEIGHT, LENGTH, laminates(lib), N_WIDTH, N_HEIGHT, N_SPAN)
    sec = ThinWalledSection.from_blade_mesh(box, z=LENGTH)
    zs = box.span_positions
    sections = {float(z): ThinWalledSection.from_blade_mesh(box, z=float(z)) for z in zs[1:]}
    p = sec.summary()
    print(f"Saint-Venant GJ {sec.GJ:.4g} N m2 (Bredt {sec.J_membrane:.4g}, wall twisting {sec.GJ_walls:.3g})")
    print(f"EGamma {sec.EGamma:.4g} N m4, Ip {sec.Ip:.4g}, Benscoter mu {sec.benscoter_mu:.3f}, "
          f"warping decay length {sec.decay_length:.3f} m, shear centre ({p['shear_centre_x']:.2e}, {p['shear_centre_y']:.2e})")

    # ---------------------------------------------------------------- case A: torque through a tip rib
    beam = RestrainedTorsionBeam.from_section(sec, LENGTH)
    vlasov = RestrainedTorsionBeam.from_section(sec, LENGTH, vlasov=True)
    z_fine = np.linspace(0.0, LENGTH, 401)
    sv = beam.saint_venant(TORQUE, z_fine)
    ben = beam.cantilever_tip_torque(TORQUE, z_fine)
    vla = vlasov.cantilever_tip_torque(TORQUE, z_fine)
    g_rigid = gbt_case(sec, [LENGTH], "rib", "classical", 0)
    g_full = gbt_case(sec, [LENGTH], "rib", "full", None)
    mesh, case, sol = shell_case(box, sec, [LENGTH], "rib")
    shell_twist = np.array([0.0] + [shell_section_motion(sol, mesh, sections[float(z)], float(z)).twist for z in zs[1:]])
    res = ResultsExtractor(sol, case)
    rows = list(zip(zs[:-1], zs[1:]))
    zb, shell_B = shell_bimoment(res, mesh, sec, rows)
    tip = {"Saint-Venant": sv.twist[-1], "Vlasov": vla.twist[-1], "Benscoter": ben.twist[-1],
           "GBT rigid section": g_rigid.mean_twist([LENGTH])[0], "GBT + distortion": g_full.mean_twist([LENGTH])[0],
           "shell": shell_twist[-1]}
    print("Tip twist under a tip torque through a tip rib [mrad]:")
    for k, v in tip.items():
        print(f"  {k:<18} {1e3 * v:8.4f}  ({v / tip['shell'] - 1:+.2%} on the shell)")
    print(f"Root bimoment [N m2]: Vlasov {vla.bimoment[0]:.2f}, Benscoter {ben.bimoment[0]:.2f}, "
          f"GBT rigid {g_rigid.bimoment([0.0])[0]:.2f}, GBT + distortion {g_full.bimoment([0.0])[0]:.2f}; "
          f"shell first element row {shell_B[0]:.2f}, GBT there {g_full.bimoment([zb[0]])[0]:.2f}")

    # ---------------------------------------------------------------- case B: web force couple, ribs
    couple = {}
    for spacing in RIB_SPACINGS:
        ribs = ribs_at(spacing)
        ribs_shell = [z for z in ribs if z < LENGTH - 1e-9]
        mesh_b, _, sol_b = shell_case(box, sec, ribs_shell, "couple")
        shell_tw = np.array([0.0] + [shell_section_motion(sol_b, mesh_b, sections[float(z)], float(z)).twist for z in zs[1:]])
        gbt = {n: gbt_case(sec, ribs_shell, "couple", "full", n).mean_twist(zs) for n in (0, 1, 2, None)}
        couple[spacing] = (shell_tw, gbt)
        label = "no ribs" if spacing is None else f"ribs every {spacing:g} m"
        print(f"Web couple, {label}: tip twist shell {1e3 * shell_tw[-1]:.4f} mrad; GBT " + ", ".join(
            f"{'all' if n is None else n} modes {1e3 * v[-1]:.4f}" for n, v in gbt.items()))
    # distorted tip sections, no ribs
    _, _, sol_c = shell_case(box, sec, [], "couple")
    g_c = gbt_case(sec, [], "couple", "full", None)

    # ================================================================ figure 1: the section
    modes = g_full.beam.modes
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    ax = axes[0]
    for k, (a, b) in enumerate(sec.contour()):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=SERIES[1] if sec.walls[k].name.startswith("cap") else SERIES[2],
                lw=3 if sec.walls[k].name.startswith("cap") else 1.8, solid_capstyle="butt")
    for k in range(0, len(sec.walls), 2):
        m = 0.5 * (sec.points[sec.start[k]] + sec.points[sec.end[k]])
        t = sec.tangent[k] * np.sign(sec.shear_flow[k]) * 0.05
        ax.annotate("", m + t, m - t, arrowprops=dict(arrowstyle="->", color=TEXT_SECONDARY, lw=1.0))
    ax.plot(*sec.shear_centre, "o", color=SERIES[3], mfc="none", mew=1.8, label="shear centre")
    ax.plot([], [], color=SERIES[1], lw=3, label="carbon caps")
    ax.plot([], [], color=SERIES[2], lw=1.8, label="glass webs")
    ax.set_aspect("equal")
    ax.set_title("Section and Saint-Venant shear flow", loc="left")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper center", fontsize=7.5, ncol=3, bbox_to_anchor=(0.5, -0.25))
    ax = axes[1]
    om = sec.omega
    vmax = np.max(np.abs(om))
    for k in range(len(sec.walls)):
        i, j = sec.start[k], sec.end[k]
        s = np.linspace(0, 1, 6)
        pts = sec.points[i] + s[:, None] * (sec.points[j] - sec.points[i])
        vals = (1 - s) * om[i] + s * om[j]
        ax.scatter(pts[:, 0], pts[:, 1], c=vals, cmap=DIVERGING, vmin=-vmax, vmax=vmax, s=14, lw=0)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-vmax * 1e3, vmax * 1e3))
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, label="ω [10⁻³ m²]")
    ax.set_aspect("equal")
    ax.set_title("Warping function ω (Benscoter)", loc="left")
    ax.set_xlabel("x [m]")
    for idx, ax in zip((3, 4), axes[2:]):
        shape = modes.in_plane_shape(idx)
        base = modes.wall_points()
        scale = 0.08
        for p0, du in zip(base, shape):
            ax.plot(p0[:, 0], p0[:, 1], color=TEXT_SECONDARY, lw=0.8)
            ax.plot(p0[:, 0] + scale * du[:, 0], p0[:, 1] + scale * du[:, 1], color=SERIES[0], lw=2)
        ax.set_aspect("equal")
        ax.set_title(f"Distortional mode {idx - 2}", loc="left")
        ax.set_xlabel("x [m]")
        ax.text(0.02, 0.02, f"stiffness {modes.all_distortional_stiffness[idx - 3]:.3g}", transform=ax.transAxes,
                fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "box_spar_section.png")

    # ================================================================ figure 2: tip torque through a rib
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.plot(z_fine, 1e3 * (sv.twist - sv.twist), color=REFERENCE, ls="--", lw=1.2, label="Saint-Venant (reference line)")
    ref = sv.twist
    zs_f = np.asarray(zs, float)
    ax.plot(z_fine, 1e3 * (vla.twist - ref), color=SERIES[3], label="Vlasov")
    ax.plot(z_fine, 1e3 * (ben.twist - ref), color=SERIES[1], label="Benscoter")
    ax.plot(zs_f, 1e3 * (g_rigid.mean_twist(zs_f) - TORQUE * zs_f / sec.GJ), "s", color=SERIES[1], mfc="none",
            ms=5, label="GBT, rigid section")
    ax.plot(zs_f, 1e3 * (g_full.mean_twist(zs_f) - TORQUE * zs_f / sec.GJ), color=SERIES[0], label="GBT + distortion")
    ax.plot(zs_f, 1e3 * (shell_twist - TORQUE * zs_f / sec.GJ), "o", color=REFERENCE, mfc="none", ms=5, label="shell")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("twist − Saint-Venant twist [mrad]")
    ax.set_title("Twist lost to warping restraint", loc="left")
    ax.legend(fontsize=7.5)
    ax = axes[1]
    zz = np.linspace(0.0, 0.8, 161)
    ax.plot(z_fine, vla.bimoment, color=SERIES[3], label="Vlasov")
    ax.plot(z_fine, ben.bimoment, color=SERIES[1], label="Benscoter")
    ax.plot(zz, g_rigid.bimoment(zz), "--", color=SERIES[1], lw=1.2, label="GBT, rigid section")
    ax.plot(zz, g_full.bimoment(zz), color=SERIES[0], label="GBT + distortion")
    ax.plot(zb, shell_B, "o", color=REFERENCE, mfc="none", ms=5, label="shell (element rows)")
    ax.set_xlim(0.0, 0.8)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("bimoment B [N m²]")
    ax.set_title("Bimoment near the clamped root", loc="left")
    ax.legend(fontsize=7.5)
    ax = axes[2]
    order, s0 = loop_order(sec, range(len(sec.walls)))
    g = res.gauss_points()
    zg = np.unique(np.round(g.xyz[g.xyz[:, 2] < zs[1], 2], 12))
    z_row = float(zg[np.argmin(np.abs(zg - 0.5 * (zs[0] + zs[1])))])  # a Gauss line in the first element row
    sel = np.abs(g.xyz[:, 2] - z_row) < 1e-9
    s_shell = contour_coordinate(sec, order, s0, g.xyz[sel, :2])
    ax.plot(s_shell, g.N[sel, 0] / 1e3, "o", color=REFERENCE, mfc="none", ms=4, label="shell")
    for sol_g, colour, ls, label in ((g_rigid, SERIES[1], "--", "GBT, rigid section"), (g_full, SERIES[0], "-", "GBT + distortion")):
        wr = sol_g.wall_resultants(z_row)
        ss, nn = [], []
        pts_by_wall = {}
        for p in sol_g.beam.modes.points_:
            pts_by_wall.setdefault(p.wall, []).append(p.s_fraction)
        for (k, fwd), s_start in zip(order, s0[:-1]):
            frac = np.array(pts_by_wall[k])
            vals = wr[k][:, 0]
            sk = s_start + (frac if fwd else 1 - frac) * sec.length[k]
            ss.extend(sk)
            nn.extend(vals)
        o = np.argsort(ss)
        ax.plot(np.array(ss)[o], np.array(nn)[o] / 1e3, ls, color=colour, label=label)
    ax.set_xlabel("contour position s [m]")
    ax.set_ylabel("N_zz [kN/m]")
    ax.set_title(f"Warping stress resultant at z = {z_row:.3f} m", loc="left")
    ax.legend(fontsize=7.5)
    fig.text(0.0, -0.04, f"Box {WIDTH:g} x {HEIGHT:g} m, {LENGTH:g} m long, root clamped, {TORQUE:g} N m at the tip through a tip rib. "
             "Caps [0₂/±45/∓45/0₂] T300/5208 (4 mm), webs [±45]s E-glass (2 mm). Shell: flat Quad8 FSDT, "
             f"{N_SPAN} elements along, {N_WIDTH} across each cap, {N_HEIGHT} up each web.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "box_spar_restrained_warping.png")

    # ================================================================ figure 3: web force couple
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]})
    ax = axes[0]
    shell_tw, gbt = couple[None]
    for (n, tw), colour in zip(gbt.items(), (SERIES[1], SERIES[2], SERIES[3], SERIES[0])):
        ax.plot(zs, 1e3 * tw, color=colour, label="GBT, rigid section" if n == 0 else
                f"GBT, {'all' if n is None else n} distortional mode{'s' if n != 1 else ''}")
    ax.plot(zs, 1e3 * shell_tw, "o", color=REFERENCE, mfc="none", ms=5, label="shell")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("twist [mrad]")
    ax.set_title("Torque as a force couple on the webs", loc="left")
    ax.legend(fontsize=7.5)
    ax = axes[1]
    labels = ["none" if s is None else f"{s:g} m" for s in RIB_SPACINGS]
    xs = np.arange(len(RIB_SPACINGS))
    rib_ref = tip["shell"]
    ax.bar(xs - 0.2, [couple[s][0][-1] / rib_ref for s in RIB_SPACINGS], 0.4, color=REFERENCE, alpha=0.75, label="shell")
    ax.bar(xs + 0.2, [couple[s][1][None][-1] / rib_ref for s in RIB_SPACINGS], 0.4, color=SERIES[0], label="GBT, all modes")
    ax.axhline(1.0, color=TEXT_SECONDARY, lw=0.8, ls="--")
    ax.set_xticks(xs, labels)
    ax.set_xlabel("rib spacing")
    ax.set_ylabel("tip twist / twist with the torque through a rib")
    ax.set_yscale("log")
    ax.set_title("Ribs stop the distortion", loc="left")
    ax.legend(fontsize=7.5)
    ax = axes[2]
    tip_sec = sections[float(LENGTH)]
    d_shell = shell_section_motion(sol_c, box.mesh, tip_sec, LENGTH).displacement
    d_gbt = g_c.section_displacement(LENGTH)
    scale = 0.1 / np.max(np.abs(d_gbt[:, :2]))
    for k in range(len(tip_sec.walls)):
        i, j = tip_sec.start[k], tip_sec.end[k]
        P = tip_sec.points[[i, j]]
        ax.plot(P[:, 0], P[:, 1], color=TEXT_SECONDARY, lw=0.8)
        ax.plot(P[:, 0] + scale * d_shell[[i, j], 0], P[:, 1] + scale * d_shell[[i, j], 1], "o-", color=REFERENCE,
                mfc="none", ms=3, lw=1.0, label="shell" if k == 0 else None)
    for p0, du in zip(g_c.beam.modes.wall_points(), g_c.beam.modes._contour(
            g_c.beam.modes.in_plane @ g_c.amplitudes(LENGTH)[1], 8)):
        ax.plot(p0[:, 0] + scale * du[:, 0], p0[:, 1] + scale * du[:, 1], color=SERIES[0], lw=1.8)
    ax.plot([], [], color=SERIES[0], lw=1.8, label="GBT, all modes")
    ax.set_aspect("equal")
    ax.set_title(f"Tip section, no ribs (x{scale:.0f})", loc="left")
    ax.set_xlabel("x [m]")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.text(0.0, -0.04, f"The same box with the {TORQUE:g} N m tip torque applied as equal and opposite vertical forces spread over the two webs, "
             "with no tip rib, and with ribs (rigid in their plane, free to warp) at the spacing shown.", fontsize=7.5,
             color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "box_spar_distortion.png")


if __name__ == "__main__":
    main()
