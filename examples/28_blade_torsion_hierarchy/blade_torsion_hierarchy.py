"""Torsion of a blade section, layer by layer: Saint-Venant, restrained warping, distortion.

A prismatic blade segment (NACA 0024, 1 m chord, 6 m long: glass skins, carbon spar
caps, two glass shear webs, the section of example 20) is clamped at the root and
twisted at the tip. Each layer of torsion theory adds one physical contribution, and
each is checked against a shell model of the same blade (curved Quad8 FSDT, skins on
their mid-surface):

1. Saint-Venant torsion. Shear flows circulate in the three cells (leading-edge cell,
   spar box, trailing-edge cell) and the section warps freely: GJ from
   ``ThinWalledSection`` against the shell with ribs every 0.5 m (no distortion,
   warping free), and against ``beam_section_properties``.
2. Restrained warping. The clamped root stops the section warping. Benscoter's theory
   gives the bimoment and the axial warping stresses it causes, and the twist lost to
   the restraint; Vlasov's theory (warping tied to the twist rate) overestimates both.
3. Distortion (GBT). The cells are not rigid. Near the clamp the section distorts and
   relieves most of the warping restraint, and a torque introduced as forces on the
   skins (as aerodynamic loads are) rather than through a rib distorts the section.
   GBT adds the distortional modes of the section, softest first; ribs stop them.

Figures: the section (cells and shear flows, warping function, distortional modes);
the Saint-Venant and restrained-warping responses; the distortional response.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import DIVERGING, REFERENCE, SEQUENTIAL, SERIES, TEXT_SECONDARY, plt, save
from _shared.torsion import (add_rib, contour_coordinate, loop_order, section_nodes, shell_bimoment,
                             shell_section_motion)
from laminate_modelling_package import (Airfoil, BladeBuilder, BoundaryConditions, FSDTModel, GeneralizedBeam,
                                        Laminate, LoadCase, ResultsExtractor, RestrainedTorsionBeam, ShearWeb,
                                        SkinRegion, Station, StructuralGlobalSolver, ThinWalledSection,
                                        beam_section_properties, create_default_library)

CHORD, LENGTH, N_SPAN = 1.0, 6.0, 24
REGIONS = [SkinRegion("le_panel", 0.0, 0.18, 4), SkinRegion("spar_cap", 0.18, 0.48, 6),
           SkinRegion("te_panel", 0.48, 1.0, 8)]
WEBS = [ShearWeb("web_fwd", 0.18, 4), ShearWeb("web_aft", 0.48, 4)]
TORQUE = 1.0e3  # N m
RIB_SPACINGS = [None, 3.0, 1.5, 0.75]
N_GBT = 24  # GBT elements along the span
N_SHOWN_MODES = 5  # distortional mode shapes drawn in the section figure


def laminates(lib):
    glass, carbon = lib.get("E-glass/epoxy"), lib.get("T300/5208")
    skin = Laminate.from_angles(glass, [0, 45, -45, -45, 45, 0], 0.5e-3)
    web = Laminate.from_angles(glass, [45, -45, -45, 45], 0.5e-3)
    cap = Laminate.from_angles(carbon, [0] * 16, 0.5e-3)
    return {"le_panel": skin, "te_panel": skin, "web_fwd": web, "web_aft": web, "spar_cap": cap}


def build(lib):
    airfoil = Airfoil.naca4("0024")
    stations = [Station(0.0, CHORD, airfoil, pitch_axis=0.3), Station(LENGTH, CHORD, airfoil, pitch_axis=0.3)]
    return BladeBuilder(stations, REGIONS, WEBS, n_span=N_SPAN, mesh_surface="mid").build(laminates(lib), FSDTModel())


def ribs_at(spacing, tip=False):
    if spacing is None:
        out = []
    else:
        out = [float(z) for z in np.arange(spacing, LENGTH + 1e-9, spacing) if z < LENGTH - 1e-9]
    return out + ([LENGTH] if tip else [])


def skin_couple(section):
    """Vertical force F on the leading-edge panels and -F on the trailing-edge panels: torque F (x_te - x_le)."""
    def centre(prefix):
        ks = [k for k, w in enumerate(section.walls) if w.name.startswith(prefix)]
        mids = 0.5 * (section.points[section.start[ks]] + section.points[section.end[ks]])
        return ks, float(np.sum(mids[:, 0] * section.length[ks]) / np.sum(section.length[ks]))

    le, x_le = centre("le_panel")
    te, x_te = centre("te_panel")
    F = TORQUE / (x_le - x_te)  # +F up on the LE panels, -F on the TE panels gives +T about z
    return le, te, F


def shell_case(blade, section, ribs, load):
    mesh = blade.mesh.copy()
    bcs = BoundaryConditions().clamp(blade.nodes("root"))
    case = LoadCase("tip")
    refs = {z: add_rib(mesh, bcs, z, section.shear_centre) for z in ribs}
    if load == "rib":
        case.add_nodal_load(refs[LENGTH], 5, TORQUE)
    else:
        le, te, F = skin_couple(section)
        for walls, force in ((le, F), (te, -F)):
            length = float(np.sum(section.length[walls]))
            nodes = {section.mesh_nodes[p] for k in walls for p in (section.start[k], section.end[k])}
            case.add_line_load(sorted(nodes), (0.0, force / length, 0.0))
    sol = StructuralGlobalSolver().solve(mesh, bcs, case)
    return mesh, case, sol


def gbt_case(section, ribs, load, warping="full", distortional=None, modes=None, extensible=False):
    z_nodes = GeneralizedBeam.graded_nodes(LENGTH, N_GBT, ribs)
    beam = GeneralizedBeam(section, LENGTH, warping=warping, distortional_modes=distortional, z_nodes=z_nodes,
                           modes=modes, extensible=extensible).clamp(0.0)
    for z in ribs:
        beam.diaphragm(z)
    if load == "rib":
        beam.add_torque(LENGTH, TORQUE)
    else:
        le, te, F = skin_couple(section)
        beam.add_wall_load(LENGTH, le, (0.0, F))
        beam.add_wall_load(LENGTH, te, (0.0, -F))
    return beam.solve()


def main():
    lib = create_default_library()
    blade = build(lib)
    zs = np.asarray(blade.span_positions, float)
    sec = ThinWalledSection.from_blade_mesh(blade, z=LENGTH)
    sections = {float(z): ThinWalledSection.from_blade_mesh(blade, z=float(z)) for z in zs[1:]}
    print(f"{sec}: {sec.n_cells} cells, shear centre x = {sec.shear_centre[0]:.4f} m")
    print(f"Saint-Venant GJ {sec.GJ:.4g} N m2, EGamma {sec.EGamma:.4g} N m4 (secondary {sec.EGamma_secondary:.3g}), "
          f"Benscoter mu {sec.benscoter_mu:.3f}, warping decay length {sec.decay_length:.3f} m")

    # ---------------------------------------------------------------- 1. Saint-Venant
    props = beam_section_properties(blade)
    mesh_r, _, sol_r = shell_case(blade, sec, ribs_at(0.5, tip=True), "rib")
    tw_r = np.array([0.0] + [shell_section_motion(sol_r, mesh_r, sections[float(z)], float(z)).twist for z in zs[1:]])
    mid = (zs >= 2.0) & (zs <= 4.0)
    rate = np.polyfit(zs[mid], tw_r[mid], 1)[0]
    gj = {"thin-wall Saint-Venant": sec.GJ, "shell, ribs every 0.5 m": TORQUE / rate,
          "beam_section_properties (no ribs)": props.GJ}
    print("GJ [N m2]: " + ", ".join(f"{k} {v:.4g} ({v / sec.GJ - 1:+.2%})" for k, v in gj.items()))

    # ---------------------------------------------------------------- 2. restrained warping, tip rib only
    beam = RestrainedTorsionBeam.from_section(sec, LENGTH)
    vlasov = RestrainedTorsionBeam.from_section(sec, LENGTH, vlasov=True)
    z_fine = np.linspace(0.0, LENGTH, 601)
    sv, ben, vla = beam.saint_venant(TORQUE, z_fine), beam.cantilever_tip_torque(TORQUE, z_fine), \
        vlasov.cantilever_tip_torque(TORQUE, z_fine)
    modes_all = GeneralizedBeam(sec, LENGTH, 2, "full", None).modes
    g_rigid = gbt_case(sec, [LENGTH], "rib", "classical", 0)
    g_full = gbt_case(sec, [LENGTH], "rib", "full", None, modes_all)
    mesh, case, sol = shell_case(blade, sec, [LENGTH], "rib")
    shell_tw = np.array([0.0] + [shell_section_motion(sol, mesh, sections[float(z)], float(z)).twist for z in zs[1:]])
    res = ResultsExtractor(sol, case)
    zb, shell_B = shell_bimoment(res, mesh, sec, list(zip(zs[:-1], zs[1:])))
    print("Tip torque through a tip rib, tip twist [mrad]: " + ", ".join(
        f"{k} {1e3 * v:.4f}" for k, v in (("Saint-Venant", sv.twist[-1]), ("Vlasov", vla.twist[-1]),
                                          ("Benscoter", ben.twist[-1]), ("GBT rigid", g_rigid.mean_twist([LENGTH])[0]),
                                          ("GBT + distortion", g_full.mean_twist([LENGTH])[0]), ("shell", shell_tw[-1]))))
    print(f"Root bimoment [N m2]: Vlasov {vla.bimoment[0]:.3g}, Benscoter {ben.bimoment[0]:.3g}, "
          f"GBT rigid {g_rigid.bimoment([0.0])[0]:.3g}, GBT + distortion {g_full.bimoment([0.0])[0]:.3g}; "
          f"first element row: shell {shell_B[0]:.3g}, GBT + distortion {g_full.bimoment([zb[0]])[0]:.3g}")

    # ---------------------------------------------------------------- 3. distortion: skin force couple
    # GBT levels: 0 = rigid section, n = n distortional modes, None = all, "ext" = all + wall stretching
    couple = {}
    mode_counts = (0, 1, 2, 3, 4, 5, 8, None, "ext")
    modes_ext = GeneralizedBeam(sec, LENGTH, 2, "full", None, extensible=True).modes

    def level(n, ribs):
        if n == "ext":
            return gbt_case(sec, ribs, "couple", "full", None, modes_ext)
        return gbt_case(sec, ribs, "couple", "full", n, None if n is not None else modes_all)

    def name(n):
        return {0: "rigid section", None: "all modes", "ext": "all modes + wall stretching"}.get(n, f"{n} modes")

    for spacing in RIB_SPACINGS:
        ribs = ribs_at(spacing)
        m_c, _, s_c = shell_case(blade, sec, ribs, "couple")
        tw = np.array([0.0] + [shell_section_motion(s_c, m_c, sections[float(z)], float(z)).twist for z in zs[1:]])
        counts = mode_counts if spacing is None else (0, None, "ext")
        gbt = {n: level(n, ribs) for n in counts}
        couple[spacing] = (tw, gbt, s_c)
        label = "no ribs" if spacing is None else f"ribs every {spacing:g} m"
        print(f"Skin force couple, {label}: twist [mrad] at z = 5 m / tip: shell {1e3 * tw[-5]:.3f} / {1e3 * tw[-1]:.3f}; "
              + "; ".join(f"GBT {name(n)} {1e3 * g.mean_twist([zs[-5]])[0]:.3f} / {1e3 * g.mean_twist([LENGTH])[0]:.3f}"
                          for n, g in gbt.items()))

    # ================================================================ figure 1: the section
    fig = plt.figure(figsize=(15, 4.6))
    upper, lower = fig.subfigures(2, 1, height_ratios=[1.35, 0.65])
    top = list(upper.subplots(1, 3, gridspec_kw={"wspace": 0.5}))
    mode_axes = list(lower.subplots(1, N_SHOWN_MODES, sharex=True, sharey=True))
    ax = top[0]
    qmax = np.max(np.abs(sec.shear_flow))
    for k in range(len(sec.walls)):
        a, b = sec.points[sec.start[k]], sec.points[sec.end[k]]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=SEQUENTIAL(0.25 + 0.75 * abs(sec.shear_flow[k]) / qmax),
                lw=3.0 if sec.walls[k].name.startswith("spar") else 2.0, solid_capstyle="butt")
    for k in range(1, len(sec.walls), 4):
        m = 0.5 * (sec.points[sec.start[k]] + sec.points[sec.end[k]])
        t = sec.tangent[k] * np.sign(sec.shear_flow[k]) * 0.035
        ax.annotate("", m + t, m - t, arrowprops=dict(arrowstyle="->", color=TEXT_SECONDARY, lw=0.9))
    ax.plot(*sec.shear_centre, "o", color=SERIES[3], mfc="none", mew=1.8, label="shear centre")
    ax.set_ylim(-0.2, 0.2)
    sm = plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=plt.Normalize(0, TORQUE * qmax / sec.J_membrane / 1e3))
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="|q| per kN m [kN/m]")
    ax.set_aspect("equal")
    ax.set_title("Saint-Venant shear flow in three cells", loc="left")
    ax.legend(loc="upper right", fontsize=7.5, ncol=1)
    ax = top[1]
    om = sec.omega
    vmax = np.max(np.abs(om))
    for k in range(len(sec.walls)):
        i, j = sec.start[k], sec.end[k]
        s = np.linspace(0, 1, 5)
        pts = sec.points[i] + s[:, None] * (sec.points[j] - sec.points[i])
        ax.scatter(pts[:, 0], pts[:, 1], c=(1 - s) * om[i] + s * om[j], cmap=DIVERGING, vmin=-vmax, vmax=vmax, s=10, lw=0)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-vmax * 1e3, vmax * 1e3))
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="ω [10⁻³ m²]")
    ax.set_aspect("equal")
    ax.set_title("Warping function ω", loc="left")
    ax = top[2]
    k_d = modes_all.all_distortional_stiffness
    ax.semilogy(np.arange(1, len(k_d) + 1), k_d, ".", color=SERIES[0], ms=4)
    ax.semilogy(np.arange(1, N_SHOWN_MODES + 1), k_d[:N_SHOWN_MODES], "o", color=SERIES[0], mfc="none", ms=7,
                label=f"modes 1 to {N_SHOWN_MODES}, shown below")
    ax.legend(fontsize=7.5)
    ax.set_xlabel("distortional mode")
    ax.set_ylabel("transverse bending stiffness")
    ax.set_title(f"{len(k_d)} distortional modes, softest first", loc="left")
    base = modes_all.wall_points()
    for c, ax in enumerate(mode_axes):
        shape = modes_all.in_plane_shape(3 + c)
        for p0, du in zip(base, shape):
            ax.plot(p0[:, 0], p0[:, 1], color=TEXT_SECONDARY, lw=0.7)
            ax.plot(p0[:, 0] + 0.06 * du[:, 0], p0[:, 1] + 0.06 * du[:, 1], color=SERIES[0], lw=1.8)
        ax.set_aspect("equal")
        ax.set_title(f"Distortional mode {c + 1}", loc="left")
        ax.set_xlabel("x [m]")
    save(fig, __file__, "blade_torsion_section.png")

    # ================================================================ figure 2: Saint-Venant and bimoment
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    ax.plot(z_fine, 1e3 * sv.twist, color=REFERENCE, ls="--", lw=1.2, label=f"Saint-Venant, GJ = {sec.GJ / 1e6:.3f} MN m²")
    ax.plot(zs, 1e3 * tw_r, "o", color=REFERENCE, mfc="none", ms=5, label="shell, ribs every 0.5 m")
    ax.plot(zs, 1e3 * shell_tw, "s", color=SERIES[0], mfc="none", ms=5, label="shell, tip rib only")
    ax.plot(zs, 1e3 * g_full.mean_twist(zs), color=SERIES[0], lw=1.4, label="GBT, tip rib only")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("twist [mrad]")
    ax.set_title("1 · Saint-Venant torsion", loc="left")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.text(0.98, 0.04, "GJ / thin-wall GJ:\n" + "\n".join(f"{k}: {v / sec.GJ:.4f}" for k, v in gj.items()),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color=TEXT_SECONDARY)
    ax = axes[1]
    zz = np.linspace(0.0, 1.0, 201)
    ax.plot(z_fine, vla.bimoment, color=SERIES[3], label="Vlasov")
    ax.plot(z_fine, ben.bimoment, color=SERIES[1], label="Benscoter")
    ax.plot(zz, g_rigid.bimoment(zz), "--", color=SERIES[1], lw=1.2, label="GBT, rigid section")
    ax.plot(zz, g_full.bimoment(zz), color=SERIES[0], label="GBT + distortion")
    ax.plot(zb, shell_B, "o", color=REFERENCE, mfc="none", ms=5, label="shell (element rows)")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("bimoment B [N m²]")
    ax.set_title("2 · Restrained warping: the bimoment", loc="left")
    ax.legend(fontsize=7.5)
    ax = axes[2]
    skin = [k for k, w in enumerate(sec.walls) if not w.name.startswith("web")]
    order, s0 = loop_order(sec, skin)
    g = res.gauss_points()
    zg = np.unique(np.round(g.xyz[g.xyz[:, 2] < zs[1], 2], 12))
    z_row = float(zg[np.argmin(np.abs(zg - 0.5 * zs[1]))])
    sel = (np.abs(g.xyz[:, 2] - z_row) < 1e-9) & np.isin(g.element_ids, blade.elements("skin"))
    s_shell = contour_coordinate(sec, order, s0, g.xyz[sel, :2])
    ax.plot(s_shell, g.N[sel, 0] / 1e3, "o", color=REFERENCE, mfc="none", ms=3.5, label="shell")
    fr = {}
    for p in g_full.beam.modes.points_:
        fr.setdefault(p.wall, []).append(p.s_fraction)
    for sol_g, colour, ls, label in ((g_rigid, SERIES[1], "--", "GBT, rigid section"),
                                     (g_full, SERIES[0], "-", "GBT + distortion")):
        wr = sol_g.wall_resultants(z_row)
        ss, nn = [], []
        for (k, fwd), s_start in zip(order, s0[:-1]):
            f = np.array(fr[k])
            ss.extend(s_start + (f if fwd else 1 - f) * sec.length[k])
            nn.extend(wr[k][:, 0])
        o = np.argsort(ss)
        ax.plot(np.array(ss)[o], np.array(nn)[o] / 1e3, ls, color=colour, lw=1.4, label=label)
    ax.set_xlabel("position round the skin from the trailing edge [m]")
    ax.set_ylabel("N_zz [kN/m]")
    ax.set_title(f"Warping stress resultant, z = {z_row:.3f} m", loc="left")
    ax.legend(fontsize=7.5)
    fig.text(0.0, -0.04, f"NACA 0024, chord {CHORD:g} m, {LENGTH:g} m segment, root clamped, {TORQUE:g} N m at the tip through a tip rib "
             "(rigid in its plane, free to warp). Skins [0/±45]s E-glass 3 mm, caps 16 x 0.5 mm T300/5208 (18–48% chord), "
             f"webs [±45]s E-glass. Shell: curved Quad8 FSDT on the mid-surface, {N_SPAN} elements along.",
             fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "blade_torsion_warping.png")

    # ================================================================ figure 3: distortion
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1.15, 1.0, 1.25]})
    ax = axes[0]
    tw, gbt, s_c = couple[None]
    for n, gs in gbt.items():
        colour = {0: SERIES[1], None: SERIES[0], "ext": SERIES[7]}.get(n) or plt.cm.Greens(0.3 + 0.7 * (n - 1) / 7)
        ax.plot(zs, 1e3 * gs.mean_twist(zs), color=colour, lw=1.6, ls="--" if n == "ext" else "-",
                label=f"GBT, {name(n)}" if n in (0, None, "ext") else f"GBT, {n} distortional mode{'s' if n > 1 else ''}")
    ax.plot(zs, 1e3 * tw, "o", color=REFERENCE, mfc="none", ms=5, label="shell")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("twist [mrad]")
    ax.set_title("3 · Distortion: torque from forces on the skins", loc="left")
    ax.legend(fontsize=7.2)
    ax = axes[1]
    for spacing, colour in zip(RIB_SPACINGS, (SERIES[0], SERIES[2], SERIES[3], SERIES[4])):
        tw_s, gbt_s, _ = couple[spacing]
        label = "no ribs" if spacing is None else f"ribs every {spacing:g} m"
        ax.plot(zs, 1e3 * gbt_s["ext"].mean_twist(zs), color=colour, lw=1.6, label=f"GBT, {label}")
        ax.plot(zs, 1e3 * tw_s, "o", color=colour, mfc="none", ms=4)
        for z in ribs_at(spacing):
            ax.axvline(z, color=colour, lw=0.6, ls=":")
    ax.plot([], [], "o", color=REFERENCE, mfc="none", ms=4, label="shell")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("twist [mrad]")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-3.0, 200.0)
    ax.set_title("Ribs confine it to the bay with the load", loc="left")
    ax.legend(fontsize=7.2, loc="upper left")
    ax = axes[2]
    tip_sec = sections[float(LENGTH)]
    d_shell = shell_section_motion(s_c, blade.mesh, tip_sec, LENGTH).displacement
    g_all = gbt["ext"]
    d_gbt = g_all.section_displacement(LENGTH)
    scale = 0.08 / np.max(np.abs(d_gbt[:, :2]))
    for k in range(len(tip_sec.walls)):
        i, j = tip_sec.start[k], tip_sec.end[k]
        P = tip_sec.points[[i, j]]
        ax.plot(P[:, 0], P[:, 1], color=TEXT_SECONDARY, lw=0.7)
        ax.plot(P[:, 0] + scale * d_shell[[i, j], 0], P[:, 1] + scale * d_shell[[i, j], 1], "o-", color=REFERENCE,
                mfc="none", ms=2.5, lw=0.9, label="shell" if k == 0 else None)
    m = g_all.beam.modes
    for p0, du in zip(m.wall_points(), m._contour(m.in_plane @ g_all.amplitudes(LENGTH)[1], 8)):
        ax.plot(p0[:, 0] + scale * du[:, 0], p0[:, 1] + scale * du[:, 1], color=SERIES[0], lw=1.6)
    ax.plot([], [], color=SERIES[0], lw=1.6, label="GBT, + wall stretching")
    ax.set_aspect("equal")
    ax.set_title(f"Tip section, no ribs (displacements x{scale:.0f})", loc="left")
    ax.set_xlabel("x [m]")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.text(0.0, -0.04, f"The same blade with the {TORQUE:g} N m tip torque applied as equal and opposite vertical forces spread over the "
             "leading-edge and trailing-edge panels (both surfaces), with no tip rib, and with ribs at the spacing shown.",
             fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "blade_torsion_distortion.png")


if __name__ == "__main__":
    main()
