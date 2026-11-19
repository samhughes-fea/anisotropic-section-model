"""Isotropic sections against the closed forms: thin-wall theory and the solid section.

The starting point of the suite. Before laminates and couplings, both section engines
are checked where the answer is known exactly.

1. Thin-wall theory (``ThinWalledSection``) on five aluminium sections with textbook
   solutions: a closed circular tube (Bredt-Batho), the same tube slit along its length
   (open section: torsion only from the wall twisting, shear centre 2R from the centre,
   Vlasov warping constant), a doubly symmetric I-beam, a channel (shear centre off the
   web) and an unsymmetric two-cell box (Bredt-Batho for several cells).
2. The tube's polygon converging to the circle, at second order in the wall size.
3. The solid section (``SolidSectionAnalysis``) on rectangles from square to 10 : 1, against
   Saint-Venant's series for the torsion constant and the 5/6 shear factor, and its
   convergence with mesh size for Q4, Q8 and T6 elements.
4. Where thin-wall theory stops: a tube of growing wall thickness, thin-wall and solid
   against the exact thick-tube stiffness.

Figures: the five sections with their warping and shear centres; the convergence of the
polygon and of the solid mesh; the thick tube.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

from _shared.checks import Checks  # noqa: E402
from _shared.plotting import DIVERGING, REFERENCE, SERIES, TEXT_SECONDARY, plt, save  # noqa: E402
from anisotropic_section_model import ThinWalledSection  # noqa: E402
from anisotropic_section_model.solid import SolidMaterial, SolidSectionAnalysis  # noqa: E402
from anisotropic_section_model.solid.structured import arc_patch, rectangle_patch, structured_mesh  # noqa: E402
from laminate_constitutive_model import Laminate, create_default_library  # noqa: E402

DATA = HERE / "data" / "results.json"

T = 2.0e-3            # wall thickness of the thin-walled sections [m]
R = 0.1               # tube mid-line radius
B, H = 0.1, 0.2       # I-beam and channel: flange width and mid-line height
CELL_A, CELL_B, CELL_H = 0.3, 0.15, 0.12   # two-cell box: cell widths and height
WEB_T = 1.0e-3        # two-cell box: middle web thickness
SLIT = 1.0e-4         # slit tube: the gap angle [rad]


# ----------------------------------------------------------------------------- sections
def segments(a, b, n, lam, name=""):
    a, b = np.asarray(a, float), np.asarray(b, float)
    p = [a + (b - a) * k / n for k in range(n + 1)]
    return [(p[k], p[k + 1], lam, name) for k in range(n)]


def arc(n, lam, theta0=0.0, theta1=2 * np.pi, radius=R):
    th = np.linspace(theta0, theta1, n + 1)
    p = np.c_[radius * np.cos(th), radius * np.sin(th)]
    return [(p[k], p[k + 1], lam) for k in range(n)]


def thin_sections(lam, web):
    return {
        "tube": ThinWalledSection.from_segments(arc(256, lam)),
        "slit tube": ThinWalledSection.from_segments(arc(256, lam, SLIT / 2, 2 * np.pi - SLIT / 2)),
        "I-beam": ThinWalledSection.from_segments(
            segments((-B / 2, H / 2), (0, H / 2), 8, lam) + segments((0, H / 2), (B / 2, H / 2), 8, lam)
            + segments((-B / 2, -H / 2), (0, -H / 2), 8, lam) + segments((0, -H / 2), (B / 2, -H / 2), 8, lam)
            + segments((0, -H / 2), (0, H / 2), 16, lam)),
        "channel": ThinWalledSection.from_segments(
            segments((B, H / 2), (0, H / 2), 10, lam) + segments((0, H / 2), (0, -H / 2), 20, lam)
            + segments((0, -H / 2), (B, -H / 2), 10, lam)),
        "two-cell box": ThinWalledSection.from_segments(_two_cell(lam, web)),
    }


def _two_cell(lam, web):
    a, b, h = CELL_A, CELL_B, CELL_H
    xs = [-a, 0.0, b]
    out = []
    for x0, x1 in zip(xs[:-1], xs[1:]):
        out += segments((x0, -h / 2), (x1, -h / 2), 6, lam) + segments((x1, h / 2), (x0, h / 2), 6, lam)
    out += segments((-a, h / 2), (-a, -h / 2), 4, lam) + segments((b, -h / 2), (b, h / 2), 4, lam)
    out += segments((0.0, -h / 2), (0.0, h / 2), 4, web)
    return out


# ----------------------------------------------------------------------------- closed forms
def closed_forms(E, G):
    t = T
    wall_twist = lambda length, tt=t: G * length * tt**3 / 3.0  # noqa: E731 Saint-Venant of a thin strip
    out = {}
    out["tube"] = {"EA": E * 2 * np.pi * R * t, "EI": E * np.pi * R**3 * t,
                   "GJ": G * 2 * np.pi * R**3 * t + wall_twist(2 * np.pi * R),
                   "kappa": 0.5, "x_sc": 0.0, "EGamma": 0.0}
    out["slit tube"] = {"EA": E * 2 * np.pi * R * t, "EI": E * np.pi * R**3 * t, "GJ": wall_twist(2 * np.pi * R),
                        "x_sc": -2.0 * R, "EGamma": E * 2 * np.pi * R**5 * t * (np.pi**2 / 3 - 2)}
    out["I-beam"] = {"EA": E * (2 * B + H) * t, "EI": E * (t * H**3 / 12 + 2 * B * t * (H / 2) ** 2),
                     "GJ": wall_twist(2 * B + H), "x_sc": 0.0, "EGamma": E * t * B**3 * H**2 / 24}
    out["channel"] = {"EA": E * (2 * B + H) * t, "EI": E * (t * H**3 / 12 + 2 * B * t * (H / 2) ** 2),
                      "GJ": wall_twist(2 * B + H), "x_sc": -3 * B**2 / (H + 6 * B),
                      "EGamma": E * t * B**3 * H**2 / 12 * (3 * B + 2 * H) / (6 * B + H)}
    # two cells: Bredt-Batho, q_i * delta_i - q_j * delta_ij = 2 A_i G theta (per unit twist rate)
    a, b, h = CELL_A, CELL_B, CELL_H
    A = np.array([a * h, b * h])
    d_web = h / WEB_T
    d = np.array([(2 * a + h) / t + d_web, (2 * b + h) / t + d_web])
    M = np.array([[d[0], -d_web], [-d_web, d[1]]])
    q = np.linalg.solve(M, 2 * A * G)
    walls = wall_twist(2 * (a + b) + 2 * h) + wall_twist(h, WEB_T)
    out["two-cell box"] = {"GJ": 2 * float(q @ A) + walls, "q_ratio": float(q[1] / q[0])}
    return out


def thin_results(lib):
    al = lib.get("Al-6061-T6")
    E, G = al.elastic.E1, al.elastic.G12
    lam = Laminate.from_angles(al, [0], T)
    web = Laminate.from_angles(al, [0], WEB_T)
    secs = thin_sections(lam, web)
    out = {"E": E, "G": G, "exact": closed_forms(E, G), "computed": {}, "drawing": {}}
    for name, s in secs.items():
        K = s.timoshenko_stiffness(s.shear_centre)
        out["computed"][name] = {"EA": s.EA, "EI": s.EI_xx, "GJ": s.GJ, "EGamma": s.EGamma,
                                 "x_sc": s.shear_centre[0], "y_sc": s.shear_centre[1],
                                 "x_c": s.centroid[0], "kappa": K[1, 1] / (G * s.EA / E)}
        out["drawing"][name] = {"points": s.points.tolist(), "start": s.start.tolist(), "end": s.end.tolist(),
                                "omega": np.asarray(s.omega).tolist()}
    # two-cell: the shear flows of the two cells per unit twist, from the walls' flows
    s = secs["two-cell box"]
    mid = 0.5 * (s.points[s.start] + s.points[s.end])
    top = np.isclose(mid[:, 1], CELL_H / 2)
    q = np.asarray(s.shear_flow)
    q_a = np.mean(np.abs(q[top & (mid[:, 0] < 0)]))
    q_b = np.mean(np.abs(q[top & (mid[:, 0] > 0)]))
    out["computed"]["two-cell box"]["q_ratio"] = float(q_b / q_a)

    # the tube as a polygon of n walls: second-order convergence to the circle
    conv = {"n": [], "EA": [], "EI": [], "GJ": []}
    ex = out["exact"]["tube"]
    for n in (8, 16, 32, 64, 128, 256):
        t_ = ThinWalledSection.from_segments(arc(n, lam))
        conv["n"].append(n)
        for k, v in (("EA", t_.EA), ("EI", t_.EI_xx), ("GJ", t_.GJ)):
            conv[k].append(abs(v / ex[k] - 1))
    out["polygon"] = conv
    return out


# ----------------------------------------------------------------------------- solid
def rectangle_torsion(a, b, terms=400):
    """Saint-Venant torsion constant of an a x b rectangle, a >= b (Timoshenko & Goodier)."""
    s = sum(math.tanh(n * math.pi * a / (2 * b)) / n**5 for n in range(1, 2 * terms, 2))
    return a * b**3 / 3 * (1 - 192 * b / (math.pi**5 * a) * s)


def solid_results():
    # nu = 0: the elasticity solution of a shear-loaded rectangle is the parabola, so kappa = 5/6 exactly
    E, nu = 1.0, 0.0
    G = E / (2 * (1 + nu))
    mat = {1: SolidMaterial.isotropic(E, nu)}
    iso = lambda u, v: (1, 0.0)  # noqa: E731
    b = 0.1
    out = {"aspect": {}, "convergence": {}}
    for ar in (1, 2, 5, 10):
        a = ar * b
        mesh = structured_mesh([rectangle_patch(-a / 2, -b / 2, a, b, 8 * ar, 8, iso)], mat, "Q8")
        K = SolidSectionAnalysis(mesh).stiffness
        out["aspect"][str(ar)] = {"J": K[5, 5] / G, "J_exact": rectangle_torsion(a, b),
                                  "kappa_y": K[3, 3] / (G * a * b), "kappa_z": K[4, 4] / (G * a * b),
                                  "EA": K[0, 0] / (E * a * b), "EI": K[2, 2] / (E * b * a**3 / 12)}
    # mesh convergence of the square's torsion constant
    J0 = rectangle_torsion(b, b)
    for kind in ("Q4", "Q8", "T6"):
        rows = []
        for n in (2, 4, 8, 16, 32 if kind == "Q4" else 24):
            mesh = structured_mesh([rectangle_patch(-b / 2, -b / 2, b, b, n, n, iso)], mat, kind)
            K = SolidSectionAnalysis(mesh).stiffness
            rows.append((1.0 / n, abs(K[5, 5] / G / J0 - 1)))
        out["convergence"][kind] = rows

    # thick tube: thin-wall and solid against the exact stiffness, t / R from 1 % to 100 %
    al = create_default_library().get("Al-6061-T6")
    E_al, G_al = al.elastic.E1, al.elastic.G12
    mat_al = {1: SolidMaterial.isotropic(E_al, al.elastic.v12)}
    thick = {"t_over_R": [], "thin_GJ": [], "thin_EI": [], "solid_GJ": [], "solid_EI": []}
    for ratio in (0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
        t = ratio * R
        ro, ri = R + t / 2, R - t / 2
        EI_x, GJ_x = np.pi / 4 * (ro**4 - ri**4), np.pi / 2 * (ro**4 - ri**4)
        mesh = structured_mesh([arc_patch((0.0, 0.0), ri, ro, 0.0, 2 * np.pi, 96, 4, iso)], mat_al, "Q8")
        K = SolidSectionAnalysis(mesh).stiffness
        lam = Laminate.from_angles(al, [0], t)
        s = ThinWalledSection.from_segments(arc(256, lam))
        thick["t_over_R"].append(ratio)
        thick["solid_GJ"].append(K[5, 5] / (G_al * GJ_x))
        thick["solid_EI"].append(K[2, 2] / (E_al * EI_x))
        thick["thin_GJ"].append(s.GJ / (G_al * GJ_x))
        thick["thin_EI"].append(s.EI_xx / (E_al * EI_x))
    out["thick_tube"] = thick
    return out


# ----------------------------------------------------------------------------- checks
def check(r) -> Checks:
    c = Checks("08 isotropic sections")
    t, ex = r["thin"]["computed"], r["thin"]["exact"]
    scale = {"x_sc": R}
    for name, terms in ex.items():
        for k, ref in terms.items():
            if k not in t[name]:
                continue
            tol = {"tube": 3e-4, "slit tube": 3e-3}.get(name, 1e-3)
            if k == "EGamma" and ref == 0.0:
                c.close(f"{name} EGamma (zero for a closed tube)", t[name][k], 0.0, 1e-6, "closed tube: no warping",
                        scale=ex[name]["EI"] * R**2)
                continue
            src = {"tube": "Bredt-Batho, thin ring", "slit tube": "open thin ring", "I-beam": "Vlasov, thin I",
                   "channel": "thin channel", "two-cell box": "Bredt-Batho, two cells"}[name]
            c.close(f"{name} {k}", t[name][k], ref, tol, src, scale=scale.get(k) if ref == 0.0 or k == "x_sc" else None)
    s = r["solid"]
    for ar, v in s["aspect"].items():
        c.close(f"solid rectangle {ar}:1 J", v["J"], v["J_exact"], 1e-4, "Saint-Venant series")
        c.close(f"solid rectangle {ar}:1 kappa (nu = 0)", v["kappa_z"], 5 / 6, 1e-4, "parabolic shear")
        c.close(f"solid rectangle {ar}:1 EI", v["EI"], 1.0, 1e-9, "b h^3 / 12")
    q8 = s["convergence"]["Q8"]
    rate = math.log(q8[-2][1] / q8[-1][1]) / math.log(q8[-2][0] / q8[-1][0])
    c.true(f"Q8 torsion error converges at order >= 3.5 (got {rate:.2f})", rate >= 3.5, "h^4 expected", rate)
    th = r["solid"]["thick_tube"]
    for k in ("solid_GJ", "solid_EI"):
        c.close(f"thick tube t = R, {k.replace('_', ' ')}", th[k][-1], 1.0, 1e-4, "exact annulus")
    c.close("thin-wall tube t = 0.01 R, GJ", th["thin_GJ"][0], 1.0, 1e-3, "exact annulus")
    return c


# ----------------------------------------------------------------------------- figures
def figure_sections(r):
    names = list(r["thin"]["drawing"])
    fig, axes = plt.subplots(1, len(names), figsize=(15, 3.6), gridspec_kw={"width_ratios": [1, 1.5, 0.55, 0.75, 2.2]})
    for ax, name in zip(axes, names):
        d = r["thin"]["drawing"][name]
        P, om = np.array(d["points"]), np.array(d["omega"])
        lim = max(np.max(np.abs(om)), 1e-6)  # a closed tube's omega is round-off: draw it as zero
        for i, j in zip(d["start"], d["end"]):
            ax.plot(P[[i, j], 0], P[[i, j], 1], color=DIVERGING(0.5 + 0.5 * (om[i] + om[j]) / (2 * lim)), lw=3.2,
                    solid_capstyle="round")
        cp = r["thin"]["computed"][name]
        ax.plot(cp["x_c"], 0.0, "+", color=TEXT_SECONDARY, ms=9, mew=1.5, label="centroid")
        ax.plot(cp["x_sc"], cp["y_sc"], "o", color=SERIES[1], ms=7, label="shear centre")
        ex = r["thin"]["exact"][name]
        if "x_sc" in ex:
            ax.plot(ex["x_sc"], 0.0, "x", color=REFERENCE, ms=9, mew=1.5, label="closed form")
        ax.set_aspect("equal")
        ax.set_title(name, loc="left")
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Thin-wall shear centres sit on the closed forms; colour is the warping function ω "
                 "(zero round the closed tube)", x=0.0, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.0, -0.03, f"Aluminium 6061-T6, walls {T * 1e3:g} mm. Tube R = {R:g} m (256 walls); I-beam and channel "
             f"{B:g} x {H:g} m mid-line; two-cell box {CELL_A:g} + {CELL_B:g} x {CELL_H:g} m, middle web "
             f"{WEB_T * 1e3:g} mm.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "isotropic_sections.png")


def figure_convergence(r):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    ax = axes[0]
    p = r["thin"]["polygon"]
    for k, col in zip(("EA", "EI", "GJ"), SERIES):
        ax.loglog(p["n"], p[k], "o-", color=col, label=k)
    n = np.array(p["n"], float)
    ax.loglog(n, 2.0 * p["GJ"][0] * (n[0] / n) ** 2, ls="--", color=REFERENCE, lw=1, label="slope -2")
    ax.set_xlabel("walls round the tube")
    ax.set_ylabel("relative error against the circle")
    ax.set_title("Polygon tube: second order in the wall size", loc="left")
    ax.legend()
    ax = axes[1]
    for kind, col in zip(("Q4", "Q8", "T6"), SERIES):
        h, e = np.array(r["solid"]["convergence"][kind]).T
        ax.loglog(1 / h, e, "o-", color=col, label=kind)
    h = np.array(r["solid"]["convergence"]["Q8"])[:, 0]
    ax.loglog(1 / h, 0.5 * r["solid"]["convergence"]["Q8"][1][1] * (h / h[1]) ** 4, ls="--", color=REFERENCE, lw=1,
              label="slope -4")
    ax.set_xlabel("elements along each side")
    ax.set_ylabel("torsion constant error")
    ax.set_title("Solid square: Q8 and T6 converge at h^4, Q4 at h^2", loc="left")
    ax.legend()
    ax = axes[2]
    th = r["solid"]["thick_tube"]
    ax.semilogx(th["t_over_R"], th["thin_GJ"], "o-", color=SERIES[0], label="thin-wall GJ")
    ax.semilogx(th["t_over_R"], th["thin_EI"], "s-", color=SERIES[1], label="thin-wall EI")
    ax.semilogx(th["t_over_R"], th["solid_GJ"], "o--", color=SERIES[2], mfc="none", label="solid GJ")
    ax.semilogx(th["t_over_R"], th["solid_EI"], "s--", color=SERIES[3], mfc="none", label="solid EI")
    ax.axhline(1.0, color=REFERENCE, lw=1, ls=":")
    ax.set_xlabel("wall thickness / mid-line radius")
    ax.set_ylabel("stiffness / exact annulus")
    ax.set_title("Thick tube: thin-wall error grows as (t/R)^2", loc="left")
    ax.legend(fontsize=7.5)
    fig.text(0.0, -0.04, "Left: aluminium tube R = 0.1 m, t = 2 mm, as a polygon. Middle: unit square, E = 1, ν = 0, "
             "against Saint-Venant's series. Right: aluminium annulus of mid-line radius 0.1 m, solid 96 x 4 Q8.",
             fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "isotropic_convergence.png")


def report(r):
    t, ex = r["thin"]["computed"], r["thin"]["exact"]
    print("Thin-wall sections against the closed forms (computed / exact):")
    for name in ex:
        parts = []
        for k, v in ex[name].items():
            if k in t[name] and v != 0.0:
                parts.append(f"{k} {t[name][k] / v:.5f}")
        print(f"  {name:<13} " + ", ".join(parts))
    print("Solid rectangles (J / series, kappa_z / (5/6)):")
    for ar, v in r["solid"]["aspect"].items():
        print(f"  {ar:>2}:1  J {v['J'] / v['J_exact']:.6f}  kappa {v['kappa_z'] / (5 / 6):.6f}")
    th = r["solid"]["thick_tube"]
    print("Thick tube GJ / exact: " + ", ".join(f"t/R {a:g}: thin {b:.3f}, solid {c:.5f}"
                                                for a, b, c in zip(th["t_over_R"], th["thin_GJ"], th["solid_GJ"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", action="store_true", help="redraw from data/results.json")
    args = parser.parse_args()
    if args.reuse and DATA.exists():
        r = json.loads(DATA.read_text())
    else:
        r = {"thin": thin_results(create_default_library()), "solid": solid_results()}
        DATA.parent.mkdir(exist_ok=True)
        DATA.write_text(json.dumps(r, indent=1))
    report(r)
    figure_sections(r)
    figure_convergence(r)
    check(r).report()


if __name__ == "__main__":
    main()
