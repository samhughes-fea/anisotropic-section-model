"""Natural frequencies and mode shapes of beams and blades, coupled, from their sections.

``ModalBeam`` takes the 6 x 6 Timoshenko stiffness and the 6 x 6 mass matrix of each
section, with every coupling, and finds the beam's natural frequencies and mode shapes.
The example checks it three ways:

1. **Exact solutions.** A composite box cantilever, orthotropic and bend-twist coupled
   (CAS), against ``prismatic_frequencies`` (the exact solution of the same 12 beam
   equations, by a matrix exponential) and against the textbook closed forms for
   bending (Euler-Bernoulli), torsion and extension.
2. **A shell model.** The same two boxes as Quad8 shell cantilevers (shell-fem's
   stiffness, a consistent shell mass matrix): frequencies, mode shapes (MAC) and how
   much the CAS box twists as it bends in its first mode.
3. **The IEA 15 MW blade.** Its official BeamDyn deck, curved (prebent) and straight,
   against BeamDyn itself: free-vibration runs of OpenFAST's ``beamdyn_driver`` (when
   ``BEAMDYN_DRIVER`` points at it; otherwise the committed BeamDyn frequencies are
   used), with the frequencies identified from the tip motion.

Figures: the IEA 15 MW mode shapes; every frequency against its reference; the CAS box's
coupled first mode against the shell.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import numpy as np  # noqa: E402

from _shared import composite_box as box  # noqa: E402
from _shared.checks import Checks  # noqa: E402
from _shared.plotting import REFERENCE, SERIES, TEXT_SECONDARY, plt, save  # noqa: E402
from anisotropic_section_model import BeamDynBlade, ThinWalledSection  # noqa: E402
from anisotropic_section_model.modal import ModalBeam, _ordinal, mac, prismatic_frequencies  # noqa: E402

DATA = HERE / "data"
RESULTS = DATA / "results.json"
REFERENCE_DIR = DATA / "reference"
IEA_COMMIT = "28ae083f6e4d2e2e4ea35ac9100fcab104940484"
IEA_URL = f"https://raw.githubusercontent.com/IEAWindTask37/IEA-15-240-RWT/{IEA_COMMIT}/OpenFAST/IEA-15-240-RWT/"
IEA_FILES = ("IEA-15-240-RWT_BeamDyn.dat", "IEA-15-240-RWT_BeamDyn_blade.dat")

#: the spar box of example 10 with 12 plies (12 mm walls), so its panels' own modes
#: lie above the beam modes compared here
SPAR = box.Box(0.404, 0.204, 12, 1.0e-3, "T300/5208")
LENGTH = 12.0          # m, a slender cantilever (L / width = 30)
N_SPAN = 80            # shell elements along the span
THETA = 15.0           # CAS ply angle, degrees
N_MODES = 10
LAYUPS = {"orthotropic": lambda: box.cas(0.0, SPAR.plies), "CAS": lambda: box.cas(THETA, SPAR.plies)}
EULER_BERNOULLI = (1.875104069, 4.694091133, 7.854757438)   # beta L of a cantilever


# ====================================================================== 1 and 2: the box
def shell_box(lib, name):
    """The shell cantilever, its section (6 x 6 and mass, about the box centre) and the
    mesh's stiffness and consistent mass reduced to the free DOFs."""
    from shell_fem import BoundaryConditions, GlobalAssembler
    from shell_fem.laminate_solver_pipeline import ConstraintEliminator

    b = box.shell(lib, LAYUPS[name](), LENGTH, n_width=8, n_depth=4, n_span=N_SPAN, b=SPAR)
    mesh = b.mesh
    sec = ThinWalledSection.from_blade_mesh(b, z=LENGTH / 2)
    xyz = {n: np.asarray(mesh.node_coordinates(n)) for n in mesh.node_ids}
    bcs = BoundaryConditions().clamp([n for n, p in xyz.items() if abs(p[2]) < 1e-12])
    K = GlobalAssembler(sparse=True).assemble_stiffness(mesh, bcs)
    M = shell_mass(mesh, xyz)
    T, _ = ConstraintEliminator().build(mesh, bcs)
    return sec, mesh, xyz, (T.T @ K @ T).tocsc(), (T.T @ M @ T).tocsc(), T


def shell_mass(mesh, xyz):
    """Consistent translational mass of the Quad8 shells (flat rectangles here): the
    areal density times the integral of N N^T. Rotary inertia of the walls is left out."""
    import scipy.sparse as sp

    xi = np.array([-1, 1, 1, -1, 0, 1, 0, -1.0])
    et = np.array([-1, -1, 1, 1, -1, 0, 1, 0.0])

    def n8(x, e):
        N = np.zeros(8)
        for a in range(4):
            N[a] = 0.25 * (1 + xi[a] * x) * (1 + et[a] * e) * (xi[a] * x + et[a] * e - 1)
        for a in (4, 6):
            N[a] = 0.5 * (1 - x * x) * (1 + et[a] * e)
        for a in (5, 7):
            N[a] = 0.5 * (1 + xi[a] * x) * (1 - e * e)
        return N

    g, w = np.polynomial.legendre.leggauss(3)
    ref = sum(wi * wj * np.outer(n8(a, c), n8(a, c)) for a, wi in zip(g, w) for c, wj in zip(g, w))
    rows, cols, vals = [], [], []
    for el in mesh.elements:
        ids = list(el.node_ids)
        P = np.array([xyz[i] for i in ids[:4]])
        area = np.linalg.norm(P[1] - P[0]) * np.linalg.norm(P[3] - P[0])
        Me = ref * area / 4.0 * el.laminate.areal_density()
        for d in range(3):
            idx = [mesh.dof_index(i, d) for i in ids]
            rows += list(np.repeat(idx, 8))
            cols += list(np.tile(idx, 8))
            vals += list(Me.ravel())
    return sp.csr_matrix((vals, (rows, cols)), shape=(mesh.n_dof, mesh.n_dof))


def section_motion(U, mesh, xyz, z_stations):
    """[u_x, u_y, u_z, twist] of each ring of nodes: mean translations and the
    least-squares rotation about the ring's centre."""
    out = []
    for z in z_stations:
        nodes = [i for i, p in xyz.items() if abs(p[2] - z) < 1e-9]
        P = np.array([xyz[i][:2] for i in nodes])
        d = np.array([[U[mesh.dof_index(i, k)] for k in range(3)] for i in nodes])
        r, ut = P - P.mean(0), d[:, :2] - d[:, :2].mean(0)
        out.append([*d.mean(0), float(np.sum(r[:, 0] * ut[:, 1] - r[:, 1] * ut[:, 0]) / np.sum(r * r))])
    return np.array(out)


def boxes(lib):
    import scipy.sparse.linalg as sla

    out = {}
    for name in LAYUPS:
        t0 = time.perf_counter()
        sec, mesh, xyz, Kr, Mr, T = shell_box(lib, name)
        K6, M6 = sec.timoshenko_stiffness((0.0, 0.0)), sec.mass_matrix((0.0, 0.0))
        beam = ModalBeam.prismatic(K6, M6, LENGTH).clamp(0.0).modes(N_MODES)
        exact = prismatic_frequencies(K6, M6, LENGTH, 1.001 * beam.frequencies[-1])
        lam, V = sla.eigsh(Kr, k=3 * N_MODES, M=Mr, sigma=-1.0, which="LM")
        order = np.argsort(lam)
        f_shell = np.sqrt(np.abs(lam[order])) / (2 * np.pi)
        V = V[:, order]
        zs = np.unique(np.round([p[2] for p in xyz.values()], 12))[2::2]  # element ends, root left out
        shell_shapes = [section_motion(T @ V[:, k], mesh, xyz, zs) for k in range(V.shape[1])]
        radius = 0.5 * np.hypot(SPAR.width, SPAR.depth)  # twist as a displacement at the corners
        scale = np.array([1.0, 1.0, 1.0, radius])
        modes = []
        for k in range(N_MODES):
            b = beam.shape(k, zs)[:, [0, 1, 2, 5]]
            m = [mac(b * scale, s * scale) for s in shell_shapes]
            j = int(np.argmax(m))
            tip_b, tip_s = b[-1], shell_shapes[j][-1]
            d = int(np.argmax(np.abs(tip_b[:2])))
            modes.append({"label": beam.labels()[k], "beam": float(beam.frequencies[k]),
                          "exact": float(exact[k]), "shell": float(f_shell[j]), "shell_mode": j + 1,
                          "mac": m[j], "twist_per_deflection_beam": float(tip_b[3] / tip_b[d]),
                          "twist_per_deflection_shell": float(tip_s[3] / tip_s[d])})
            if k == 0:  # the first mode's shapes, for the figure
                modes[0].update(beam_shape=_rounded(b), shell_shape=_rounded(shell_shapes[j]))
        mass = M6[0, 0]
        # towards the Euler-Bernoulli beam of the same section: shear stiffness scaled up
        # and bending rotary inertia down, by 100 (which leaves 1e-4 of their effect)
        K_eb, M_eb = K6.copy(), M6.copy()
        K_eb[:2, :] *= 10.0
        K_eb[:, :2] *= 10.0
        M_eb[3:5, 3:5] *= 1e-2
        eb = ModalBeam.prismatic(K_eb, M_eb, LENGTH).clamp(0.0).modes(6)
        closed = {"bending_x": [bl ** 2 / (2 * np.pi * LENGTH ** 2) * np.sqrt(K6[4, 4] / mass) for bl in
                                EULER_BERNOULLI],
                  "bending_y": [bl ** 2 / (2 * np.pi * LENGTH ** 2) * np.sqrt(K6[3, 3] / mass) for bl in
                                EULER_BERNOULLI],
                  "torsion": [(2 * n - 1) / (4 * LENGTH) * np.sqrt(K6[5, 5] / M6[5, 5]) for n in (1, 2)],
                  "axial": [1 / (4 * LENGTH) * np.sqrt(K6[2, 2] / mass)]}
        eb_x = [float(f) for f, lab in zip(eb.frequencies, eb.labels()) if lab.endswith("x-bending")]
        eb_y = [float(f) for f, lab in zip(eb.frequencies, eb.labels()) if lab.endswith("y-bending")]
        out[name] = {"z": zs.tolist(), "modes": modes, "closed_form": closed, "euler_bernoulli": {"x": eb_x, "y": eb_y}, "shell_frequencies": f_shell.tolist(),
                     "K6": K6.tolist(), "M6": M6.tolist()}
        print(f"{name} box: {len(f_shell)} shell modes, {time.perf_counter() - t0:.1f} s")
    return out


# ====================================================================== 3: IEA 15 MW
def fetch_reference():
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    for name in IEA_FILES:
        path = REFERENCE_DIR / name
        if not path.exists():
            with urllib.request.urlopen(IEA_URL + name, timeout=60) as r:
                path.write_bytes(r.read())
    return REFERENCE_DIR / IEA_FILES[0]


def write_multi_member(blade, path, n_members=7, order=8):
    """A BeamDyn deck with ``n_members`` spectral elements of ``order`` (Gauss
    quadrature) and no numerical damping (rhoinf = 1), so BeamDyn's own discretisation
    error is small next to the differences being checked."""
    blade.write(path, version=4, order_elem=order, quadrature=1)
    lines = path.read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if "member_total" in line)
    k = len(blade.key_points)
    cuts = np.round(np.linspace(0, k - 1, n_members + 1)).astype(int)
    members = [f"{m + 1:>6d} {cuts[m + 1] - cuts[m] + 1:>6d}                 - Member number; Number of key "
               "points in this member" for m in range(n_members)]
    new = [f"{n_members:>11d}   member_total    - Total number of members (-)", lines[i + 1], *members,
           lines[i + 3], lines[i + 4], *lines[i + 5:i + 5 + k]]
    out = lines[:i] + new + lines[i + 5 + k:]
    out = [re.sub(r"^\s*0\s+rhoinf", "          1   rhoinf", line) for line in out]
    path.write_text("\n".join(out) + "\n")
    return path


def beamdyn_free_vibration(driver, primary, tip_load, t_final=60.0, dt=0.002):
    """BeamDyn dynamic run from rest under a constant tip load (a step): the tip then
    oscillates about its static deflection in the modes the load excites."""
    primary = Path(primary).resolve()
    lines = ["--- BeamDyn driver", "free vibration", "--- SIMULATION CONTROL", "True DynamicSolve", "0 t_initial",
             f"{t_final} t_final", f"{dt} dt", "--- GRAVITY", "0 G", "0 G", "0 G", "--- FRAME", "0 GlbPos1",
             "0 GlbPos2", "0 GlbPos3", "--- DCM", "--- DCM", "1 0 0", "0 1 0", "0 0 1", "T GlbRotBladeT0",
             "--- ROOT VELOCITY", "0 RootVel4", "0 RootVel5", "0 RootVel6", "--- APPLIED FORCE", *["0 DistrLoad"] * 6,
             *[f"{v:.12e} TipLoad" for v in tip_load], "0 NumPointLoads", "eta F M", "(-) (N) (N-m)",
             "--- PRIMARY INPUT FILE", f"\"{primary.name}\" InputFile", "--- Outputs", "0 WrVTK", "15 VTK_fps"]
    drv = primary.with_name(primary.stem + "_driver.inp")
    drv.write_text("\n".join(lines) + "\n")
    subprocess.run([driver, drv.name], cwd=primary.parent, check=True, capture_output=True)
    out = drv.with_suffix(".out").read_text().splitlines()
    j = next(i for i, line in enumerate(out) if line.split()[:1] == ["Time"])
    names = out[j].split()
    data = np.array([[float(v) for v in line.split()] for line in out[j + 2:] if line.strip()])
    return dict(zip(names, data.T))


def identify(t, y, f_max=8.0, n_peaks=12, floor=1e-3):
    """Frequencies in a free-vibration record: peaks of the windowed, zero-padded
    spectrum, refined by a least-squares fit of a sum of sinusoids. Components below
    ``floor`` of the largest amplitude (spectral leakage) are dropped."""
    from scipy.optimize import least_squares

    y = y - y.mean()
    dt = t[1] - t[0]
    n_fft = 16 * len(y)
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)), n_fft))
    f = np.fft.rfftfreq(n_fft, dt)
    sel = (f > 0.1) & (f < f_max)
    i = np.where(sel[1:-1] & (Y[1:-1] > Y[:-2]) & (Y[1:-1] > Y[2:]) & (Y[1:-1] > floor * Y[sel].max()))[0] + 1
    f0 = np.sort(f[i[np.argsort(-Y[i])][:n_peaks]])

    def basis(fr):
        return np.column_stack([np.ones_like(t)] + [g(2 * np.pi * fk * t) for fk in fr for g in (np.cos, np.sin)])

    def residual(fr):
        B = basis(fr)
        return B @ np.linalg.lstsq(B, y, rcond=None)[0] - y

    fr = least_squares(residual, f0, x_scale=0.01, xtol=1e-14, ftol=1e-14).x
    c = np.linalg.lstsq(basis(fr), y, rcond=None)[0]
    amp = np.hypot(c[1::2], c[2::2])
    keep = amp > floor * amp.max()
    return sorted(float(v) for v in fr[keep])


def beamdyn_frequencies(driver, blade, work):
    deck = write_multi_member(blade, work / "iea15_modal.dat")
    run = beamdyn_free_vibration(driver, deck, (1e3, 1e3, 0.0, 0.0, 0.0, 1e3))  # flap, edge, torque
    found = []
    for ch in ("TipTDxr", "TipTDyr", "TipRDzr"):
        found += identify(run["Time"], run[ch])
    found = np.sort(found)
    merged = [found[0]]  # the same mode seen in several channels
    for v in found[1:]:
        if v - merged[-1] > 2e-3 * v:
            merged.append(v)
        else:
            merged[-1] = 0.5 * (merged[-1] + v)
    return [float(v) for v in merged]


def iea15(driver, saved):
    try:
        primary = fetch_reference()
    except OSError as exc:
        print(f"IEA 15 MW part skipped (no reference files: {exc})")
        return saved
    blade = BeamDynBlade.read(primary)
    curved = ModalBeam.from_beamdyn(blade).modes(N_MODES)
    straight = ModalBeam.from_beamdyn(blade, curved=False).modes(N_MODES)
    coarse = ModalBeam.from_beamdyn(blade, order=4).modes(N_MODES)
    out = {"length": blade.length, "stations": len(blade.eta), "labels": curved.labels(),
           "curved": curved.frequencies.tolist(), "straight": straight.frequencies.tolist(),
           "order4": coarse.frequencies.tolist(), "energy": curved.energy_fractions().tolist(),
           "eta": np.linspace(0.0, 1.0, 41).tolist()}
    s = np.asarray(out["eta"]) * curved.beam.length
    out["shapes"] = [_rounded(curved.shape(k, s)) for k in range(6)]
    if driver:
        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.perf_counter()
            out["beamdyn"] = beamdyn_frequencies(driver, blade, Path(tmp))
            print(f"BeamDyn free vibration: {len(out['beamdyn'])} frequencies, {time.perf_counter() - t0:.0f} s")
        banner = subprocess.run([driver, "-v"], capture_output=True, text=True).stdout
        version = re.search(r"OpenFAST - (v[\d.]+)", banner)
        out["beamdyn_source"] = f"beamdyn_driver, OpenFAST {version.group(1) if version else '?'}"
    elif saved and "beamdyn" in saved:
        out["beamdyn"] = saved["beamdyn"]
        out["beamdyn_source"] = saved.get("beamdyn_source", "committed results")
        print("BeamDyn frequencies from the committed results (set BEAMDYN_DRIVER to rerun BeamDyn)")
    return out


def _rounded(a, digits=6):
    """A shape as nested lists of floats to ``digits`` significant digits (keeps the JSON small)."""
    return [[float(f"{v:.{digits}g}") for v in row] for row in np.asarray(a)]


def nearest(values, target):
    values = np.asarray(values)
    return float(values[np.argmin(np.abs(values - target))])


# ====================================================================== checks
def check(r) -> Checks:
    c = Checks("13 beam modal analysis")
    for name, b in r["boxes"].items():
        modes = b["modes"]
        for m in modes:
            c.close(f"{name} box {m['label']}: FE against the exact solution", m["beam"], m["exact"], 1e-6,
                    "prismatic_frequencies")
        cf = b["closed_form"]
        fx = [m["beam"] for m in modes if m["label"].endswith("x-bending")]
        fy = [m["beam"] for m in modes if m["label"].endswith("y-bending")]
        ft = [m["beam"] for m in modes if m["label"].endswith("torsion")]
        eb = b["euler_bernoulli"]
        if name == "orthotropic":
            for d in "xy":
                for k in range(2):
                    c.close(f"orthotropic box, near shear-rigid: {_ordinal(k + 1)} {d}-bending, Euler-Bernoulli",
                            eb[d][k], cf[f"bending_{d}"][k], 5e-4 if k == 0 else 2e-3, "(beta L)^2 / (2 pi L^2) sqrt(EI / m)")
        else:  # the CAS box's flapwise bending couples with twist, so it sits well below
            c.true(f"CAS box 1st y-bending is {1 - fy[0] / cf['bending_y'][0]:.0%} below Euler-Bernoulli "
                   "on K44 (bend-twist coupling)", fy[0] < 0.9 * cf["bending_y"][0], "Euler-Bernoulli",
                   value=fy[0])
        c.true(f"{name} box: shear and rotary inertia lower the 1st bending modes "
               f"({fx[0] / cf['bending_x'][0] - 1:+.1%}, x)", fx[0] < cf["bending_x"][0], "Euler-Bernoulli")
        if name == "orthotropic":
            c.close("orthotropic box 1st torsion, closed form", ft[0], cf["torsion"][0], 1e-6,
                    "(1/4L) sqrt(GJ / i_p)")
            ax = [m["beam"] for m in modes if m["label"].endswith("axial")]
            if ax:
                c.close("orthotropic box 1st axial, closed form", ax[0], cf["axial"][0], 1e-6, "(1/4L) sqrt(EA / m)")
        for m in modes:
            if "bending" in m["label"] and int(m["label"][0]) <= 2:
                c.close(f"{name} box {m['label']} against the shell", m["beam"], m["shell"],
                        3e-3 if name == "orthotropic" else 1.5e-2, "shell, matched by MAC")
                c.true(f"{name} box {m['label']} shape: MAC {m['mac']:.4f} > 0.995", m["mac"] > 0.995, "shell")
        t1 = next(m for m in modes if m["label"] == "1st torsion")
        c.true(f"{name} box 1st torsion: the shell is lower ({t1['shell'] / t1['beam'] - 1:+.1%}), "
               "within 6 %, from distortion", 0.94 < t1["shell"] / t1["beam"] < 1.0,
               "no ribs: the section distorts in the shell")
        if name == "CAS":
            m1 = modes[0]
            c.close("CAS box 1st mode: tip twist per tip deflection", m1["twist_per_deflection_beam"],
                    m1["twist_per_deflection_shell"], 0.03, "shell")
            c.true("CAS box couples bending with twist in its first mode",
                   abs(m1["twist_per_deflection_beam"]) > 0.05, value=m1["twist_per_deflection_beam"])
    i = r.get("iea15")
    if i:
        f, labels = i["curved"], i["labels"]
        c.close("IEA 15 MW 1st flap, order 4 against order 6", i["order4"][0], f[0], 2e-5, "converged elements")
        c.true("IEA 15 MW mode order: flap, edge, flap, edge, flap, torsion",
               labels[:6] == ["1st flap", "1st edge", "2nd flap", "2nd edge", "3rd flap", "1st torsion"])
        if "beamdyn" in i:
            for k in range(N_MODES):
                c.close(f"IEA 15 MW {labels[k]} against BeamDyn", f[k], nearest(i["beamdyn"], f[k]), 2.5e-3,
                        f"BeamDyn free vibration ({i['beamdyn_source']})")
        k = labels.index("2nd edge")
        c.true(f"IEA 15 MW 2nd edge: the prebend matters ({i['straight'][k] / f[k] - 1:+.1%} straight)",
               abs(i["straight"][k] / f[k] - 1) > 0.01)
    return c


# ====================================================================== figures and report
def figure_iea(r):
    i = r.get("iea15")
    if not i:
        return
    eta = np.asarray(i["eta"])
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
    for k, ax in enumerate(axes.ravel()):
        phi = np.asarray(i["shapes"][k])
        scale = np.max(np.abs(phi[:, :2])) or 1.0
        ax.plot(eta, phi[:, 0] / scale, color=SERIES[0], label="flap u_x")
        ax.plot(eta, phi[:, 1] / scale, color=SERIES[1], label="edge u_y")
        ax.plot(eta, 10.0 * phi[:, 5] / scale,
                color=SERIES[2], label="twist × 10 m")
        ref = f", BeamDyn {nearest(i['beamdyn'], i['curved'][k]):.3f}" if "beamdyn" in i else ""
        ax.set_title(f"{i['labels'][k]}: {i['curved'][k]:.3f} Hz{ref}", fontsize=9.5)
        ax.axhline(0.0, color=TEXT_SECONDARY, lw=0.6)
        if k == 0:
            ax.legend(loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("span fraction")
    fig.suptitle("IEA 15 MW blade, clamped at the root: the first six modes (ModalBeam, curved axis)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, __file__, "iea15_modes.png")


def figure_frequencies(r):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for ax, name, col in zip(axes[:2], r["boxes"], SERIES):
        modes = r["boxes"][name]["modes"]
        x = np.arange(1, len(modes) + 1)
        ax.plot(x, [100 * (m["shell"] / m["beam"] - 1) for m in modes], "o", color=col, label="shell")
        ax.plot(x, [100 * (m["exact"] / m["beam"] - 1) for m in modes], "_", color=REFERENCE, ms=14, label="exact")
        ax.set_xticks(x, [m["label"].replace("-bending", "") for m in modes], rotation=60, fontsize=7.5)
        ax.set_title(f"{name} box: reference / ModalBeam − 1", fontsize=10)
        ax.set_ylabel("%")
        ax.legend()
    ax = axes[2]
    i = r.get("iea15")
    if i:
        x = np.arange(1, len(i["curved"]) + 1)
        if "beamdyn" in i:
            ax.plot(x, [100 * (nearest(i["beamdyn"], f) / f - 1) for f in i["curved"]], "o", color=SERIES[2],
                    label="BeamDyn")
        ax.plot(x, [100 * (s / f - 1) for s, f in zip(i["straight"], i["curved"])], "s", color=SERIES[3],
                label="ModalBeam, axis straightened")
        ax.set_xticks(x, i["labels"], rotation=60, fontsize=7.5)
        ax.set_title("IEA 15 MW: reference / ModalBeam − 1", fontsize=10)
        ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, __file__, "frequencies.png")


def figure_cas_mode(r):
    b = r["boxes"]["CAS"]
    m = b["modes"][0]
    z = np.asarray(b["z"])
    bs, ss = np.asarray(m["beam_shape"]), np.asarray(m["shell_shape"])
    k = int(np.argmax(np.abs(bs[-1, :2])))
    sb, sh = bs[-1, k], ss[-1, k]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, c, title in ((axes[0], k, "flapwise deflection / tip value"), (axes[1], 3, "twist / tip deflection (rad/m)")):
        ax.plot(z, bs[:, c] / sb, color=SERIES[1], label="ModalBeam")
        ax.plot(z, ss[:, c] / sh, "o", ms=3, color=REFERENCE, label="shell")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("z (m)")
    axes[0].legend()
    fig.suptitle(f"CAS [{THETA:g}]{SPAR.plies} box, {LENGTH:g} m cantilever, 1st mode ({m['beam']:.3f} Hz): "
                 "it twists as it bends", x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, __file__, "cas_coupled_mode.png")


def report(r):
    for name, b in r["boxes"].items():
        print(f"\n{name} box ({SPAR.plies} x {SPAR.ply * 1e3:g} mm {SPAR.material}, {LENGTH:g} m cantilever):")
        print(f"  {'mode':<16}{'ModalBeam':>11}{'exact':>11}{'shell':>11}{'MAC':>8}")
        for m in b["modes"]:
            print(f"  {m['label']:<16}{m['beam']:11.4f}{m['exact']:11.4f}{m['shell']:11.4f}{m['mac']:8.4f}")
    i = r.get("iea15")
    if i:
        print(f"\nIEA 15 MW blade ({i['length']:.1f} m, {i['stations']} stations), root clamped:")
        print(f"  {'mode':<14}{'curved':>9}{'straight':>10}{'BeamDyn':>10}")
        for k, lab in enumerate(i["labels"]):
            bd = f"{nearest(i['beamdyn'], i['curved'][k]):10.4f}" if "beamdyn" in i else ""
            print(f"  {lab:<14}{i['curved'][k]:9.4f}{i['straight'][k]:10.4f}{bd}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", action="store_true", help="redraw and re-check from data/results.json")
    args = parser.parse_args()
    saved = json.loads(RESULTS.read_text()) if RESULTS.exists() else None
    if args.reuse and saved:
        r = saved
    else:
        from laminate_constitutive_model import create_default_library

        driver = os.environ.get("BEAMDYN_DRIVER")
        r = {"boxes": boxes(create_default_library()), "iea15": iea15(driver, (saved or {}).get("iea15"))}
        DATA.mkdir(exist_ok=True)
        RESULTS.write_text(json.dumps(r, indent=1))
    report(r)
    figure_iea(r)
    figure_frequencies(r)
    figure_cas_mode(r)
    check(r).report()


if __name__ == "__main__":
    main()
