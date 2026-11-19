"""BeamDyn and windIO export: frames, mass matrix, the spanwise sweep and the file formats.

``test_beamdyn_driver_matches_timoshenko_beam`` runs OpenFAST's BeamDyn driver on the
written deck when ``BEAMDYN_DRIVER`` points at a ``beamdyn_driver`` executable (built
from OpenFAST 4 or 5), and is skipped otherwise.
"""

import os
import re
import subprocess
import numpy as np
import pytest

from anisotropic_section_model import BeamDynBlade, ThinWalledSection, TimoshenkoBeam
from anisotropic_section_model.axes import twist_from_kernel
from anisotropic_section_model.beamdyn import MIRROR, rotation6, to_beamdyn_frame
from implicit_geometry_kernel import Airfoil, BladeGeometry, Station
from shell_fem import Laminate, create_default_library

AIRFOIL = Airfoil.naca4("0024")
LENGTH = 20.0


@pytest.fixture(scope="module")
def materials():
    lib = create_default_library()
    return lib.get("E-glass/epoxy"), lib.get("T300/5208")


@pytest.fixture(scope="module")
def laminates(materials):
    glass, carbon = materials
    return {"skin": Laminate.from_angles(glass, [0, 45, -45, -45, 45, 0], 0.5e-3),
            "web": Laminate.from_angles(glass, [45, -45, -45, 45], 0.5e-3),
            "cap": Laminate.from_angles(carbon, [20] * 16, 0.5e-3)}  # off-axis caps: couplings


def _geometry(twist=0.0, offset=(0.0, 0.0), chord=(1.0, 1.0), layout="2D-CN-LC-0R"):
    """Chord-normal webs ("-CN") turn with the chord, so the twisted section is the
    untwisted one turned; flapwise webs ("-F") stay along y."""
    stations = [Station(0.0, chord[0], AIRFOIL, twist=twist, pitch_axis=0.3, offset=offset),
                Station(LENGTH, chord[1], AIRFOIL, twist=twist, pitch_axis=0.3, offset=offset)]
    return BladeGeometry(stations, layout, 0.003, pivot_fraction=0.3, spar_cap_thickness=0.008,
                         shear_web_thickness=0.002, web_positions=(0.18, 0.48))


def _box(material, width=0.6, height=0.3, plies=4, t=0.25e-3):
    lam = Laminate.from_angles(material, [0] * plies, t)
    pts = [(0, 0), (width, 0), (width, height), (0, height)]
    return ThinWalledSection(pts, [(0, 1, lam), (1, 2, lam), (2, 3, lam), (3, 0, lam)]), lam


# ---------------------------------------------------------------- mass matrix
def test_mass_matrix_of_a_thin_box(materials):
    glass, _ = materials
    w, h = 0.6, 0.3
    sec, lam = _box(glass, w, h)
    rho_t = sum(p.thickness * p.material.density for p in lam.plies)
    t = lam.thickness
    M = sec.mass_matrix((0.0, 0.0))
    m = rho_t * 2 * (w + h)
    assert M[0, 0] == pytest.approx(m, rel=1e-12)
    # walls on the midline from the origin corner: mass centre at the box centre (the
    # through-thickness offsets of opposite walls cancel)
    assert M[1, 5] / m == pytest.approx(w / 2, rel=1e-9)
    assert -M[0, 5] / m == pytest.approx(h / 2, rel=1e-9)
    assert np.allclose(M[2, 3:5], [m * h / 2, -m * w / 2])
    # about the centre: thin-wall closed forms, plus the walls' own t^2/12
    Mc = sec.mass_matrix((w / 2, h / 2))
    ixx = rho_t * (2 * w * (h / 2) ** 2 + 2 * h ** 3 / 12)  # ∫ρ y²
    iyy = rho_t * (2 * h * (w / 2) ** 2 + 2 * w ** 3 / 12)
    assert Mc[3, 3] == pytest.approx(ixx, rel=2 * (t / h) ** 2)
    assert Mc[4, 4] == pytest.approx(iyy, rel=2 * (t / h) ** 2)
    assert Mc[5, 5] == pytest.approx(Mc[3, 3] + Mc[4, 4])
    assert abs(Mc[3, 4]) < 1e-9 * Mc[3, 3]
    assert np.allclose(Mc[0, 5], 0.0, atol=1e-12 * m) and np.allclose(Mc[1, 5], 0.0, atol=1e-12 * m)
    # parallel axis: kinetic energy of the same motion is the same about any point
    assert np.all(np.linalg.eigvalsh(M) > 0)


def test_mass_matrix_moves_with_the_reference(laminates):
    cut = _geometry().section_at(5.0)
    sec = ThinWalledSection.from_geometry(cut, laminates)
    a, b = np.array([0.1, -0.02]), np.array([-0.2, 0.05])
    Ma, Mb = sec.mass_matrix(a), sec.mass_matrix(b)
    d = b - a
    S = np.eye(6)
    S[0, 5], S[1, 5] = -d[1], d[0]
    S[2, 3], S[2, 4] = d[1], -d[0]
    # point velocities at b from the motion about a: v_b = v_a + theta x (b - a), so M_a = S^T M_b S
    assert np.allclose(Ma, S.T @ Mb @ S, rtol=1e-10, atol=1e-10 * Ma[0, 0])


# ---------------------------------------------------------------- frames
def test_mirror_and_rotation():
    assert np.allclose(MIRROR @ MIRROR, np.eye(6))
    assert np.linalg.det(MIRROR[:3, :3]) == pytest.approx(-1.0)
    R = rotation6(30.0)
    assert np.allclose(R @ R.T, np.eye(6))
    assert np.allclose(R[:2, :2], [[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
    # stiffness terms land where BeamDyn expects them: flap (bending about the chord line,
    # the section's kappa_x) becomes K55, edge (about y) becomes K44
    K = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.allclose(np.diag(to_beamdyn_frame(K)), [2.0, 1.0, 3.0, 5.0, 4.0, 6.0])


def test_twisted_cut_gives_the_untwisted_local_matrices(laminates):
    """BeamDyn's matrices are in the chord frame: a section cut at any twist gives the same."""
    flat = BeamDynBlade.from_geometry(_geometry(0.0), laminates, z=[0.0, 10.0, LENGTH])
    twisted = BeamDynBlade.from_geometry(_geometry(15.0, offset=(0.04, -0.02)), laminates, z=[0.0, 10.0, LENGTH])
    scale = np.abs(flat.stiffness).max()
    assert np.allclose(twisted.stiffness, flat.stiffness, atol=1e-7 * scale)
    assert np.allclose(twisted.mass, flat.mass, atol=1e-9 * np.abs(flat.mass).max())
    assert np.allclose(twisted.twist, -15.0)  # public (and BeamDyn) twist = -kernel twist
    assert np.allclose(twisted.key_points[:, :2], 0.0)  # a constant offset is the root's: a straight axis
    K = flat.stiffness[1]
    assert np.abs(K[2, 5]) > 0.1 * np.sqrt(K[2, 2] * K[5, 5])  # the same 20° plies in both caps: extension-twist


def test_beamdyn_frame_reproduces_the_section_beam(laminates):
    """The written matrices turned back by the twist and mirrored are the section's own."""
    geo = _geometry(12.0, offset=(0.05, -0.03))
    blade = BeamDynBlade.from_geometry(geo, laminates, z=np.linspace(0.0, LENGTH, 3))
    cut = geo.section_at(LENGTH / 2)
    sec = blade.sections[1]
    T = MIRROR @ rotation6(cut.twist)
    assert np.allclose(T.T @ blade.stiffness[1] @ T, sec.timoshenko_stiffness(cut.offset))
    # the same tip load on both beams gives the same (mirrored) tip motion
    f = np.array([1.0, 2.0, 30.0, 4.0, -5.0, 6.0])
    u_sec = TimoshenkoBeam(sec.timoshenko_stiffness(cut.offset), LENGTH, 2).clamp(0.0).add_force(LENGTH, f).solve()
    R = rotation6(-blade.twist[1])  # BeamDyn turns its local frame by -twist about Z
    K_blade = R @ blade.stiffness[1] @ R.T
    u_bd = TimoshenkoBeam(K_blade, LENGTH, 2).clamp(0.0).add_force(LENGTH, MIRROR @ f).solve()
    assert np.allclose(u_bd.node_displacement(LENGTH), MIRROR @ u_sec.node_displacement(LENGTH))


def test_kernel_twist_is_the_negative_of_the_public_twist(laminates):
    """A kernel cut twisted by t is the untwisted section turned by rotation6(-t): the
    public twist (right-handed about z, BeamDyn's and windIO's) is -t."""
    flat = ThinWalledSection.from_geometry(_geometry(0.0).section_at(10.0), laminates)
    cut = _geometry(20.0).section_at(10.0)
    sec = ThinWalledSection.from_geometry(cut, laminates)
    t = float(twist_from_kernel(cut.twist))
    assert t == pytest.approx(-20.0)
    R = rotation6(t)
    K0, K1 = flat.timoshenko_stiffness(), sec.timoshenko_stiffness()
    assert np.allclose(R @ K0 @ R.T, K1, atol=1e-9 * np.abs(K0).max())


def test_from_sections_takes_the_public_twist(laminates):
    geo = _geometry(15.0, offset=(0.04, -0.02))
    z = [0.0, 10.0, LENGTH]
    swept = BeamDynBlade.from_geometry(geo, laminates, z=z)
    cuts = [geo.section_at(v) for v in z]
    sections = [ThinWalledSection.from_geometry(c, laminates) for c in cuts]
    built = BeamDynBlade.from_sections(sections, z, twist=-15.0, offset=[c.offset for c in cuts])
    assert np.allclose(built.twist, swept.twist) and np.allclose(built.twist, -15.0)
    assert np.allclose(built.stiffness, swept.stiffness)


# ---------------------------------------------------------------- sweep
def test_sweep_along_a_tapered_blade(laminates, materials):
    glass, carbon = materials
    geo = BladeGeometry([Station(0.0, 2.0, AIRFOIL, twist=10.0, pitch_axis=0.3),
                         Station(LENGTH, 1.0, AIRFOIL, twist=0.0, pitch_axis=0.3, offset=(0.0, -0.5))],
                        "2D-F-LC-0R", 0.003, pivot_fraction=0.3, spar_cap_thickness=lambda z: 0.012 - 0.0002 * z,
                        shear_web_thickness=0.002, web_positions=(0.18, 0.48))

    def lams(z):  # the caps thin with the span: 24 plies to 16
        n = int(round((0.012 - 0.0002 * z) / 0.5e-3))
        return {**laminates, "cap": Laminate.from_angles(carbon, [0] * n, 0.5e-3)}

    z = np.linspace(0.0, LENGTH, 5)
    blade = BeamDynBlade.from_geometry(geo, lams, z=z)
    assert len(blade.eta) == 5 and blade.eta[0] == 0.0 and blade.eta[-1] == 1.0
    v = blade.station_values()
    assert np.all(np.diff(v["mass"]) < 0) and np.all(np.diff(v["EI_flap"]) < 0)
    assert np.all(v["EI_edge"] > v["EI_flap"])  # K44 is edgewise in BeamDyn's axes
    assert np.allclose(blade.twist, -np.interp(z, [0, LENGTH], [10.0, 0.0]))
    assert blade.key_points[-1, 0] == pytest.approx(-0.5)  # prebend towards the pressure side: kp_xr < 0
    assert blade.key_points[-1, 2] == pytest.approx(LENGTH)
    assert blade.length == pytest.approx(np.hypot(LENGTH, 0.5), rel=1e-9)
    # stations default to the geometry's
    assert len(BeamDynBlade.from_geometry(_geometry(), laminates, z=[0.0, 5.0, LENGTH]).eta) == 3
    with pytest.raises(ValueError):
        BeamDynBlade.from_geometry(_geometry(), laminates, z=[0.0, LENGTH])


# ---------------------------------------------------------------- files
@pytest.fixture(scope="module")
def blade(laminates):
    return BeamDynBlade.from_geometry(_geometry(12.0, offset=(0.05, -0.03)), laminates, z=np.linspace(0.0, LENGTH, 5),
                                      name="prismatic test blade")


@pytest.mark.parametrize("version", [4, 5])
def test_beamdyn_files_round_trip(blade, tmp_path, version):
    mu = [1e-3, 1e-3, 1e-3, 1.4e-3, 2.2e-3, 2.2e-3]
    primary = blade.write(tmp_path / "blade.dat", damping=mu, version=version)
    text = primary.read_text()
    assert ("PITCH ACTUATOR" in text) == (version < 5)
    blade_text = (tmp_path / "blade_blade.dat").read_text()
    assert ("n_modes" in blade_text) == (version >= 5)
    back = BeamDynBlade.read(primary)
    assert np.array_equal(back.eta, blade.eta)
    assert np.array_equal(back.stiffness, blade.stiffness)
    assert np.array_equal(back.mass, blade.mass)
    assert np.array_equal(back.key_points, blade.key_points)
    assert np.array_equal(back.twist, blade.twist)
    assert back.damp_type == 1 and back.damping == mu
    assert back.name == "prismatic test blade"


def test_blade_file_options(blade, tmp_path):
    p = blade.write_blade(tmp_path / "b.dat", modal_damping=[0.01, 0.02])
    b = BeamDynBlade.read_blade(p)
    assert b.damp_type == 2 and b.modal_damping == [0.01, 0.02]
    assert BeamDynBlade.read_blade(blade.write_blade(tmp_path / "u.dat", version=4)).damp_type == 0
    with pytest.raises(ValueError):
        blade.write_blade(tmp_path / "x.dat", modal_damping=[0.01], version=4)
    with pytest.raises(ValueError):
        blade.write_blade(tmp_path / "x.dat", damping=[0.0] * 6, modal_damping=[0.01])
    with pytest.raises(ValueError):
        blade.write_blade(tmp_path / "x.dat", damping=[0.0] * 5)


def test_windio_round_trip(blade, tmp_path):
    w = blade.to_windio(structural_damping={"mode_name": ["flap_1"], "damping_ratio": [0.003]})
    ep = w["structure"]["elastic_properties"]
    assert len([k for k in ep["stiffness_matrix"] if k.startswith("K")]) == 21
    im = ep["inertia_matrix"]
    M = blade.mass[2]
    m = im["mass"][2]
    # windIO's fields fill BeamDyn's mass matrix as WISDEM writes it
    assert M[0, 5] == pytest.approx(-m * im["cm_y"][2]) and M[1, 5] == pytest.approx(m * im["cm_x"][2])
    assert M[3, 3] == im["i_edge"][2] and M[4, 4] == im["i_flap"][2] and M[3, 4] == -im["i_cp"][2]
    for path in (tmp_path / "blade.yaml", tmp_path / "blade.json"):
        back = BeamDynBlade.from_windio(blade.write_windio(path))
        assert np.allclose(back.stiffness, blade.stiffness, rtol=1e-14, atol=0)
        assert np.allclose(back.mass, blade.mass, rtol=1e-12, atol=1e-12 * blade.mass.max())
        assert np.allclose(back.key_points, blade.key_points) and np.allclose(back.twist, blade.twist)


# ---------------------------------------------------------------- BeamDyn itself
DRIVER = os.environ.get("BEAMDYN_DRIVER")


def _run_driver(driver, primary, tip_load=(0.0,) * 6, gravity=(0.0,) * 3):
    lines = ["--- BeamDyn driver", "static check", "--- SIMULATION CONTROL", "False DynamicSolve",
             "0 t_initial", "0 t_final", "0.002 dt", "--- GRAVITY", *[f"{g:.12e} G" for g in gravity],
             "--- FRAME", "0 GlbPos1", "0 GlbPos2", "0 GlbPos3", "--- DCM", "--- DCM", "1 0 0", "0 1 0", "0 0 1",
             "T GlbRotBladeT0", "--- ROOT VELOCITY", "0 RootVel4", "0 RootVel5", "0 RootVel6", "--- APPLIED FORCE",
             *["0 DistrLoad"] * 6, *[f"{v:.12e} TipLoad" for v in tip_load], "0 NumPointLoads", "eta F M",
             "(-) (N) (N-m)", "--- PRIMARY INPUT FILE", f"\"{primary.name}\" InputFile", "--- Outputs", "0 WrVTK",
             "15 VTK_fps"]
    drv = primary.with_name("driver.inp")
    drv.write_text("\n".join(lines) + "\n")
    subprocess.run([driver, drv.name], cwd=primary.parent, check=True, capture_output=True)
    out = drv.with_suffix(".out").read_text().splitlines()
    names = next(line.split() for line in out if line.split()[:1] == ["Time"])
    values = dict(zip(names, map(float, out[-1].split())))
    return np.array([values[n] for n in ("TipTDxr", "TipTDyr", "TipTDzr", "TipRDxr", "TipRDyr", "TipRDzr")])


@pytest.mark.skipif(not DRIVER, reason="set BEAMDYN_DRIVER to an OpenFAST beamdyn_driver to run BeamDyn")
def test_beamdyn_driver_matches_timoshenko_beam(laminates, tmp_path):
    """BeamDyn on the written deck against ASM's own beam on the section, tip loads and gravity.

    A 20 m prismatic blade, twisted 12°, off the pitch axis and with 20° caps (all six
    strains coupled). Loads are small, so BeamDyn's geometrically exact beam stays linear.
    """
    banner = subprocess.run([DRIVER, "-v"], capture_output=True, text=True).stdout  # prints "OpenFAST - v4.1.2"
    version = int(re.search(r"OpenFAST - v(\d+)", banner).group(1))
    geo = _geometry(12.0, offset=(0.05, -0.03))
    blade = BeamDynBlade.from_geometry(geo, laminates, z=np.linspace(0.0, LENGTH, 5))
    primary = blade.write(tmp_path / "blade.dat", quadrature=1, version=version)
    cut = geo.section_at(LENGTH / 2)
    sec = blade.sections[2]
    K, M = sec.timoshenko_stiffness(cut.offset), sec.mass_matrix(cut.offset)
    for k, scale in enumerate([1.0, 1.0, 10.0, 1.0, 1.0, 1.0]):
        f = np.zeros(6)
        f[k] = scale
        u = MIRROR @ TimoshenkoBeam(K, LENGTH, 4).clamp(0.0).add_force(LENGTH, f).solve().node_displacement(LENGTH)
        assert np.allclose(_run_driver(DRIVER, primary, tip_load=MIRROR @ f), u, rtol=0, atol=1e-4 * np.abs(u).max())
    for g in np.eye(3) * 1e-3:  # self-weight from the mass matrix: force and its moment about the axis
        q = M @ np.concatenate([MIRROR[:3, :3] @ g, np.zeros(3)])
        u = MIRROR @ TimoshenkoBeam(K, LENGTH, 4).clamp(0.0).add_distributed(q).solve().node_displacement(LENGTH)
        assert np.allclose(_run_driver(DRIVER, primary, gravity=g), u, rtol=0, atol=1e-4 * np.abs(u).max())
