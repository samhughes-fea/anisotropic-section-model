"""Blade beam properties for OpenFAST BeamDyn and windIO, swept along a blade.

``BeamDynBlade`` holds what a BeamDyn blade needs: the reference axis as key points
with their initial twist, and at each station along it the 6 x 6 Timoshenko stiffness
and 6 x 6 mass matrix. It is built from a ``BladeGeometry`` (one ``ThinWalledSection``
cut per station) or from sections you already have, and it writes

- the BeamDyn primary input file (geometry, mesh and solver options, outputs) and the
  blade input file (stations, damping, the stiffness and mass matrices), in the format
  of OpenFAST 5 (``version=5``, the default, with its modal damping block) or of
  OpenFAST 3 and 4 (``version=4``);
- the windIO 2.0 blade fields WISDEM reads for the same model: ``reference_axis``,
  ``outer_shape.twist`` and ``structure.elastic_properties``.

It also reads both, so a reference model (the IEA 15 MW windIO file, an OpenFAST
BeamDyn deck) can be compared station by station or rewritten.

Axes
----
BeamDyn's blade frame (``X_r`` towards the suction side, ``Y_r`` towards the trailing
edge, ``Z_r`` from root to tip, right-handed) is the frame of every array here. The
stiffness and mass are in each station's local frame, which BeamDyn turns by the
initial twist: chord-aligned, about the reference axis. WISDEM writes windIO 2.0's
``elastic_properties`` into BeamDyn as they are (the IEA 15 MW windIO file and its
BeamDyn deck carry the same numbers), so the windIO fields here are the same arrays.

The package's public frame (``axes``: x towards the trailing edge, y towards the
suction side, z along the span, the geometry kernel's and shell-fem's blade frame) is
a mirror image of BeamDyn's, so going from a section's axes to BeamDyn's swaps x and y
and flips the sign of the moments and rotations (they are pseudo-vectors): ``MIRROR``,
the same mirror as ``publish_frame`` in example 03. The public twist is right-handed
about z, positive towards feather, and is the same number as BeamDyn's and windIO's
twist. The kernel's ``SectionCut.twist`` is its negative (it moves the leading edge
towards the suction side), so ``from_geometry`` converts it. The kernel's offset (sweep
along x, prebend along y) becomes the key points' ``kp_yr`` and ``kp_xr``.

Sections are cut normal to the z axis, as PreComp's and the windIO files' are; BeamDyn
takes them normal to its curved reference axis, which differs by the slope of the
prebend and sweep (second order in the slope).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from .axes import MIRROR, rotation6, to_beamdyn_frame, twist_from_kernel  # noqa: F401
from .thin_wall import ThinWalledSection

PathLike = Union[str, os.PathLike]

# MIRROR, rotation6 and to_beamdyn_frame are defined in ``axes`` and re-exported here


def section_matrices(section: ThinWalledSection, reference=(0.0, 0.0), twist: float = 0.0):
    """BeamDyn-frame (stiffness, mass) of one section about ``reference`` (its pitch axis)."""
    return (to_beamdyn_frame(section.timoshenko_stiffness(reference), twist),
            to_beamdyn_frame(section.mass_matrix(reference), twist))


def _arc_eta(points: np.ndarray) -> np.ndarray:
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    return s / s[-1]


def _symmetric(A: np.ndarray, name: str, rtol: float = 1e-8) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    scale = np.max(np.abs(A), axis=(-2, -1), keepdims=True)
    if np.any(np.abs(A - np.swapaxes(A, -1, -2)) > rtol * np.maximum(scale, 1e-300)):
        raise ValueError(f"{name} matrices must be symmetric")
    return 0.5 * (A + np.swapaxes(A, -1, -2))


class BeamDynBlade:
    """A blade's beam model in BeamDyn's frame: key points, twist, stations and matrices.

    Parameters
    ----------
    eta:
        (n,) station positions along the reference axis, 0 at the root and 1 at the
        tip (arc-length fractions).
    stiffness, mass:
        (n, 6, 6) Timoshenko stiffness and mass per unit length at the stations, in
        BeamDyn's order and local frame ([F_x, F_y, N, M_x, M_y, M_z] against
        [gamma_x, gamma_y, eps, kappa_x, kappa_y, kappa_z]; [u, theta] for the mass).
    key_points:
        (k, 3) reference axis ``kp_xr, kp_yr, kp_zr`` (m), k >= 3, root first.
    twist:
        (k,) initial twist at the key points (degrees: BeamDyn's and windIO's sign, which
        is the public one).
    name:
        Title written on the files' second line.
    """

    def __init__(self, eta, stiffness, mass, key_points, twist=None, name: str = "",
                 sections: Optional[Sequence[ThinWalledSection]] = None):
        self.eta = np.asarray(eta, dtype=float).ravel()
        n = len(self.eta)
        self.stiffness = _symmetric(np.asarray(stiffness, dtype=float).reshape(n, 6, 6), "stiffness")
        self.mass = _symmetric(np.asarray(mass, dtype=float).reshape(n, 6, 6), "mass")
        self.key_points = np.asarray(key_points, dtype=float).reshape(-1, 3)
        k = len(self.key_points)
        self.twist = np.zeros(k) if twist is None else np.asarray(twist, dtype=float).ravel()
        if len(self.twist) != k:
            raise ValueError("twist needs one value per key point")
        if n < 2 or abs(self.eta[0]) > 1e-9 or abs(self.eta[-1] - 1.0) > 1e-9 or np.any(np.diff(self.eta) <= 0.0):
            raise ValueError("station eta must rise from 0 (root) to 1 (tip)")
        self.eta[0], self.eta[-1] = 0.0, 1.0
        if k < 3:
            raise ValueError("BeamDyn needs at least three key points")
        self.name = name
        self.sections = None if sections is None else list(sections)

    # ------------------------------------------------------------ construction
    @classmethod
    def from_sections(cls, sections: Sequence[ThinWalledSection], z, twist=0.0, offset=(0.0, 0.0),
                      name: str = "") -> "BeamDynBlade":
        """A blade from sections cut at span positions ``z`` (root first).

        Each section is in the public frame at its station: turned by ``twist`` (degrees,
        right-handed about z, as BeamDyn's and windIO's twist; the kernel's
        ``SectionCut.twist`` is the negative, ``axes.twist_from_kernel``) and with its pitch
        axis at ``offset`` (x, y), where the stiffness and mass are taken. ``twist`` and
        ``offset`` are one value for all stations or one per station. The key points
        are the stations' pitch-axis points, measured from the root's.
        """
        z = np.asarray(z, dtype=float).ravel()
        n = len(z)
        if len(sections) != n:
            raise ValueError("one section per span position")
        if n < 3:
            raise ValueError("BeamDyn needs at least three stations here (they are also the key points)")
        if np.any(np.diff(z) <= 0.0):
            raise ValueError("span positions must increase from root to tip")
        tw = np.broadcast_to(np.asarray(twist, dtype=float), (n,))
        off = np.broadcast_to(np.asarray(offset, dtype=float), (n, 2))
        K, M = zip(*(section_matrices(s, off[i], tw[i]) for i, s in enumerate(sections)))
        kp = np.column_stack([off[:, 1] - off[0, 1], off[:, 0] - off[0, 0], z - z[0]])  # from the root
        return cls(_arc_eta(kp), np.array(K), np.array(M), kp, tw.copy(), name=name, sections=sections)

    @classmethod
    def from_geometry(cls, geometry, laminates: Union[Mapping, Callable[[float], Mapping]], z=None,
                      name: str = "", **section_options) -> "BeamDynBlade":
        """Sweep a ``BladeGeometry``: one ``ThinWalledSection.from_geometry`` cut per station.

        ``laminates`` maps wall labels to laminates as ``from_geometry`` takes them, or
        is a function of z returning that mapping (for layups that change along the
        span). ``z`` defaults to the geometry's stations. Other keywords go to
        ``ThinWalledSection.from_geometry`` (``thickness_tol``, ``wall_bending``).
        """
        if z is None:
            z = [s.z for s in geometry.stations]
        z = [float(v) for v in z]
        cuts = [geometry.section_at(v) for v in z]
        lams = laminates if callable(laminates) else (lambda _z: laminates)
        sections = [ThinWalledSection.from_geometry(c, lams(v), **section_options) for c, v in zip(cuts, z)]
        return cls.from_sections(sections, z, twist=twist_from_kernel([c.twist for c in cuts]), offset=[c.offset for c in cuts],
                                 name=name)

    # ------------------------------------------------------------ derived values
    @property
    def length(self) -> float:
        """Length of the reference axis through the key points (straight segments)."""
        return float(np.sum(np.linalg.norm(np.diff(self.key_points, axis=0), axis=1)))

    @property
    def key_point_eta(self) -> np.ndarray:
        return _arc_eta(self.key_points)

    def station_values(self) -> Dict[str, np.ndarray]:
        """Named values at the stations: the windIO inertia fields and the stiffness diagonal."""
        K, M = self.stiffness, self.mass
        m = M[:, 0, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_x = np.where(m > 0, M[:, 1, 5] / m, 0.0)
            cm_y = np.where(m > 0, -M[:, 0, 5] / m, 0.0)
        return {"mass": m, "cm_x": cm_x, "cm_y": cm_y, "i_edge": M[:, 3, 3], "i_flap": M[:, 4, 4],
                "i_plr": M[:, 5, 5], "i_cp": -M[:, 3, 4], "K11": K[:, 0, 0], "K22": K[:, 1, 1], "EA": K[:, 2, 2],
                "EI_edge": K[:, 3, 3], "EI_flap": K[:, 4, 4], "GJ": K[:, 5, 5]}

    # ------------------------------------------------------------ BeamDyn files
    def write_blade(self, path: PathLike, damping: Optional[Sequence[float]] = None,
                    modal_damping: Optional[Sequence[float]] = None, version: int = 5) -> Path:
        """Write the BeamDyn blade input file.

        ``damping`` is the six stiffness-proportional coefficients mu1..mu6
        (``damp_type`` 1); ``modal_damping`` the modal ratios zeta (``damp_type`` 2,
        OpenFAST 5 on); neither gives an undamped blade. ``version`` is the OpenFAST
        major version that reads the file: 5 writes the modal damping block, which
        OpenFAST 5 requires and OpenFAST 3 and 4 (``version=4``) do not read.
        """
        if damping is not None and modal_damping is not None:
            raise ValueError("give stiffness-proportional or modal damping, not both")
        if version not in (3, 4, 5):
            raise ValueError("version is the OpenFAST major version: 3, 4 or 5")
        if modal_damping is not None and version < 5:
            raise ValueError("modal damping needs OpenFAST 5 (version=5)")
        mu = np.zeros(6) if damping is None else np.asarray(damping, dtype=float).ravel()
        if len(mu) != 6:
            raise ValueError("damping needs six coefficients mu1..mu6")
        zeta = [] if modal_damping is None else [float(v) for v in modal_damping]
        damp_type = 1 if damping is not None else (2 if modal_damping is not None else 0)
        f = _fmt
        lines = ["------- BEAMDYN V1.00.* INDIVIDUAL BLADE INPUT FILE --------------------------",
                 self.name or "Blade properties from anisotropic-section-model",
                 "------ Blade Parameters --------------------------------------------------------",
                 f"{len(self.eta):<23d} station_total    - Number of blade input stations (-)",
                 f"{damp_type:<23d} damp_type        - Damping type (switch) {{0: none, 1: stiffness-proportional, "
                 f"2: modal}}",
                 "------ Stiffness-Proportional Damping [used only if damp_type=1] ---------------",
                 "   mu1        mu2        mu3        mu4        mu5        mu6",
                 "   (-)        (-)        (-)        (-)        (-)        (-)",
                 " ".join(f(v) for v in mu)]
        if version >= 5:
            lines += ["------ Modal Damping [used only if damp_type=2] --------------------------------",
                      f"{len(zeta):<23d} n_modes    - Number of modal damping coefficients (-)",
                      (" ".join(f(v) for v in zeta) if zeta else "0") + "    zeta       - Damping coefficients "
                      "for mode 1 through n_modes"]
        lines.append("------ Distributed Properties --------------------------------------------------")
        for i, e in enumerate(self.eta):
            if i:
                lines.append("")
            lines.append(f(e))
            lines += [" ".join(f(v) for v in row) for row in self.stiffness[i]]
            lines.append("")
            lines += [" ".join(f(v) for v in row) for row in self.mass[i]]
        path = Path(path)
        path.write_text("\n".join(lines) + "\n")
        return path

    def write(self, path: PathLike, blade_file: Optional[PathLike] = None, order_elem: Optional[int] = None,
              quadrature: int = 2, refine: int = 1, damping: Optional[Sequence[float]] = None,
              modal_damping: Optional[Sequence[float]] = None, version: int = 5,
              outputs: Sequence[str] = ("RootFxr, RootFyr, RootFzr", "RootMxr, RootMyr, RootMzr",
                                        "TipTDxr, TipTDyr, TipTDzr", "TipRDxr, TipRDyr, TipRDzr"),
              quasi_static_init: bool = False) -> Path:
        """Write the BeamDyn primary input file and, beside it, the blade input file.

        ``blade_file`` defaults to ``<stem>_blade.dat`` in the same folder; the primary
        file refers to it by its path relative to the primary file. One member through
        all the key points, interpolated at ``order_elem`` (default: one less than the
        number of key points, at most 10). ``quadrature`` 2 (trapezoidal, the default)
        integrates between the stations, refined ``refine`` times; 1 is Gauss. Damping
        and ``version`` are ``write_blade``'s; for OpenFAST 3 and 4 (``version=4``) the
        primary file also has the pitch actuator block that OpenFAST 5 removed.
        """
        path = Path(path)
        blade = path.with_name(path.stem + "_blade.dat") if blade_file is None else Path(blade_file)
        if not blade.is_absolute() and blade.parent == Path("."):
            blade = path.parent / blade
        self.write_blade(blade, damping=damping, modal_damping=modal_damping, version=version)
        rel = os.path.relpath(blade, path.parent)
        k = len(self.key_points)
        order = min(10, k - 1) if order_elem is None else int(order_elem)
        if quadrature not in (1, 2):
            raise ValueError("quadrature is 1 (Gauss) or 2 (trapezoidal)")
        tf = lambda b: "True " if b else "False"  # noqa: E731
        lines = ["--------- BEAMDYN with OpenFAST INPUT FILE -------------------------------------------",
                 self.name or "Blade properties from anisotropic-section-model",
                 "---------------------- SIMULATION CONTROL --------------------------------------",
                 "False         Echo             - Echo input data to \"<RootName>.ech\" (flag)",
                 f"{tf(quasi_static_init)}         QuasiStaticInit  - Use quasistatic pre-conditioning with "
                 "centripetal accelerations in initialization (flag) [dynamic solve only]",
                 "          0   rhoinf           - Numerical damping parameter for generalized-alpha integrator",
                 f"{quadrature:>11d}   quadrature       - Quadrature method: 1=Gaussian; 2=Trapezoidal (switch)",
                 f"{refine:>11d}   refine           - Refinement factor for trapezoidal quadrature (-) "
                 "[DEFAULT = 1; used only when quadrature=2]",
                 "\"DEFAULT\"     n_fact           - Factorization frequency for the Jacobian in N-R iteration(-) "
                 "[DEFAULT = 5]",
                 "\"DEFAULT\"     DTBeam           - Time step size (s)",
                 "\"DEFAULT\"     load_retries     - Number of factored load retries before quitting the "
                 "simulation [DEFAULT = 20]",
                 "\"DEFAULT\"     NRMax            - Max number of iterations in Newton-Raphson algorithm (-) "
                 "[DEFAULT = 10]",
                 "\"DEFAULT\"     stop_tol         - Tolerance for stopping criterion (-) [DEFAULT = 1E-5]",
                 "\"DEFAULT\"     tngt_stf_fd      - Use finite differenced tangent stiffness matrix? (flag)",
                 "\"DEFAULT\"     tngt_stf_comp    - Compare analytical finite differenced tangent stiffness "
                 "matrix? (flag)",
                 "\"DEFAULT\"     tngt_stf_pert    - Perturbation size for finite differencing (-) "
                 "[DEFAULT = 1E-6]",
                 "\"DEFAULT\"     tngt_stf_difftol - Maximum allowable relative difference between analytical "
                 "and fd tangent stiffness (-); [DEFAULT = 0.1]",
                 "True          RotStates        - Orient states in the rotating frame during linearization? "
                 "(flag) [used only when linearizing]",
                 "---------------------- GEOMETRY PARAMETER --------------------------------------",
                 "          1   member_total    - Total number of members (-)",
                 f"{k:>11d}   kp_total        - Total number of key points (-) [must be at least 3]",
                 f"     1 {k:>6d}                 - Member number; Number of key points in this member",
                 "       kp_xr                    kp_yr                    kp_zr                initial_twist",
                 "        (m)                      (m)                      (m)                    (deg)"]
        lines += [" ".join(_fmt(v) for v in (*p, t)) for p, t in zip(self.key_points, self.twist)]
        lines += ["---------------------- MESH PARAMETER ------------------------------------------",
                  f"{order:>11d}   order_elem     - Order of interpolation (basis) function (-)",
                  "---------------------- MATERIAL PARAMETER --------------------------------------",
                  f"\"{rel}\"    BldFile - Name of file containing properties for blade (quoted string)"]
        if version < 5:  # OpenFAST 5 dropped BeamDyn's pitch actuator
            lines += ["---------------------- PITCH ACTUATOR PARAMETERS -------------------------------",
                      "False         UsePitchAct - Whether a pitch actuator should be used (flag)",
                      "        200   PitchJ      - Pitch actuator inertia (kg-m^2) [used only when UsePitchAct is "
                      "true]",
                      "      2E+07   PitchK      - Pitch actuator stiffness (kg-m^2/s^2) [used only when UsePitchAct "
                      "is true]",
                      "     500000   PitchC      - Pitch actuator damping (kg-m^2/s) [used only when UsePitchAct is "
                      "true]"]
        lines += ["---------------------- OUTPUTS -------------------------------------------------",
                  "False         SumPrint       - Print summary data to \"<RootName>.sum\" (flag)",
                  "\"ES16.8E2\"    OutFmt         - Format used for text tabular output, excluding the time channel.",
                  "          0   NNodeOuts      - Number of nodes to output to file [0 - 9] (-)",
                  "          1   OutNd          - Nodes whose values will be output  (-)",
                  "          OutList            - The next line(s) contains a list of output parameters. See "
                  "OutListParameters.xlsx for a listing of available output channels, (-)"]
        lines += [f"\"{o}\"" for o in outputs]
        lines += ["END of input file (the word \"END\" must appear in the first 3 columns of this last OutList "
                  "line)",
                  "---------------------------------------------------------------------------------------"]
        path.write_text("\n".join(lines) + "\n")
        return path

    @classmethod
    def read_blade(cls, path: PathLike, key_points=None, twist=None) -> "BeamDynBlade":
        """Read a BeamDyn blade input file (OpenFAST 3 to 5).

        A blade file has no reference axis: ``key_points`` default to a straight unit
        axis through the stations. The damping read is kept in ``damping``,
        ``modal_damping`` and ``damp_type``.
        """
        lines = Path(path).read_text().splitlines()
        n = int(lines[3].split()[0])
        damp_type = int(lines[4].split()[0])
        i_mu = next(i for i, l in enumerate(lines) if re.match(r"\s*mu1\b", l))
        mu = [float(v) for v in lines[i_mu + 2].split()[:6]]
        zeta = []
        i_modes = next((i for i, l in enumerate(lines) if re.search(r"\bn_modes\b", l)), None)
        if i_modes is not None:
            nm = int(lines[i_modes].split()[0])
            zeta = [float(v) for v in lines[i_modes + 1].split()[:nm]]
        i_dp = next(i for i, l in enumerate(lines) if "distributed properties" in l.lower())
        values = np.array([float(v) for l in lines[i_dp + 1:] for v in l.split()])
        if len(values) < n * 73:
            raise ValueError(f"{path}: expected {n} stations of 73 values")
        data = values[: n * 73].reshape(n, 73)
        eta = data[:, 0]
        K = data[:, 1:37].reshape(n, 6, 6)
        M = data[:, 37:].reshape(n, 6, 6)
        if key_points is None:
            key_points = np.column_stack([np.zeros(3), np.zeros(3), [0.0, 0.5, 1.0]])
        blade = cls(eta, K, M, key_points, twist, name=lines[1].strip())
        blade.damp_type, blade.damping, blade.modal_damping = damp_type, mu, zeta
        return blade

    @classmethod
    def read(cls, path: PathLike) -> "BeamDynBlade":
        """Read a BeamDyn primary input file and the blade file it names."""
        path = Path(path)
        lines = path.read_text().splitlines()
        i = next(k for k, l in enumerate(lines) if re.search(r"\bmember_total\b", l))
        n_members = int(lines[i].split()[0])
        k = i + 2
        points, twist = [], []
        for _ in range(n_members):
            n_kp = int(lines[k].split()[1])
            rows = [[float(v) for v in lines[k + 3 + j].split()[:4]] for j in range(n_kp)]
            if points:  # consecutive members share their end key point
                rows = rows[1:]
            points += [r[:3] for r in rows]
            twist += [r[3] for r in rows]
            k += 3 + n_kp
        bld = next(l for l in lines if re.search(r"\bBldFile\b", l))
        name = re.search(r"\"([^\"]+)\"|'([^']+)'", bld)
        name = (name.group(1) or name.group(2)) if name else bld.split()[0]
        blade = cls.read_blade(path.parent / name, points, twist)
        blade.name = lines[1].strip()
        return blade

    # ------------------------------------------------------------ windIO
    def to_windio(self, structural_damping: Optional[Mapping[str, Sequence]] = None) -> dict:
        """The windIO 2.0 blade fields of this beam model, as a nested dict.

        ``reference_axis`` (x, y, z on the key points' arc-length grid),
        ``outer_shape.twist`` and ``structure.elastic_properties``: the 21 stiffness
        entries K11..K66 and the inertia fields (mass, cm_x, cm_y, i_edge, i_flap,
        i_plr, i_cp) on the stations' grid. These are BeamDyn's numbers, as WISDEM
        passes them through. windIO also requires ``structural_damping``
        (``{"mode_name": [...], "damping_ratio": [...]}``); it is included if given.
        Merge the result into a turbine's ``components.blade``.
        """
        g = self.key_point_eta.tolist()
        grid = self.eta.tolist()
        stiffness = {"grid": grid}
        for i in range(6):
            for j in range(i, 6):
                stiffness[f"K{i + 1}{j + 1}"] = self.stiffness[:, i, j].tolist()
        v = self.station_values()
        inertia = {"grid": grid, **{k: v[k].tolist() for k in ("mass", "cm_x", "cm_y", "i_edge", "i_flap", "i_plr",
                                                                "i_cp")}}
        elastic = {"stiffness_matrix": stiffness, "inertia_matrix": inertia}
        if structural_damping is not None:
            elastic["structural_damping"] = {k: list(val) for k, val in structural_damping.items()}
        kp = self.key_points
        return {
            "reference_axis": {ax: {"grid": g, "values": kp[:, c].tolist()} for c, ax in enumerate("xyz")},
            "outer_shape": {"twist": {"grid": g, "values": self.twist.tolist()}},
            "structure": {"elastic_properties": elastic},
        }

    def write_windio(self, path: PathLike, **options) -> Path:
        """Write ``{"components": {"blade": to_windio()}}`` as YAML (needs PyYAML), or JSON
        for a ``.json`` path."""
        doc = {"components": {"blade": self.to_windio(**options)}}
        path = Path(path)
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(doc, indent=1))
            return path
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("writing windIO YAML needs PyYAML (pip install pyyaml), or use a .json path") from exc
        path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=1 << 16))
        return path

    @classmethod
    def from_windio(cls, blade, n_key_points: Optional[int] = None, name: str = "") -> "BeamDynBlade":
        """A beam model from a windIO 2.0 blade (a ``components.blade`` dict, or a turbine
        YAML/JSON file).

        Stations are the elastic properties' grid. Key points are the reference axis
        grid (or ``n_key_points`` evenly spaced in arc length), with x, y, z and the
        twist interpolated onto it. Missing stiffness entries are zero.
        """
        if not isinstance(blade, Mapping):
            p = Path(blade)
            text = p.read_text()
            if p.suffix.lower() == ".json":
                doc = json.loads(text)
            else:
                import yaml
                doc = yaml.safe_load(text)
            blade = doc["components"]["blade"]
            name = name or str(doc.get("name", ""))
        ep = blade["structure"]["elastic_properties"] if "structure" in blade else blade["elastic_properties"]
        sm, im = ep["stiffness_matrix"], ep["inertia_matrix"]
        eta = np.asarray(sm["grid"], dtype=float)
        n = len(eta)
        K = np.zeros((n, 6, 6))
        for i in range(6):
            for j in range(i, 6):
                key = f"K{i + 1}{j + 1}"
                if key in sm:
                    K[:, i, j] = K[:, j, i] = sm[key]
        ig = np.asarray(im["grid"], dtype=float)
        at = lambda key, default=0.0: np.interp(eta, ig, np.asarray(im.get(key, [default] * len(ig)), float))  # noqa
        m, xc, yc = at("mass"), at("cm_x"), at("cm_y")
        ie, iflap, icp = at("i_edge"), at("i_flap"), at("i_cp")
        ip = at("i_plr") if "i_plr" in im else ie + iflap
        M = np.zeros((n, 6, 6))
        M[:, 0, 0] = M[:, 1, 1] = M[:, 2, 2] = m
        M[:, 0, 5] = M[:, 5, 0] = -m * yc
        M[:, 1, 5] = M[:, 5, 1] = m * xc
        M[:, 2, 3] = M[:, 3, 2] = m * yc
        M[:, 2, 4] = M[:, 4, 2] = -m * xc
        M[:, 3, 3], M[:, 4, 4], M[:, 5, 5] = ie, iflap, ip
        M[:, 3, 4] = M[:, 4, 3] = -icp
        ra = blade["reference_axis"]
        if n_key_points is None:
            g = np.asarray(ra["z"]["grid"], dtype=float)
        else:
            g = np.linspace(0.0, 1.0, int(n_key_points))
        kp = np.column_stack([np.interp(g, ra[a]["grid"], ra[a]["values"]) for a in "xyz"])
        kp -= kp[0]
        tw = blade["outer_shape"]["twist"]
        twist = np.interp(g, tw["grid"], tw["values"])  # windIO 2.0: degrees
        return cls(eta, K, M, kp, twist, name=name)


def _fmt(v: float) -> str:
    return f"{float(v): .16e}"


__all__ = ["BeamDynBlade", "MIRROR", "rotation6", "section_matrices", "to_beamdyn_frame"]
