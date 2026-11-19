"""Read and write BECAS's 2D input files (``N2D.in``, ``E2D.in``, ``EMAT.in``, ``MATPROPS.in``)
as ASM ``SectionMesh`` objects, and read BECAS's stiffness output.

This is our own reader and writer of BECAS's documented file formats (BECAS user guide,
section "Input"); no BECAS code is used or copied. BECAS itself is licensed and is never
part of this repository: point ``BECAS_HOME`` at a local copy to run the cases that need it.

Formats
-------
- ``N2D.in``: ``node, x, y`` per row.
- ``E2D.in``: ``element, n1 .. n8`` per row; Q4 rows have zeros in ``n5 .. n8``. Corners
  first, then the mid-side nodes, ``n(4 + k)`` between corners ``k`` and ``k + 1`` (as ASM).
- ``EMAT.in``: ``element, material, fibre angle, fibre plane angle`` (degrees).
- ``MATPROPS.in``: ``E1 E2 E3 G12 G13 G23 nu12 nu13 nu23 rho`` per material, row ``i`` being
  material ``i``.

Axes and angles
---------------
BECAS's beam axis is ``z`` and its section axes are ``(x, y)``; ASM's beam axis is ``x``
and its section axes ``(y, z)``, so BECAS ``(x, y)`` is ASM ``(y, z)`` unchanged. With both
angles zero, BECAS's material axes are ``1 = z`` (beam), ``2 = x``, ``3 = y``, which is ASM's
``theta = alpha = 0`` (``e1 = beam``, ``e2 = s = y``, ``e3 = n = z``). The fibre plane angle
turns the wall direction about the beam axis from ``x`` toward ``y``, ASM's ``alpha``; the
fibre angle turns the fibre in that plane from the beam axis toward the wall direction,
ASM's ``theta``. ``THETA_SIGN`` and ``ALPHA_SIGN`` hold the signs; they are fixed by the
data, not chosen: ``run.py`` checks them on BECAS's composite pipe (W3) and three-cell (T2)
meshes, whose couplings change sign if either is wrong.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from anisotropic_section_model.solid import SectionMesh, SolidMaterial
from anisotropic_section_model.solid.elements import quadrature, shape

THETA_SIGN = 1.0
ALPHA_SIGN = 1.0


def _rows(path: Path) -> List[List[float]]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith(("%", "#")):
            out.append([float(v) for v in line.split()])
    return out


def _ccw(mesh_nodes: np.ndarray, conn: List[int], kind: str) -> List[int]:
    """The element's nodes counter-clockwise (BECAS accepts either orientation; ASM needs
    a positive Jacobian). Reversing keeps corner ``k`` / mid-side ``k`` pairing."""
    X = mesh_nodes[conn]
    _, dN = shape(kind, *quadrature(kind)[0][0])
    if np.linalg.det(dN @ X) > 0.0:
        return conn
    if kind == "Q4":
        return [conn[0], conn[3], conn[2], conn[1]]
    c, m = conn[:4], conn[4:]
    return [c[0], c[3], c[2], c[1], m[3], m[2], m[1], m[0]]


def read_becas(folder, theta_sign: float = None, alpha_sign: float = None) -> SectionMesh:
    """The BECAS input in ``folder`` as an ASM ``SectionMesh``."""
    ts = THETA_SIGN if theta_sign is None else theta_sign
    as_ = ALPHA_SIGN if alpha_sign is None else alpha_sign
    folder = Path(folder)
    nrows = _rows(folder / "N2D.in")
    index = {int(r[0]): i for i, r in enumerate(nrows)}
    nodes = np.array([[r[1], r[2]] for r in nrows])
    emat = {int(r[0]): (int(r[1]), r[2], r[3]) for r in _rows(folder / "EMAT.in")}
    props = _rows(folder / "MATPROPS.in")
    materials = {i + 1: SolidMaterial(*p[:10]) for i, p in enumerate(props)}
    elements, kinds, mids, theta, alpha, labels = [], [], [], [], [], []
    for r in _rows(folder / "E2D.in"):
        label = int(r[0])
        ids = [int(v) for v in r[1:] if int(v) != 0]
        kind = {4: "Q4", 8: "Q8", 6: "T6"}[len(ids)]
        conn = _ccw(nodes, [index[i] for i in ids], kind)
        m, fib, plane = emat[label]
        elements.append(conn)
        kinds.append(kind)
        mids.append(m)
        theta.append(ts * fib)
        alpha.append(as_ * plane)
        labels.append(label)
    used = {m: materials[m] for m in set(mids)}
    mesh = SectionMesh(nodes, elements, kinds, mids, theta, alpha, used)
    mesh.becas_labels = labels  # type: ignore[attr-defined]
    return mesh


def write_becas(mesh: SectionMesh, folder, theta_sign: float = None, alpha_sign: float = None) -> Path:
    """Write ``mesh`` as BECAS input in ``folder`` (created). Materials are numbered in the
    order of their ASM ids; nodes and elements from 1."""
    ts = THETA_SIGN if theta_sign is None else theta_sign
    as_ = ALPHA_SIGN if alpha_sign is None else alpha_sign
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ids = sorted(mesh.materials)
    number = {m: i + 1 for i, m in enumerate(ids)}
    with open(folder / "N2D.in", "w") as f:
        for i, (y, z) in enumerate(mesh.nodes):
            f.write(f"{i + 1:d} {y:.17g} {z:.17g}\n")
    with open(folder / "E2D.in", "w") as f:
        for e, conn in enumerate(mesh.elements):
            if mesh.kinds[e] == "T6":
                raise ValueError("BECAS's 2D input takes Q4 and Q8 rows here; write T6 meshes as Q8")
            row = [int(n) + 1 for n in conn] + [0] * (8 - len(conn))
            f.write(f"{e + 1:d} " + " ".join(str(v) for v in row) + "\n")
    with open(folder / "EMAT.in", "w") as f:
        for e in range(len(mesh.elements)):
            # ASM's alpha = 90 +- 360 k describes the same wall; BECAS takes any angle
            f.write(f"{e + 1:d} {number[mesh.material_ids[e]]:d} "
                    f"{ts * mesh.theta[e]:.17g} {as_ * mesh.alpha[e]:.17g}\n")
    with open(folder / "MATPROPS.in", "w") as f:
        for m in ids:
            p = mesh.materials[m]
            vals = (p.E1, p.E2, p.E3, p.G12, p.G13, p.G23, p.nu12, p.nu13, p.nu23, p.rho if p.rho > 0 else 1.0)
            f.write(" ".join(f"{v:.17g}" for v in vals) + "\n")
    return folder


def read_matrix_block(text: str, name: str) -> Optional[np.ndarray]:
    """The 6 x 6 block printed after a line ``name=`` (BECAS's ``BECASresults.txt`` and the
    ``becas_out.txt`` that ``run_becas.m`` writes)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == f"{name}=":
            rows = []
            for row in lines[i + 1:]:
                vals = row.split()
                if len(vals) != 6:
                    if rows:
                        break
                    continue
                rows.append([float(v) for v in vals])
                if len(rows) == 6:
                    return np.array(rows)
    return None


def read_value(text: str, name: str) -> Optional[float]:
    for line in text.splitlines():
        s = line.strip().replace(" ", "")
        if s.startswith(name + "="):
            try:
                return float(s.split("=", 1)[1].rstrip(";"))
            except ValueError:
                return None
    return None


def read_becas_results(path) -> Dict[str, object]:
    """``K`` (BECAS order), shear and tension (elastic) centres from a BECAS results text file."""
    text = Path(path).read_text()
    K = read_matrix_block(text, "K")
    return dict(K=None if K is None else K.tolist(),
                shear_centre=(read_value(text, "ShearX"), read_value(text, "ShearY")),
                tension_centre=(read_value(text, "ElasticX"), read_value(text, "ElasticY")))


def _fortran_float(v: str) -> float:
    """A Fortran real, including the exponent-without-``E`` form VABS prints for tiny
    values (``2.51-321``)."""
    try:
        return float(v)
    except ValueError:
        m = re.fullmatch(r"([+-]?\d*\.?\d+)([+-]\d+)", v)
        if m is None:
            raise
        return float(f"{m.group(1)}e{m.group(2)}")


def read_vabs_timoshenko(path) -> Optional[np.ndarray]:
    """VABS's Timoshenko 6 x 6 (``[F1, F2, F3, M1, M2, M3]``) from a ``.K`` output file."""
    lines = Path(path).read_text().splitlines()
    for i, line in enumerate(lines):
        if "Timoshenko Stiffness Matrix" in line:
            rows = []
            for row in lines[i + 1:]:
                vals = row.split()
                if len(vals) == 6:
                    try:
                        rows.append([_fortran_float(v) for v in vals])
                    except ValueError:
                        continue
                    if len(rows) == 6:
                        return np.array(rows)
    return None


def read_vabs_point(path, title: str) -> Optional[Tuple[float, float]]:
    """``(X2, X3)`` printed under the VABS heading containing ``title`` (e.g. "Shear Center")."""
    lines = Path(path).read_text().splitlines()
    for i, line in enumerate(lines):
        if title in line:
            vals = []
            for row in lines[i + 1:i + 8]:
                m = re.match(r"^\s*X\w+\s*=\s*(\S+)", row)
                if m:
                    vals.append(_fortran_float(m.group(1)))
                if len(vals) == 2:
                    return vals[0], vals[1]
    return None


def becas_home() -> Optional[Path]:
    """The local BECAS folder (env ``BECAS_HOME``), or ``None``."""
    import os
    p = os.environ.get("BECAS_HOME")
    return Path(p) if p and Path(p).is_dir() else None


__all__: Sequence[str] = ("THETA_SIGN", "ALPHA_SIGN", "read_becas", "write_becas", "read_becas_results",
                          "read_vabs_timoshenko", "read_vabs_point", "becas_home")
