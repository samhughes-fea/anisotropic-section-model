"""Ply stresses and failure exposures of one beam section under its section forces.

``SectionStrength`` takes a section (thin-walled or meshed in 2D), the strength of every
ply material, and a set of section forces, and gives the 3D ply stresses at sampling
points over the section, every failure criterion of ``ply_failure`` (maximum stress,
Tsai-Wu, Hashin, Puck) at each point, and the critical point: which wall or element, which
ply, where in the section and through the thickness, by which criterion and mode.

It is a section-level check only: it knows nothing about where the forces come from.
Running it along a beam (forces from a beam solve, envelopes over load cases) is 1d-fem's.

Forces
------
Section forces are in this package's beam order (``TimoshenkoBeam``,
``ThinWalledSection.timoshenko_stiffness``, BeamDyn): ``[Fx, Fy, N, Mx, My, Mz]`` in the
section axes ``(x, y)``, beam axis ``z``, about ``reference`` (the section origin by
default). ``order="analysis"`` takes them in the section analyses' own order,
``[N, M_y, M_z, V_y, V_z, T]`` about the origin, with their beam axis ``x`` and section
``(y, z)`` being ``z`` and ``(x, y)`` here.

Recovery
--------
- **Thin-walled** (``from_thin_walled``): the coupled Saint-Venant wall strains of
  ``thin_walled.ThinWalledSectionAnalysis``, the released wall resultants set to zero, and
  the ply stresses of each wall laminate through the thickness
  (``thin_walled.stress_recovery``), with the interlaminar stresses from 3D equilibrium.
  ``points_per_wall`` points along every wall (ends included) and ``points_per_ply``
  heights through every ply (bottom, ..., top).
- **Solid** (``from_solid``): the 3D stresses of ``solid.SolidSectionAnalysis`` in material
  axes at every quadrature point of the mesh.

Both are Saint-Venant states: the forces vary along the beam as with no distributed load
(``M_x' = -F_y``, ``M_y' = F_x``), which sets the transverse shear and the interlaminar
stresses. A distributed load adds only the small ``V'`` part of the interlaminar stresses.

Stresses are linear in the forces, so each section is solved once for six unit forces and
any number of load cases cost a matrix product; only the failure criteria are evaluated
per case. Ply stresses are in ply axes, ``ply_failure`` order
``[sigma_11, sigma_22, sigma_33, tau_23, tau_13, tau_12]``.

Public API
----------
SectionStrength, SectionStrengthResult, CriticalPoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from anisotropic_section_model.ply_failure import CRITERIA, PlyStrength, exposures

__all__ = ["SectionStrength", "SectionStrengthResult", "CriticalPoint", "ANALYSIS_FROM_BEAM", "to_analysis_forces"]

Array = npt.NDArray[np.float64]

#: positions of the section analyses' forces [N, M_y, M_z, V_y, V_z, T] in the beam order
#: [Fx, Fy, N, Mx, My, Mz] (the inverse of ``thin_wall.BEAMDYN_FROM_ONEDFEM``)
ANALYSIS_FROM_BEAM = [2, 3, 4, 0, 1, 5]

StrengthSpec = Union[PlyStrength, Mapping, Callable, None]


def to_analysis_forces(forces: npt.ArrayLike, reference: Sequence[float] = (0.0, 0.0)) -> Array:
    """Beam-order section forces ``(..., 6)`` about ``reference`` to the analysis order about the origin."""
    f = np.array(forces, dtype=np.float64, copy=True)
    x, y = float(reference[0]), float(reference[1])
    if x or y:
        f[..., 3] += y * f[..., 2]
        f[..., 4] -= x * f[..., 2]
        f[..., 5] += x * f[..., 1] - y * f[..., 0]
    return f[..., ANALYSIS_FROM_BEAM]


@dataclass(frozen=True)
class CriticalPoint:
    """The most exposed sampling point of a section for one criterion (or all of them).

    ``wall`` is the wall index (thin-walled) or ``None``; ``element`` the mesh element
    (solid) or ``None``. ``name`` is the wall's name, when it has one. ``position`` is
    ``(x, y)`` in the section axes, ``n`` the height through the wall (thin-walled).
    ``criterion`` is the governing criterion and ``mode`` its mode. ``stress`` is the ply
    stress there, ply axes.
    """

    exposure: float
    criterion: str
    mode: str
    index: int
    ply: int
    wall: Optional[int]
    element: Optional[int]
    name: str
    position: Tuple[float, float]
    n: float
    stress: Array

    @property
    def reserve_factor(self) -> float:
        return float(1.0 / self.exposure) if self.exposure > 0 else float("inf")

    @property
    def margin(self) -> float:
        return self.reserve_factor - 1.0

    def as_dict(self) -> Dict[str, object]:
        return {"exposure": self.exposure, "reserve_factor": self.reserve_factor, "criterion": self.criterion,
                "mode": self.mode, "ply": self.ply, "wall": self.wall, "element": self.element, "name": self.name,
                "position": list(self.position), "n": self.n, "stress": self.stress.tolist()}


@dataclass
class SectionStrengthResult:
    """Ply stresses and exposures at every sampling point of a section for one load case.

    Attributes
    ----------
    stress : ndarray (m, 6)
        Ply-axis stresses at the sampling points.
    exposures : dict
        ``ply_failure.exposures`` output per point: ``"max_stress"``, ``"tsai_wu"``,
        ``"hashin"``, ``"puck"`` exposures, their ``*_mode`` and ``"exposure"`` (the largest).
    """

    check: "SectionStrength"
    forces: Array
    stress: Array
    exposures: Dict[str, np.ndarray]

    def exposure(self, criterion: str = "exposure") -> Array:
        """Exposure per point for ``criterion`` (one of ``ply_failure.CRITERIA``, or the envelope)."""
        return np.asarray(self.exposures[criterion], dtype=np.float64)

    def critical(self, criterion: str = "exposure", where: Optional[np.ndarray] = None) -> CriticalPoint:
        """The most exposed point, over all points or the boolean selection ``where``."""
        f = self.exposure(criterion).copy()
        if where is not None:
            f[~np.asarray(where, dtype=bool)] = -np.inf
        i = int(np.argmax(f))
        if criterion == "exposure":
            governing = max(CRITERIA, key=lambda c: float(self.exposures[c][i]))
        else:
            governing = criterion
        mode_key = f"{governing}_mode"
        mode = str(self.exposures[mode_key][i]) if mode_key in self.exposures else ""
        c = self.check
        wall = None if c.wall[i] < 0 else int(c.wall[i])
        element = None if c.element[i] < 0 else int(c.element[i])
        return CriticalPoint(float(f[i]), governing, mode, i, int(c.ply[i]), wall, element, c.names[i],
                             (float(c.position[i, 0]), float(c.position[i, 1])), float(c.n[i]),
                             self.stress[i].copy())

    def by_wall(self, criterion: str = "exposure") -> Dict[int, float]:
        """Largest exposure on every wall (thin-walled)."""
        f = self.exposure(criterion)
        return {int(w): float(f[self.check.wall == w].max()) for w in np.unique(self.check.wall) if w >= 0}

    def by_name(self, criterion: str = "exposure") -> Dict[str, float]:
        """Largest exposure on every named part (wall names such as ``spar_cap_ss``)."""
        f = self.exposure(criterion)
        names = np.asarray(self.check.names)
        return {str(k): float(f[names == k].max()) for k in dict.fromkeys(self.check.names)}


def _strength_lookup(spec: StrengthSpec, default: Callable[[object], Optional[PlyStrength]]):
    if spec is None:
        return default
    if isinstance(spec, PlyStrength):
        return lambda key: spec
    if isinstance(spec, Mapping):
        return lambda key: spec.get(key)
    if callable(spec):
        return spec
    raise TypeError("strengths must be a PlyStrength, a mapping, a callable or None")


def _strength_of_card(material) -> Optional[PlyStrength]:
    strength = getattr(material, "strength", None)
    if strength is None:
        return None
    return PlyStrength.from_strength_properties(strength)


class SectionStrength:
    """Ply stresses and failure exposures of one section (build with ``from_thin_walled``
    or ``from_solid``; then ``evaluate`` a load case).

    Attributes (one entry per sampling point)
    ----------------------------------------
    wall, element : ndarray of int
        Wall index (thin-walled) or mesh element (solid); -1 where not applicable.
    ply : ndarray of int
        Ply index from the bottom of its laminate (-1 for an untagged solid element).
    position : ndarray (m, 2)
        ``(x, y)`` in the section axes.
    n : ndarray
        Height through the wall from its reference line (thin-walled; 0 for solid).
    names : list of str
        Wall names (thin-walled) or ``"material <id>"`` (solid).
    """

    def __init__(self, influence: Array, strengths: Sequence[PlyStrength], wall, element, ply, position, n, names,
                 kind: str):
        self.influence = np.asarray(influence, dtype=np.float64)   # (m, 6 stresses, 6 analysis forces)
        self.wall = np.asarray(wall, dtype=np.int64)
        self.element = np.asarray(element, dtype=np.int64)
        self.ply = np.asarray(ply, dtype=np.int64)
        self.position = np.asarray(position, dtype=np.float64).reshape(-1, 2)
        self.n = np.asarray(n, dtype=np.float64)
        self.names = list(names)
        self.kind = kind
        self._strengths = list(strengths)
        # group the points by strength so each criterion runs once per material
        groups: Dict[int, List[int]] = {}
        uniq: List[PlyStrength] = []
        for i, st in enumerate(self._strengths):
            for k, u in enumerate(uniq):
                if u is st or u == st:
                    break
            else:
                k = len(uniq)
                uniq.append(st)
            groups.setdefault(k, []).append(i)
        self._groups = [(uniq[k], np.asarray(idx, dtype=np.int64)) for k, idx in groups.items()]

    @property
    def n_points(self) -> int:
        return self.influence.shape[0]

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_thin_walled(cls, section, strengths: StrengthSpec = None, points_per_wall: int = 5,
                         points_per_ply: int = 3, walls: Optional[Sequence[int]] = None,
                         laminates: Optional[Mapping[int, object]] = None) -> "SectionStrength":
        """Sampling points over the walls of a thin-walled section.

        Parameters
        ----------
        section : ThinWalledSection or thin_walled.ThinWalledSectionAnalysis
            A ``ThinWalledSection`` (its coupled ``analysis`` is used; wall ``k`` is wall
            index ``k`` and keeps its name) or a section analysis.
        strengths : PlyStrength, mapping, callable or None
            The strength of each ply. ``None`` takes it from the laminate-constitutive-model
            ply material's ``strength`` (``ThinWalledSection`` walls are LCM laminates). A
            mapping or callable is keyed by the ply's material: the LCM material card for a
            ``ThinWalledSection``, the ``PlyMaterial`` for an analysis. Plies with no
            strength are left out.
        points_per_wall : int
            Points along each wall, both ends included.
        points_per_ply : int
            Heights through each ply, bottom and top included.
        walls : sequence of int, optional
            Only these walls.
        """
        from anisotropic_section_model.thin_walled.stress_recovery import wall_point_stress
        from anisotropic_section_model.thin_walled.section_analysis import T_RATE

        analysis = getattr(section, "analysis", section)
        cards: Dict[int, Sequence] = {}
        names: Dict[int, str] = {}
        if hasattr(section, "analysis"):
            for k, w in enumerate(section.walls):
                cards[k] = [p.material for p in w.laminate.plies]
                names[k] = w.name
        lookup = _strength_lookup(strengths, _strength_of_card)
        geometry = analysis.section
        wall_ids = [w.wall_id for w in geometry.walls] if walls is None else list(walls)
        npw = max(int(points_per_wall), 1)

        rows: List[Array] = []
        meta = {"wall": [], "ply": [], "position": [], "n": [], "names": [], "strength": []}
        for wid in wall_ids:
            w = geometry.wall(wid)
            lam = analysis.laminates[w.laminate_id]
            L = geometry.length(w)
            ends = analysis.element_ends(wid)
            ply_strength = []
            for k, ply in enumerate(lam.plies):
                key = cards[wid][k] if wid in cards else ply.material
                ply_strength.append(lookup(key))
            if all(st is None for st in ply_strength):
                continue
            s_values = [0.5 * L] if npw == 1 else np.linspace(0.0, L, npw)
            for s in s_values:
                j = int(np.clip(np.searchsorted(ends, s, side="right") - 1, 0, len(ends) - 2))
                span = (float(ends[j]), float(ends[j + 1]))
                pos = analysis.point_at(wid, float(s)).position
                cols = []
                for i in range(6):
                    f = np.zeros(6)
                    f[i] = 1.0
                    fx = T_RATE @ f
                    field = (lambda ss, f=f, fx=fx: (analysis.wall_strains_at(wid, ss, f),
                                                     analysis.wall_strains_at(wid, ss, fx)))
                    st = wall_point_stress(lam, field, wid, float(s), pos, span, analysis.hoop, points_per_ply)
                    cols.append(np.array([[p.stress_ply[0], p.stress_ply[1], p.sigma_nn, p.tau_23, p.tau_13,
                                           p.stress_ply[2]] for p in st.plies]))
                    plies = st.plies
                block = np.stack(cols, axis=-1)  # (points through the wall, 6, 6)
                for r, p in enumerate(plies):
                    if ply_strength[p.ply] is None:
                        continue
                    rows.append(block[r])
                    meta["wall"].append(wid)
                    meta["ply"].append(p.ply)
                    meta["position"].append(pos)
                    meta["n"].append(p.n)
                    meta["names"].append(names.get(wid, f"wall {wid}"))
                    meta["strength"].append(ply_strength[p.ply])
        if not rows:
            raise ValueError("no ply of the section has a strength")
        m = len(rows)
        return cls(np.array(rows), meta["strength"], meta["wall"], -np.ones(m, dtype=np.int64), meta["ply"],
                   np.array(meta["position"]), meta["n"], meta["names"], "thin_walled")

    @classmethod
    def from_solid(cls, analysis, strengths: StrengthSpec) -> "SectionStrength":
        """Sampling points at the quadrature points of a ``solid.SolidSectionAnalysis``.

        ``strengths`` is one of:

        - a ``PlyStrength`` for every element;
        - a mapping from the mesh material id to a ``PlyStrength``, or, on a mesh whose
          elements are tagged with their ply (``SectionMesh.from_thin_walled``), from
          ``(laminate_id, ply)`` (``ply_strengths`` builds it for a ``ThinWalledSection``);
          elements it does not cover, such as a core, are left out;
        - a callable ``(element) -> PlyStrength or None``.

        The analysis' section ``(y, z)`` are ``(x, y)`` here.
        """
        mesh = analysis.mesh
        tags = mesh.plies

        def key_of(e):
            return tags[e] if tags is not None and tags[e] is not None else None

        if isinstance(strengths, PlyStrength):
            per_element = lambda e: strengths  # noqa: E731
        elif isinstance(strengths, Mapping):
            def per_element(e):
                tag = key_of(e)
                if tag is not None and tag in strengths:
                    return strengths[tag]
                return strengths.get(mesh.material_ids[e])
        elif callable(strengths):
            per_element = strengths
        else:
            raise TypeError("strengths must be a PlyStrength, a mapping or a callable")
        cols = []
        for i in range(6):
            f = np.zeros(6)
            f[i] = 1.0
            pts = analysis.point_stresses(forces=f)
            cols.append(np.array([p.stress_material for p in pts]))
        block = np.stack(cols, axis=-1)
        keep, st_list, elem, ply, pos, names = [], [], [], [], [], []
        for r, p in enumerate(pts):
            st = per_element(p.element)
            if st is None:
                continue
            tag = key_of(p.element)
            keep.append(r)
            st_list.append(st)
            elem.append(p.element)
            ply.append(-1 if tag is None else tag[1])
            pos.append((p.y, p.z))
            names.append(f"laminate {tag[0]}" if tag is not None else f"material {mesh.material_ids[p.element]}")
        if not keep:
            raise ValueError("no element of the mesh has a strength")
        m = len(keep)
        return cls(block[keep], st_list, -np.ones(m, dtype=np.int64), elem, ply, pos, np.zeros(m), names, "solid")

    @staticmethod
    def ply_strengths(section) -> Dict[Tuple[int, int], PlyStrength]:
        """``{(laminate_id, ply): PlyStrength}`` of a ``ThinWalledSection``'s coupled analysis,
        from the strengths of its laminate-constitutive-model plies: the keys of the ply tags
        of ``SectionMesh.from_thin_walled(section.analysis.section, section.analysis.laminates)``."""
        geometry = section.analysis.section
        out: Dict[Tuple[int, int], PlyStrength] = {}
        for k, w in enumerate(section.walls):
            lid = geometry.wall(k).laminate_id
            for j, p in enumerate(w.laminate.plies):
                st = _strength_of_card(p.material)
                if st is not None:
                    out[(lid, j)] = st
        return out

    # ------------------------------------------------------------------ evaluation
    def stresses(self, forces: npt.ArrayLike, reference: Sequence[float] = (0.0, 0.0),
                 order: str = "beam") -> Array:
        """Ply-axis stresses (m, 6) at every sampling point for one set of section forces."""
        f = self._analysis_forces(forces, reference, order)
        return self.influence @ f

    def evaluate(self, forces: npt.ArrayLike, reference: Sequence[float] = (0.0, 0.0),
                 order: str = "beam") -> SectionStrengthResult:
        """Stresses and every criterion's exposure for section forces (6,)."""
        f = self._analysis_forces(forces, reference, order)
        s = self.influence @ f
        out: Dict[str, np.ndarray] = {}
        for st, idx in self._groups:
            ex = exposures(s[idx], st)
            for key, val in ex.items():
                val = np.asarray(val)
                if key not in out:
                    out[key] = np.empty(self.n_points, dtype=val.dtype if val.dtype.kind != "U" else object)
                out[key][idx] = val
        return SectionStrengthResult(self, f, s, out)

    @staticmethod
    def _analysis_forces(forces, reference, order) -> Array:
        f = np.asarray(forces, dtype=np.float64)
        if f.shape != (6,):
            raise ValueError(f"section forces must have 6 entries, got shape {f.shape}")
        if order == "beam":
            return to_analysis_forces(f, reference)
        if order == "analysis":
            if tuple(reference) != (0.0, 0.0):
                raise ValueError("order='analysis' forces are about the section origin")
            return f.copy()
        raise ValueError(f"order must be 'beam' or 'analysis', got {order!r}")
