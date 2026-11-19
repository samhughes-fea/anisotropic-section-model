"""Spanwise strength checks: beam section forces, then ply stresses, then failure exposures.

``SpanwiseStrength`` runs a ``section_strength.SectionStrength`` check at stations along a
beam. The section forces come from a ``TimoshenkoBeam`` solution (``BeamSolution``), from a
callable ``z -> forces`` or from a table of forces per station, so the pipeline is::

    beam = TimoshenkoBeam(K, L).clamp(0.0).add_force(L, tip)        # loads and supports
    forces = beam.solve()                                           # section forces along z
    check = SpanwiseStrength(stations, sections)                    # a SectionStrength per station
    result = check.evaluate(forces, reference=pitch_axis)           # ply stresses, exposures
    result.critical()                                               # station, wall, ply, mode

The section check is the section-level part and knows nothing about the span; this module
only samples forces along it and keeps the envelope. Any section-force source can be used,
for example the section loads of an aeroelastic run.

Section forces of a statically determinate beam (a cantilever blade) follow from
equilibrium alone, so a ``TimoshenkoBeam`` with any section stiffness gives the exact
forces of a tapered blade too. A statically indeterminate beam needs the stiffness that
varies along it.

Public API
----------
SpanwiseStrength, SpanwiseStrengthResult.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from anisotropic_section_model.ply_failure import CRITERIA
from anisotropic_section_model.section_strength import CriticalPoint, SectionStrength, SectionStrengthResult

__all__ = ["SpanwiseStrength", "SpanwiseStrengthResult"]

ForceSource = Union[object, Callable[[float], npt.ArrayLike], npt.ArrayLike]


@dataclass
class SpanwiseStrengthResult:
    """Section checks at every station.

    Attributes
    ----------
    z : ndarray (n,)
        Stations along the beam.
    forces : ndarray (n, 6)
        Section forces used at each station (beam order, about each station's reference).
    sections : list of SectionStrengthResult
        The full section result at each station.
    """

    z: np.ndarray
    forces: np.ndarray
    sections: List[SectionStrengthResult]

    def exposure(self, criterion: str = "exposure") -> np.ndarray:
        """Largest exposure over each station's section, for ``criterion`` or the envelope."""
        return np.array([float(r.exposure(criterion).max()) for r in self.sections])

    def critical_points(self, criterion: str = "exposure") -> List[CriticalPoint]:
        """The critical point of every station."""
        return [r.critical(criterion) for r in self.sections]

    def critical(self, criterion: str = "exposure"):
        """``(z, CriticalPoint)`` of the most exposed point along the whole beam."""
        f = self.exposure(criterion)
        j = int(np.argmax(f))
        return float(self.z[j]), self.sections[j].critical(criterion)

    def by_name(self, criterion: str = "exposure") -> Dict[str, np.ndarray]:
        """Largest exposure of every named part (wall name) at each station."""
        rows = [r.by_name(criterion) for r in self.sections]
        names = list(dict.fromkeys(k for row in rows for k in row))
        return {k: np.array([row.get(k, np.nan) for row in rows]) for k in names}

    def summary(self) -> Dict[str, object]:
        """A JSON-ready summary: the envelope per criterion and the critical point per station."""
        z_c, c = self.critical()
        return {
            "z": self.z.tolist(),
            "forces": self.forces.tolist(),
            "exposure": {k: self.exposure(k).tolist() for k in CRITERIA + ("exposure",)},
            "stations": [p.as_dict() for p in self.critical_points()],
            "critical": {"z": z_c, **c.as_dict()},
        }


class SpanwiseStrength:
    """A section strength check at each of ``stations`` along a beam.

    Parameters
    ----------
    stations : sequence of float
        Positions ``z`` along the beam.
    sections : SectionStrength, sequence of SectionStrength, or callable
        One check for all stations (a prismatic beam), one per station, or
        ``z -> SectionStrength``.
    references : (2,) or (n, 2), optional
        The point of each section the beam's section forces are about (the beam's
        reference axis in the section axes); the section origin by default.
    """

    def __init__(self, stations: Sequence[float], sections, references: Optional[npt.ArrayLike] = None):
        self.z = np.asarray(stations, dtype=np.float64).ravel()
        if self.z.size == 0:
            raise ValueError("give at least one station")
        if isinstance(sections, SectionStrength):
            self.sections = [sections] * self.z.size
        elif callable(sections):
            self.sections = [sections(float(z)) for z in self.z]
        else:
            self.sections = list(sections)
        if len(self.sections) != self.z.size:
            raise ValueError(f"{len(self.sections)} sections for {self.z.size} stations")
        ref = np.zeros((self.z.size, 2)) if references is None else np.asarray(references, dtype=np.float64)
        self.references = np.broadcast_to(ref.reshape(-1, 2), (self.z.size, 2)).copy()

    def section_forces(self, source: ForceSource) -> np.ndarray:
        """Section forces (n, 6) at the stations from a ``BeamSolution``, a callable or a table."""
        if hasattr(source, "section_forces"):
            return np.asarray(source.section_forces(self.z), dtype=np.float64).reshape(-1, 6)
        if callable(source):
            return np.array([np.asarray(source(float(z)), dtype=np.float64) for z in self.z]).reshape(-1, 6)
        f = np.asarray(source, dtype=np.float64)
        if f.shape != (self.z.size, 6):
            raise ValueError(f"a force table needs shape ({self.z.size}, 6), got {f.shape}")
        return f.copy()

    def evaluate(self, source: ForceSource) -> SpanwiseStrengthResult:
        """Ply stresses and exposures at every station for one load case."""
        F = self.section_forces(source)
        results = [sec.evaluate(f, reference=r) for sec, f, r in zip(self.sections, F, self.references)]
        return SpanwiseStrengthResult(self.z.copy(), F, results)

    def envelope(self, sources: Sequence[ForceSource], criterion: str = "exposure") -> Dict[str, np.ndarray]:
        """Largest exposure at each station over several load cases, and the case that gives it."""
        runs = [self.evaluate(s).exposure(criterion) for s in sources]
        E = np.array(runs)
        return {"exposure": E.max(axis=0), "case": E.argmax(axis=0)}
