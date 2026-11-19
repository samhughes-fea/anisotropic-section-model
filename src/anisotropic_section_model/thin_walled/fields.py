"""Formulation fields of a thin-walled section at section, wall, Gauss and ply levels.

Warping, shear flow and wall strains live on the analysis Gauss points in memory.
``FormulationWriter`` returns them as arrays, with ply stresses recovered at those
same points, and writes one CSV per level.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from anisotropic_section_model.thin_walled.section_analysis import (
    T_RATE,
    ThinWalledSectionAnalysis,
    wall_law,
)
from anisotropic_section_model.thin_walled.stress_recovery import SaintVenantRecovery

FORMULATION_CSV_SCHEMA = "anisotropic-section-model/formulation@1"

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SectionField:
    """Section-level formulation: stiffness, centres, warping constant, geometric terms."""

    stiffness: Optional[Array]  # (6, 6), None when the walls cannot shear
    classical_stiffness: Array  # (4, 4) [N, M_y, M_z, T]
    shear_centre: Array  # (2,)
    tension_centre: Array  # (2,)
    warping_constant: float
    geometric_terms: Array  # (7, 6)
    n_walls: int
    n_gauss: int
    n_points: int
    hoop: str
    n_sub: int


@dataclass(frozen=True)
class WallField:
    """One row per wall: geometry and the condensed wall law."""

    wall_ids: np.ndarray
    laminate_ids: np.ndarray
    start: np.ndarray
    end: np.ndarray
    length: np.ndarray
    thickness: np.ndarray
    n_gauss: np.ndarray
    K: np.ndarray  # (n_walls, 4, 4) condensed wall stiffness, order WALL_STRAINS


@dataclass(frozen=True)
class GaussField:
    """Warping, shear flow and wall strains at every section Gauss point."""

    wall_ids: np.ndarray
    s: Array  # distance along the wall from its start
    position: Array  # (n, 2) (y, z)
    weight: Array
    normal: Array  # (n, 2)
    omega: Array  # torsional warping about the shear centre
    omega_s: Array  # d omega / ds
    wall_strains: Array  # (n, 4) order WALL_STRAINS, for the writer's forces
    Nxx: Array
    Nxs: Array  # membrane shear flow
    Mxx: Array
    Qx: Array  # Kirchhoff transverse shear from the force rate
    K: np.ndarray  # (n, 4, 4) condensed wall stiffness at the point


@dataclass(frozen=True)
class PlyField:
    """Through-thickness ply strains and stresses at the Gauss points."""

    wall_ids: np.ndarray
    s: Array
    ply: np.ndarray
    n: Array  # through-thickness coordinate
    strain_wall: Array  # (m, 3) [eps_xx, eps_ss, gamma_xs]
    stress_wall: Array  # (m, 3) [sigma_xx, sigma_ss, tau_xs]
    strain_ply: Array
    stress_ply: Array
    tau_xn: Array
    tau_sn: Array
    sigma_nn: Array
    tau_13: Array
    tau_23: Array


class FormulationWriter:
    """Section, wall, Gauss and ply fields of a ``ThinWalledSectionAnalysis``.

    Warping is independent of the section forces. Wall strains, shear flow and ply
    stresses are for ``forces`` (engine order ``FORCES``, ``[N, M_y, M_z, V_y, V_z, T]``;
    ``from_section`` takes a public ``ThinWalledSection`` and forces in the public order).
    Default forces are zero.
    """

    def __init__(
        self,
        analysis: ThinWalledSectionAnalysis,
        forces: Optional[npt.ArrayLike] = None,
        points_per_ply: int = 3,
    ) -> None:
        self.analysis = analysis
        self.forces = np.zeros(6, dtype=np.float64) if forces is None else np.asarray(forces, dtype=np.float64)
        if self.forces.shape != (6,):
            raise ValueError(f"forces must have shape (6,), got {self.forces.shape}")
        if points_per_ply < 2:
            raise ValueError(f"points_per_ply must be at least 2, got {points_per_ply}")
        self.points_per_ply = int(points_per_ply)
        self._section: Optional[SectionField] = None
        self._walls: Optional[WallField] = None
        self._gauss: Optional[GaussField] = None
        self._ply: Optional[PlyField] = None

    @classmethod
    def from_section(cls, section, forces: Optional[npt.ArrayLike] = None,
                     points_per_ply: int = 3) -> "FormulationWriter":
        """The writer of a public ``ThinWalledSection`` (its ``analysis``), with ``forces`` in
        the public order ``[Fx, Fy, N, Mx, My, Mz]`` about the section origin
        (``axes.FORCES``). The fields it writes are the engine's (``axes``: positions
        ``(y, z)`` are the public ``(x, y)``, and stiffnesses are in ``ENGINE_STRAINS``).
        """
        from anisotropic_section_model.axes import to_engine

        f = None if forces is None else to_engine(np.asarray(forces, dtype=np.float64))
        return cls(section.analysis, f, points_per_ply)

    def _gauss_rows(self):
        an = self.analysis
        strains = an.wall_strains(self.forces)
        s_all = np.concatenate([an.gauss_s(w.wall_id) for w in an.section.walls])
        for g, (wall_id, position, e), s in zip(an._gauss_points, strains, s_all):
            yield g, int(wall_id), np.asarray(position, dtype=np.float64), np.asarray(e, dtype=np.float64), float(s)

    def section(self) -> SectionField:
        if self._section is not None:
            return self._section
        an = self.analysis
        try:
            stiffness = an.stiffness.copy()
        except ValueError:
            stiffness = None
        self._section = SectionField(
            stiffness=stiffness,
            classical_stiffness=an.classical_stiffness.copy(),
            shear_centre=np.array(an.shear_centre, dtype=np.float64),
            tension_centre=np.array(an.tension_centre, dtype=np.float64),
            warping_constant=float(an.warping_constant),
            geometric_terms=an.geometric_terms.copy(),
            n_walls=len(an.section.walls),
            n_gauss=len(an._gauss_points),
            n_points=len(an.section.points),
            hoop=an.hoop,
            n_sub=an.n_sub,
        )
        return self._section

    def walls(self) -> WallField:
        if self._walls is not None:
            return self._walls
        an = self.analysis
        sec = an.section
        n_g = {w.wall_id: an.gauss_s(w.wall_id).size for w in sec.walls}
        ids, lids, start, end, length, thickness, n_gauss, K = [], [], [], [], [], [], [], []
        for w in sec.walls:
            lam = an.laminates[w.laminate_id]
            ids.append(w.wall_id)
            lids.append(w.laminate_id)
            start.append(w.start)
            end.append(w.end)
            length.append(sec.length(w))
            thickness.append(lam.thickness)
            n_gauss.append(n_g[w.wall_id])
            K.append(wall_law(lam, an.hoop))
        self._walls = WallField(
            np.array(ids, dtype=int), np.array(lids, dtype=int),
            np.array(start, dtype=int), np.array(end, dtype=int),
            np.array(length, dtype=np.float64), np.array(thickness, dtype=np.float64),
            np.array(n_gauss, dtype=int), np.stack(K),
        )
        return self._walls

    def gauss(self) -> GaussField:
        if self._gauss is not None:
            return self._gauss
        an = self.analysis
        f_rate = T_RATE @ self.forces
        wall_ids, s, pos, wgt, nrm = [], [], [], [], []
        omega, omega_s, strains, Nxx, Nxs, Mxx, Qx, K = [], [], [], [], [], [], [], []
        for g, wall_id, position, e, si in self._gauss_rows():
            om, om_s = an._omega_at(g)
            resultants = g.K @ e
            e_rate = an.wall_strains_at(wall_id, si, f_rate)
            wall_ids.append(wall_id)
            s.append(si)
            pos.append(position)
            wgt.append(g.weight)
            nrm.append(np.array(g.normal, dtype=np.float64))
            omega.append(om)
            omega_s.append(om_s)
            strains.append(e)
            Nxx.append(resultants[0])
            Nxs.append(resultants[1])
            Mxx.append(resultants[2])
            Qx.append((g.K @ e_rate)[2])
            K.append(g.K)
        self._gauss = GaussField(
            np.array(wall_ids, dtype=int), np.array(s), np.stack(pos), np.array(wgt),
            np.stack(nrm), np.array(omega), np.array(omega_s), np.stack(strains),
            np.array(Nxx), np.array(Nxs), np.array(Mxx), np.array(Qx), np.stack(K),
        )
        return self._gauss

    def ply(self) -> PlyField:
        if self._ply is not None:
            return self._ply
        rec = SaintVenantRecovery(self.analysis)
        wall_ids, s, ply, n = [], [], [], []
        e_w, s_w, e_p, s_p = [], [], [], []
        txn, tsn, snn, t13, t23 = [], [], [], [], []
        for _, wall_id, _, _, si in self._gauss_rows():
            state = rec.at(wall_id, si, self.forces, self.points_per_ply)
            for point in state.plies:
                wall_ids.append(wall_id)
                s.append(si)
                ply.append(point.ply)
                n.append(point.n)
                e_w.append(point.strain_wall)
                s_w.append(point.stress_wall)
                e_p.append(point.strain_ply)
                s_p.append(point.stress_ply)
                txn.append(point.tau_xn)
                tsn.append(point.tau_sn)
                snn.append(point.sigma_nn)
                t13.append(point.tau_13)
                t23.append(point.tau_23)
        self._ply = PlyField(
            np.array(wall_ids, dtype=int), np.array(s), np.array(ply, dtype=int), np.array(n),
            np.stack(e_w), np.stack(s_w), np.stack(e_p), np.stack(s_p),
            np.array(txn), np.array(tsn), np.array(snn), np.array(t13), np.array(t23),
        )
        return self._ply

    @staticmethod
    def _write_table(path: Path, columns: Dict[str, np.ndarray]) -> Path:
        names, data = [], []
        for name, values in columns.items():
            values = np.asarray(values)
            if values.ndim == 1:
                names.append(name)
                data.append(values)
            else:
                flat = values.reshape(values.shape[0], -1)
                for c in range(flat.shape[1]):
                    names.append(f"{name}_{c}")
                    data.append(flat[:, c])
        with path.open("w", newline="") as handle:
            handle.write(f"# schema={FORMULATION_CSV_SCHEMA}\n")
            writer = csv.writer(handle)
            writer.writerow(names)
            writer.writerows(zip(*data))
        return path

    @staticmethod
    def _csv_columns(obj) -> Dict[str, np.ndarray]:
        columns = {}
        for item in fields(obj):
            values = np.asarray(getattr(obj, item.name))
            if values.dtype == object or values.ndim == 0:
                continue
            columns[item.name] = values
        return columns

    def to_csv(self, directory) -> Dict[str, Path]:
        """Write section, wall, Gauss and ply CSVs in ``directory``; returns {level: path}."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sec = self.section()
        rows: List[Tuple[str, object]] = [
            ("n_walls", sec.n_walls),
            ("n_gauss", sec.n_gauss),
            ("n_points", sec.n_points),
            ("n_sub", sec.n_sub),
            ("hoop", sec.hoop),
            ("warping_constant", sec.warping_constant),
            ("shear_centre_y", sec.shear_centre[0]),
            ("shear_centre_z", sec.shear_centre[1]),
            ("tension_centre_y", sec.tension_centre[0]),
            ("tension_centre_z", sec.tension_centre[1]),
        ]
        K4 = sec.classical_stiffness.ravel()
        rows += [(f"classical_stiffness_{i}", v) for i, v in enumerate(K4)]
        if sec.stiffness is not None:
            rows += [(f"stiffness_{i}", v) for i, v in enumerate(sec.stiffness.ravel())]
        path = directory / "section.csv"
        with path.open("w", newline="") as handle:
            handle.write(f"# schema={FORMULATION_CSV_SCHEMA}\n")
            csv.writer(handle).writerows([("quantity", "value"), *rows])
        out = {"section": path}
        for level, obj in (("walls", self.walls()), ("gauss", self.gauss()), ("ply", self.ply())):
            out[level] = self._write_table(directory / f"{level}.csv", self._csv_columns(obj))
        return out
