# pre_processing/section_integration/thin_walled/thin_walled_geometry.py
"""Mid-line geometry of a thin-walled beam section.

A section is a graph: points in the section plane ``(y, z)`` joined by
straight walls. Each wall names the laminate it is made of. The beam axis
``x`` is normal to the section plane.

For a wall from point ``a`` to point ``b``:

- the tangent ``e_s = (b - a) / L`` sets the wall direction ``s``;
- the normal ``e_n = x cross e_s = (-e_s_z, e_s_y)`` is ``e_s`` turned by
  +90 degrees in the ``(y, z)`` plane. Plies are stacked along ``e_n``.

The number of closed cells is the cycle count of the graph,
``walls - points + connected_components``: 0 for an open section, 1 for a
box, 2 for a box with one shear web, and so on.

Public API
----------
Wall, ThinWalledSection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Wall:
    """A straight wall between two section points."""

    wall_id: int
    start: int
    end: int
    laminate_id: int


class ThinWalledSection:
    """Validated mid-line geometry of a thin-walled section.

    Parameters
    ----------
    points : mapping int -> (y, z)
        Section points by id.
    walls : iterable of Wall
    laminate_ids : iterable of int, optional
        If given, every wall's ``laminate_id`` must be in it.
    section_id : int, optional

    Raises
    ------
    ValueError
        Duplicate wall ids, unknown point or laminate ids, zero-length walls,
        two walls joining the same pair of points, or points on no wall.
    """

    def __init__(
        self,
        points: Mapping[int, Tuple[float, float]],
        walls: Iterable[Wall],
        laminate_ids: Optional[Iterable[int]] = None,
        section_id: int = 0,
    ) -> None:
        self.section_id = int(section_id)
        self.points: Dict[int, npt.NDArray[np.float64]] = {
            int(k): np.asarray(v, dtype=np.float64).reshape(2) for k, v in points.items()
        }
        self.walls: Tuple[Wall, ...] = tuple(walls)
        self._validate(None if laminate_ids is None else set(int(i) for i in laminate_ids))

    # ------------------------------------------------------------------ #
    def _validate(self, laminate_ids: Optional[set]) -> None:
        tag = f"section {self.section_id}"
        if not self.walls:
            raise ValueError(f"{tag}: needs at least one wall")
        seen_ids: set = set()
        seen_pairs: set = set()
        used: set = set()
        for w in self.walls:
            if w.wall_id in seen_ids:
                raise ValueError(f"{tag}: duplicate wall_id {w.wall_id}")
            seen_ids.add(w.wall_id)
            for p in (w.start, w.end):
                if p not in self.points:
                    raise ValueError(f"{tag}: wall {w.wall_id} uses unknown point {p}")
            if w.start == w.end:
                raise ValueError(f"{tag}: wall {w.wall_id} starts and ends at point {w.start}")
            pair = frozenset((w.start, w.end))
            if pair in seen_pairs:
                raise ValueError(
                    f"{tag}: more than one wall joins points {w.start} and {w.end}"
                )
            seen_pairs.add(pair)
            if laminate_ids is not None and w.laminate_id not in laminate_ids:
                raise ValueError(f"{tag}: wall {w.wall_id} uses unknown laminate {w.laminate_id}")
            used.update((w.start, w.end))
        unused = sorted(set(self.points) - used)
        if unused:
            raise ValueError(f"{tag}: points {unused} are not on any wall")
        scale = max(1.0, max(float(np.max(np.abs(p))) for p in self.points.values()))
        for w in self.walls:
            if self.length(w) <= 1e-12 * scale:
                raise ValueError(f"{tag}: wall {w.wall_id} has zero length")

    # ------------------------------------------------------------------ #
    def wall(self, wall_id: int) -> Wall:
        for w in self.walls:
            if w.wall_id == wall_id:
                return w
        raise KeyError(wall_id)

    def endpoints(self, wall: Wall) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return self.points[wall.start], self.points[wall.end]

    def length(self, wall: Wall) -> float:
        a, b = self.endpoints(wall)
        return float(np.linalg.norm(b - a))

    def tangent(self, wall: Wall) -> npt.NDArray[np.float64]:
        """Unit wall direction ``e_s`` in ``(y, z)``."""
        a, b = self.endpoints(wall)
        return (b - a) / np.linalg.norm(b - a)

    def normal(self, wall: Wall) -> npt.NDArray[np.float64]:
        """Unit through-thickness direction ``e_n = x cross e_s`` in ``(y, z)``."""
        t = self.tangent(wall)
        return np.array([-t[1], t[0]])

    @property
    def n_components(self) -> int:
        parent = {p: p for p in self.points}

        def find(p: int) -> int:
            while parent[p] != p:
                parent[p] = parent[parent[p]]
                p = parent[p]
            return p

        for w in self.walls:
            ra, rb = find(w.start), find(w.end)
            if ra != rb:
                parent[ra] = rb
        return len({find(p) for p in self.points})

    @property
    def n_cells(self) -> int:
        """Number of closed cells (graph cycle count)."""
        return len(self.walls) - len(self.points) + self.n_components

    @property
    def is_closed(self) -> bool:
        return self.n_cells > 0
