# pre_processing/section_integration/solid/elements.py
"""Quadratic 2D section elements: 6-node triangles (``T6``) and 8-node quadrilaterals (``Q8``).

Node order follows Gmsh: corners first (counter-clockwise), then the mid-side nodes, the
mid-side node ``k`` lying between corners ``k`` and ``k + 1``.

Public API
----------
ELEMENT_TYPES, n_nodes, shape, quadrature.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt

ELEMENT_TYPES: Dict[str, int] = {"T6": 6, "Q8": 8}


def n_nodes(kind: str) -> int:
    try:
        return ELEMENT_TYPES[kind]
    except KeyError:
        raise ValueError(f"unknown section element type {kind!r}; use one of {sorted(ELEMENT_TYPES)}") from None


def shape(kind: str, xi: float, eta: float) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Shape functions ``N`` (n,) and their natural derivatives ``dN`` (2, n) at ``(xi, eta)``."""
    if kind == "Q8":
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        N = np.zeros(8)
        dN = np.zeros((2, 8))
        for a, (xa, ya) in enumerate(corners):
            t = (1 + xa * xi) * (1 + ya * eta)
            N[a] = 0.25 * t * (xa * xi + ya * eta - 1)
            dN[0, a] = 0.25 * xa * (1 + ya * eta) * (2 * xa * xi + ya * eta)
            dN[1, a] = 0.25 * ya * (1 + xa * xi) * (xa * xi + 2 * ya * eta)
        # mid-side nodes 4..7 at (0,-1), (1,0), (0,1), (-1,0)
        N[4] = 0.5 * (1 - xi * xi) * (1 - eta)
        N[6] = 0.5 * (1 - xi * xi) * (1 + eta)
        N[5] = 0.5 * (1 + xi) * (1 - eta * eta)
        N[7] = 0.5 * (1 - xi) * (1 - eta * eta)
        dN[:, 4] = [-xi * (1 - eta), -0.5 * (1 - xi * xi)]
        dN[:, 6] = [-xi * (1 + eta), 0.5 * (1 - xi * xi)]
        dN[:, 5] = [0.5 * (1 - eta * eta), -eta * (1 + xi)]
        dN[:, 7] = [-0.5 * (1 - eta * eta), -eta * (1 - xi)]
        return N, dN
    if kind == "T6":
        L1, L2 = xi, eta
        L0 = 1.0 - L1 - L2
        N = np.array([L0 * (2 * L0 - 1), L1 * (2 * L1 - 1), L2 * (2 * L2 - 1),
                      4 * L0 * L1, 4 * L1 * L2, 4 * L2 * L0])
        dN = np.array([
            [-(4 * L0 - 1), 4 * L1 - 1, 0.0, 4 * (L0 - L1), 4 * L2, -4 * L2],
            [-(4 * L0 - 1), 0.0, 4 * L2 - 1, -4 * L1, 4 * L1, 4 * (L0 - L2)],
        ])
        return N, dN
    raise ValueError(f"unknown section element type {kind!r}")


def quadrature(kind: str) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Points ``(m, 2)`` and weights ``(m,)``: 3 x 3 Gauss on ``Q8``, 7-point (degree 5) on ``T6``."""
    if kind == "Q8":
        x, w = np.polynomial.legendre.leggauss(3)
        pts = np.array([[a, b] for a in x for b in x])
        wts = np.array([wa * wb for wa in w for wb in w])
        return pts, wts
    if kind == "T6":
        a1, b1 = 0.059715871789770, 0.470142064105115
        a2, b2 = 0.797426985353087, 0.101286507323456
        pts = np.array([[1 / 3, 1 / 3], [b1, b1], [a1, b1], [b1, a1], [b2, b2], [a2, b2], [b2, a2]])
        wts = 0.5 * np.array([0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506,
                              0.125939180544827, 0.125939180544827, 0.125939180544827])
        return pts, wts
    raise ValueError(f"unknown section element type {kind!r}")
