# pre_processing/section_integration/solid/element_recovery.py
"""3D stresses over a 2D-meshed section at a point of a beam element.

The element gives its section state at ``xi`` from its nodal displacements:

- **Euler-Bernoulli:** the classical section forces ``[N, M_y, M_z, T]`` from the element
  curvatures and the shear forces from beam equilibrium (``V_z = M_y'``, ``V_y = -M_z'``);
- **Timoshenko:** ``F = K psi`` from the element strains;
- **Benscoter:** the eight restrained-warping strains ``[psi6, d, psi']``.

``SolidSectionAnalysis`` then gives the 3D strains and stresses over the section
(``PointStress``), with the forces varying along the beam as with no distributed load, so
the transverse shear and interlaminar stresses of the bending gradient are included.
Element strains are element-signed (``kappa_y = -theta_y'``) and are mapped to the
right-handed section convention with ``S = diag(1, -1, 1, 1, ...)``.

Public API
----------
element_section_state, element_section_stresses.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from pre_processing.section_integration.thin_walled.element_recovery import _FD, _beam_strains, _kind, _sign


def element_section_state(element, U_e: npt.ArrayLike, xi: float, analysis
                          ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """``(forces, benscoter)`` at ``xi``: right-handed section forces (6,) for Euler-Bernoulli and
    Timoshenko elements, or right-handed Benscoter strains (8,) for Benscoter elements (the other
    entry ``None``)."""
    from pre_processing.element_library.framed_element import local_state

    element, U, xi = local_state(element, np.asarray(U_e, dtype=np.float64), xi)
    kind = _kind(element)
    if kind == "Timoshenko":
        return analysis.stiffness @ (_sign(6) * _beam_strains(element, U, xi)), None
    if kind == "EulerBernoulli":
        S = _sign(6)

        def classical(x):
            return analysis.classical_stiffness @ (S * _beam_strains(element, U, x))[[0, 1, 2, 5]]

        h = _FD
        c = classical(xi)
        dc = (classical(xi + h) - classical(xi - h)) / (2 * h) * (2.0 / element.L)
        return np.array([c[0], c[1], c[2], -dc[2], dc[1], c[3]]), None
    if kind == "Benscoter":
        B, _, _ = element._B_at(xi)
        return None, _sign(8) * (B @ U)
    raise NotImplementedError(f"section stress recovery on 2D sections does not support {type(element).__name__}")


def element_section_stresses(element, U_e: npt.ArrayLike, xi: float, analysis) -> List:
    """``PointStress`` at every quadrature point of the section at ``xi`` of an element."""
    forces, benscoter = element_section_state(element, U_e, xi, analysis)
    return analysis.point_stresses(forces=forces, benscoter=benscoter)
