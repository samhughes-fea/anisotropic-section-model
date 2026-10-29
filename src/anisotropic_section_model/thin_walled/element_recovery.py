# pre_processing/section_integration/thin_walled/element_recovery.py
"""Wall and ply stresses at a point of a beam element with a thin-walled section.

Each element type gives the wall strains it carries at ``xi`` from its nodal
displacements, and their rate along the beam:

- **Euler-Bernoulli:** the classical section forces ``[N, M_y, M_z, T]`` from the
  element curvatures, the shear forces from beam equilibrium
  (``V_z = M_y'``, ``V_y = -M_z'``, with ``M'`` from the Hermite curvature), then
  the Saint-Venant wall strains of those forces.
- **Timoshenko:** ``F = K psi`` from the element strains, then the Saint-Venant
  wall strains.
- **Benscoter:** the Saint-Venant wall strains plus the restrained-warping terms
  ``eps_xx += omega psi'`` and ``gamma_xs += omega_s (psi - phi)``.
- **GBT:** the plane-section strains, the warping-mode and the distortional-mode
  strains, including the hoop curvature ``kappa_ss``.
- **Reddy, Levinson:** the Saint-Venant wall strains plus the slaved shear warping,
  ``eps_xx += Psi(s) . chi``.

For the Saint-Venant parts the rate along the beam follows beam equilibrium
with no distributed load (``F' = T F``). The Benscoter warping terms and all GBT
terms take their rate from the element interpolation, so their interlaminar
stresses need a mesh fine enough to resolve the gradients.

Element strains are element-signed (``kappa_y = -theta_y'``); they are mapped to
the right-handed section convention with ``S = diag(1, -1, 1, 1, 1, 1, ...)``.

Public API
----------
element_wall_field, element_wall_stress.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt

from pre_processing.section_integration.thin_walled.stress_recovery import (
    KeptField,
    SaintVenantRecovery,
    WallPointStress,
    wall_point_stress,
)

_FD = 1e-4  # relative step in xi for rates along the element


def _sign(n: int) -> np.ndarray:
    S = np.ones(n)
    S[1] = -1.0
    return S


def _kind(element) -> str:
    name = type(element).__name__
    for key in ("GBT", "Benscoter", "Reddy", "Levinson", "Timoshenko", "EulerBernoulli"):
        if key in name:
            return key
    raise NotImplementedError(f"wall stress recovery does not support {name}")


def _beam_strains(element, U: np.ndarray, xi: float) -> np.ndarray:
    """Element-signed 6 beam strains at ``xi`` (Euler-Bernoulli or Timoshenko operator)."""
    N, dN, d2N = element.shape_function_operator.natural_coordinate_form(np.array([xi]))
    B = element.strain_displacement_operator.physical_coordinate_form(dN, d2N, N) if _kind(element) != "EulerBernoulli" \
        else element.strain_displacement_operator.physical_coordinate_form(dN, d2N)
    return B[0] @ U[:12]


def element_wall_field(element, U_e: npt.ArrayLike, xi: float, analysis, wall_id: int) -> Tuple[KeptField, str]:
    """Kept-strain field ``s -> (strains, x rate)`` along wall ``wall_id`` at ``xi``.

    Returns the field and the hoop condition to complete it with.
    """
    from pre_processing.section_integration.thin_walled.section_analysis import T_RATE

    U = np.asarray(U_e, dtype=np.float64)
    kind = _kind(element)
    if not getattr(element, "has_thin_walled_section", False):
        raise ValueError(f"Element {element.element_id} has no thin-walled section")
    h = _FD

    if kind in ("Timoshenko", "EulerBernoulli"):
        S = _sign(6)
        if kind == "Timoshenko":
            psi = S * _beam_strains(element, U, xi)
            F = analysis.stiffness @ psi
        else:
            def classical(x):
                p = S * _beam_strains(element, U, x)
                return analysis.classical_stiffness @ p[[0, 1, 2, 5]]

            c = classical(xi)
            dc = (classical(xi + h) - classical(xi - h)) / (2 * h) * (2.0 / element.L)
            # [N, M_y, M_z, T]; V_z = M_y', V_y = -M_z'
            F = np.array([c[0], c[1], c[2], -dc[2], dc[1], c[3]])
        rec = SaintVenantRecovery(analysis)
        return rec.kept_field(wall_id, F), analysis.hoop

    if kind in ("Reddy", "Levinson"):
        S = _sign(8)

        def e8(x):
            return S * (element._B_at(x) @ U)

        e = e8(xi)
        ex = (e8(xi + h) - e8(xi - h)) / (2 * h) * (2.0 / element.L)

        def field(s):
            G = analysis.third_order_wall_strains_at(wall_id, s)
            return G @ e, G @ ex

        return field, analysis.hoop

    if kind == "Benscoter":
        S = _sign(8)

        def eps8(x):
            B, _, _ = element._B_at(x)
            return S * (B @ U)

        e8 = eps8(xi)
        e8x = (eps8(xi + h) - eps8(xi - h)) / (2 * h) * (2.0 / element.L)
        # Saint-Venant part: rate from beam equilibrium
        F = analysis.stiffness @ e8[:6]
        e8x[:6] = analysis.flexibility @ (T_RATE @ F)

        def field(s):
            G = analysis._benscoter_wall_strains(analysis.point_at(wall_id, s))
            return G @ e8, G @ e8x

        return field, analysis.hoop

    # GBT
    from pre_processing.section_integration.thin_walled.cross_section_modes import CrossSectionModes

    cm = getattr(analysis, "_cross_section_modes", None)
    if cm is None:
        cm = CrossSectionModes(analysis)
        analysis._cross_section_modes = cm
    m, p = element.n_modes, element.n_shear
    n = 6 + 2 * element.n_warping + 3 * m
    S = _sign(n)

    def eps(x):
        B, _, _ = element._B_at(x)
        return S * (B @ U)

    e = eps(xi)
    ex = (eps(xi + h) - eps(xi - h)) / (2 * h) * (2.0 / element.L)

    def field(s):
        H = cm.wall_strain_map(wall_id, s, m, p)
        return H @ e, H @ ex

    return field, "restrained"


def element_wall_stress(
    element, U_e: npt.ArrayLike, xi: float, analysis, wall_id: int, s: float, points_per_ply: int = 3
) -> WallPointStress:
    """Recovered wall and ply state at ``xi`` of an element and ``s`` along a wall.

    Parameters
    ----------
    element
        An Euler-Bernoulli, Timoshenko, Benscoter or GBT element with a thin-walled section.
    U_e : array
        Element displacement vector, in the element's own DOF order.
    xi : float
        Natural coordinate along the element, in ``[-1, 1]``.
    analysis : ThinWalledSectionAnalysis
        The section analysis behind ``element.section_analysis``.
    wall_id, s : int, float
        Wall and distance along it.
    """
    field, hoop = element_wall_field(element, U_e, xi, analysis, wall_id)
    rec = SaintVenantRecovery(analysis)
    w = analysis.section.wall(wall_id)
    pos = analysis.point_at(wall_id, s).position
    return wall_point_stress(analysis.laminates[w.laminate_id], field, wall_id, s, pos,
                             rec.element_span(wall_id, s), hoop, points_per_ply)
