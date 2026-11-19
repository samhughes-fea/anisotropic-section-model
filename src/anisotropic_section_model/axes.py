"""The package's one axis convention, and every conversion out of it.

Public frame
------------
Every public result of this package (``ThinWalledSection``, ``SolidSection``,
``TimoshenkoBeam``, ``BeamDynBlade``, the closed-form properties) is in one frame, the
geometry kernel's and shell-fem's blade frame:

- ``z`` is the beam axis, from root to tip;
- ``x`` is a section axis towards the trailing edge;
- ``y`` is a section axis towards the suction side (``x``, ``y``, ``z`` right-handed);
- generalised strains and forces are in BeamDyn's order, ``STRAINS`` against ``FORCES``:
  ``[gamma_x, gamma_y, eps, kappa_x, kappa_y, kappa_z]`` against
  ``[Fx, Fy, N, Mx, My, Mz]``, with ``gamma_x = u_x' - theta_y`` and
  ``gamma_y = u_y' + theta_x``. A Vlasov or Benscoter warping degree of freedom is a
  seventh entry (its force is the bimoment);
- twist (``BeamDynBlade.from_sections``, ``BeamDynBlade.twist``) is in degrees,
  right-handed about ``+z`` like every other rotation here: a section twisted by ``t`` is
  the untwisted one turned by ``rotation6(t)``. Positive twist moves the leading edge
  towards the pressure side (towards feather). It is the same number as BeamDyn's
  ``initial_twist`` and windIO 2.0's ``outer_shape.twist``.

Frames at the boundaries
------------------------
Three other frames appear only where the package meets other code. This module holds
the conversion to each, so nothing else spells them out.

- **Engine frame** (``thin_walled``, ``solid``): the section solvers moved from 1d-fem
  work with the beam along ``x`` and section axes ``(y, z)``, strains ``ENGINE_STRAINS``
  = ``[eps, kappa_y, kappa_z, gamma_xy, gamma_xz, phi]``. It is a cyclic relabelling of
  the public frame, engine ``(x, y, z)`` = public ``(z, x, y)``, so section coordinates
  are the same numbers, no sign changes, and 6 x 6 matrices are a reordering
  (``from_engine``, ``to_engine``).
- **1d-fem element order** (``section_analysis.SectionAnalysis``): the engine order with
  the sign of ``kappa_y`` flipped, as 1d-fem's beam elements' ``B`` matrices have it
  (``to_element``).
- **BeamDyn and windIO** (``BeamDynBlade``): BeamDyn's blade frame has ``X`` towards the
  suction side and ``Y`` towards the trailing edge, a mirror image of the public frame.
  Going there swaps ``x`` and ``y`` and flips the sign of moments and rotations
  (``MIRROR``, ``to_beamdyn_frame``). Twist needs no conversion: BeamDyn's and windIO's
  are the public twist.
- **Geometry kernel twist** (``SectionCut.twist``, ``Station.twist``): the kernel's twist
  moves the leading edge towards the suction side, so it is left-handed about ``+z``,
  the negative of the public twist (``twist_from_kernel``, ``twist_to_kernel``). Its
  axes are the public axes. ``BeamDynBlade.from_geometry`` converts it; the kernel's own
  ``write_windio`` does not, so its twist has the opposite sign to windIO's.

Public API
----------
STRAINS, FORCES, DISPLACEMENTS, ENGINE_STRAINS, ENGINE_FORCES, FROM_ENGINE, TO_ENGINE,
from_engine, to_engine, to_element, MIRROR, rotation6, to_beamdyn_frame,
twist_from_kernel, twist_to_kernel.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt

#: public generalised strains (BeamDyn's order)
STRAINS: Tuple[str, ...] = ("gamma_x", "gamma_y", "eps", "kappa_x", "kappa_y", "kappa_z")
#: public generalised forces, work-conjugate to ``STRAINS``
FORCES: Tuple[str, ...] = ("Fx", "Fy", "N", "Mx", "My", "Mz")
#: public displacements and rotations (``TimoshenkoBeam``, the mass matrix)
DISPLACEMENTS: Tuple[str, ...] = ("u_x", "u_y", "u_z", "theta_x", "theta_y", "theta_z")

#: engine strains (``thin_walled``, ``solid``: beam axis x, section axes (y, z))
ENGINE_STRAINS: Tuple[str, ...] = ("eps", "kappa_y", "kappa_z", "gamma_xy", "gamma_xz", "phi")
#: engine forces, work-conjugate to ``ENGINE_STRAINS``
ENGINE_FORCES: Tuple[str, ...] = ("N", "M_y", "M_z", "V_y", "V_z", "T")

#: public entry ``i`` is engine entry ``FROM_ENGINE[i]``
FROM_ENGINE: Tuple[int, ...] = (3, 4, 0, 1, 2, 5)
#: engine entry ``i`` is public entry ``TO_ENGINE[i]``
TO_ENGINE: Tuple[int, ...] = tuple(int(i) for i in np.argsort(FROM_ENGINE))

#: 1d-fem's element sign on the engine order (``kappa_y = -theta_y'``)
ELEMENT_SIGN = np.diag([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])

_SWAP = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
#: public displacement-rotation (or force-moment) vector to BeamDyn's axes: x and y swap,
#: and moments and rotations (pseudo-vectors) change sign under the mirror
MIRROR = np.block([[_SWAP, np.zeros((3, 3))], [np.zeros((3, 3)), -_SWAP]])


def _permute(a: npt.ArrayLike, p) -> npt.NDArray[np.float64]:
    """Reorder the first six entries of a vector, or the first six rows and columns of a
    square matrix, by ``p``; entries past the sixth (a warping DOF) keep their place."""
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[-1]
    if n < 6 or (a.ndim >= 2 and a.shape[-2] != n):
        raise ValueError(f"expected a vector of 6 or more entries or a square matrix, got {a.shape}")
    idx = list(p) + list(range(6, n))
    if a.ndim == 1:
        return a[idx]
    return a[..., idx, :][..., :, idx]


def from_engine(a: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Strains, forces or a stiffness from the engine order to the public order.

    ``a`` is one vector of n >= 6 entries or a stack of square matrices
    (``(..., n, n)``); a seventh (warping) entry stays last. No signs change.
    """
    return _permute(a, FROM_ENGINE)


def to_engine(a: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Inverse of ``from_engine``: public order to the engine order."""
    return _permute(a, TO_ENGINE)


def to_element(K_public: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """A public 6 x 6 stiffness in 1d-fem's element order and signs
    (``section_analysis.SectionAnalysis.stiffness``)."""
    K = to_engine(K_public)
    if K.shape != (6, 6):
        raise ValueError(f"expected a (6, 6) stiffness, got {K.shape}")
    return ELEMENT_SIGN @ K @ ELEMENT_SIGN


def rotation6(angle_deg: float) -> npt.NDArray[np.float64]:
    """6 x 6 rotation of a force-moment (or strain, or displacement-rotation) vector by
    ``angle_deg`` about z: the same 3 x 3 rotation on both halves."""
    t = np.radians(float(angle_deg))
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])


def to_beamdyn_frame(matrix: npt.ArrayLike, twist: float = 0.0) -> npt.NDArray[np.float64]:
    """A public 6 x 6 stiffness or mass matrix to BeamDyn's local frame.

    ``matrix`` is in the public frame with the section turned by ``twist`` degrees
    (public sign: ``rotation6(twist)``). The result is in the untwisted, chord-aligned
    frame, mirrored into BeamDyn's axes (``MIRROR``).
    """
    T = MIRROR @ rotation6(-float(twist))
    return T @ np.asarray(matrix, dtype=np.float64) @ T.T


def twist_from_kernel(twist: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """The geometry kernel's twist (``SectionCut.twist``, degrees) to the public sign."""
    return -np.asarray(twist, dtype=np.float64)


def twist_to_kernel(twist: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Public twist (degrees) to the geometry kernel's sign (``Station(twist=...)``)."""
    return -np.asarray(twist, dtype=np.float64)


__all__ = [
    "DISPLACEMENTS",
    "ELEMENT_SIGN",
    "ENGINE_FORCES",
    "ENGINE_STRAINS",
    "FORCES",
    "FROM_ENGINE",
    "MIRROR",
    "STRAINS",
    "TO_ENGINE",
    "from_engine",
    "rotation6",
    "to_beamdyn_frame",
    "to_element",
    "to_engine",
    "twist_from_kernel",
    "twist_to_kernel",
]
