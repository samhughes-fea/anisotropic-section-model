"""Consistent GBT bases on a beam whose sections change along it.

``LinearGBTBeamElement3D`` shares the warping amplitudes ``b_j`` and the mode amplitudes
``a_k`` between neighbouring elements, so on a beam with a different section per element
(a tapered blade, cut at every element's mid-point) the ``j``-th warping field and the
``k``-th distortional mode of one section must be the continuation of those of the next.
Each section builds its own bases: the warping modes are an orthonormal basis of the
Saint-Venant warping, found by an eigen-decomposition, and the distortional modes are
eigenmodes ordered by eigenvalue with a sign convention. Between two close sections a
basis can come out rotated within its span, two modes can swap places, or a mode can
change sign.

``align_gbt_sections`` turns every section's bases to those of a neighbouring section
already aligned, walking the beam from its first element. Within each block (warping
modes, distortional modes) the turn is the orthogonal matrix ``R`` that best maps the
section's signature onto the neighbour's: ``R = polar(S_ref S^T)`` (orthogonal
Procrustes). A signature row holds the moments of one basis field over its section
(``gbt_signature``), which are continuous along a smoothly varying beam. A rotation within a block changes the basis,
not the span, so each section's own model is unchanged; only the meaning of the shared
amplitudes is made the same on both sides of a node.

``R`` is regularised towards the identity, so a field whose signature vanishes (a mode
that does not couple with the beam, as on a doubly symmetric section) keeps its own
basis. A match below ``MIN_MATCH`` (the cosine between the turned and the neighbour's
signature) is reported: the two sections' bases then do not continue each other, and
the element mesh should be refined or the number of modes changed.

Public API
----------
align_gbt_sections, gbt_signature, mass_transform, rotate_gbt_basis
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

__all__ = ["MIN_MATCH", "align_gbt_sections", "gbt_signature", "mass_transform", "rotate_gbt_basis"]

log = logging.getLogger(__name__)

#: lowest cosine between a turned signature row and the neighbour's before it is reported
MIN_MATCH = 0.9
_REG = 1e-9


def _blocks(sa) -> Tuple[int, int]:
    p = sa.gbt_sv_modes + sa.gbt_shear_modes
    return p, sa.gbt_modes


def gbt_signature(sa) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """``(S_w, S_m)``: one row per warping mode and per distortional mode of ``sa``.

    The rows are the field's moments over the section (``gbt_moments``: its displacement
    against the monomials of degree up to 3 in the section coordinates scaled to the
    section's bounding box), so they continue from one section to a similar one. A section
    without moments falls back to the field's couplings with the beam strains and the
    rigid-section motions in the stiffness and the mass.
    """
    p, m = _blocks(sa)
    W = getattr(sa, "gbt_moments", None)
    if W is not None:
        return W[6:6 + p], np.hstack([W[6 + p:6 + p + m], W[6 + p + m:6 + p + 2 * m]])
    K, M = sa.gbt_stiffness, sa.gbt_mass
    k6 = np.linalg.norm(K[:6, :6])
    m6 = max(np.linalg.norm(M[:6, :6]), 1e-300)
    oa = 6 + 2 * p
    Sw = np.hstack([K[6:6 + p, :6], K[6 + p:oa, :6], M[6:6 + p, :6] * (k6 / m6)]) / k6
    Sm = np.hstack([K[oa:oa + m, :6], K[oa + m:oa + 2 * m, :6], K[oa + 2 * m:oa + 3 * m, :6],
                    M[6 + p:6 + p + m, :6] * (k6 / m6), M[6 + p + m:6 + p + 2 * m, :6] * (k6 / m6)]) / k6
    return Sw, Sm


def _procrustes(S_ref: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, float]:
    """Orthogonal ``R`` with ``R S ~ S_ref`` (towards the identity where undetermined), and the
    lowest cosine between the rows of ``R S`` and ``S_ref`` that carry a signature."""
    n = S.shape[0]
    if n == 0:
        return np.eye(0), 1.0
    A = S_ref @ S.T
    scale = max(np.linalg.norm(A), 1e-300)
    U, _, Vt = np.linalg.svd(A + _REG * scale * np.eye(n))
    R = U @ Vt
    RS = R @ S
    nr = np.linalg.norm(S_ref, axis=1) * np.linalg.norm(RS, axis=1)
    big = nr > 1e-6 * max(nr.max(), 1e-300)
    cos = np.einsum("ij,ij->i", RS, S_ref)[big] / nr[big]
    return R, float(cos.min()) if cos.size else 1.0


def mass_transform(sa, Rw: np.ndarray, Rm: np.ndarray) -> np.ndarray:
    """``T`` on the GBT mass order: a generalised load ``f`` of the old basis is ``T^T f`` in the new."""
    p, m = _blocks(sa)
    Tm = np.eye(sa.gbt_mass.shape[0])
    Tm[6:6 + p, 6:6 + p] = Rw.T
    for start in (6 + p, 6 + p + m):
        Tm[start:start + m, start:start + m] = Rm.T
    return Tm


def rotate_gbt_basis(sa, Rw: np.ndarray, Rm: np.ndarray):
    """``sa`` with its warping basis turned by ``Rw`` and its mode basis by ``Rm``.

    New fields ``phi_new = R phi`` (rows), so the old amplitudes are ``R^T`` times the new:
    every GBT matrix becomes ``T^T K T`` with ``T`` the block-diagonal of ``R^T``.
    """
    import dataclasses

    p, m = _blocks(sa)
    nE = sa.gbt_stiffness.shape[0]
    T = np.eye(nE)
    oa = 6 + 2 * p
    for start in (6, 6 + p):
        T[start:start + p, start:start + p] = Rw.T
    for start in (oa, oa + m, oa + 2 * m):
        T[start:start + m, start:start + m] = Rm.T
    Tm = mass_transform(sa, Rw, Rm)
    K = T.T @ sa.gbt_stiffness @ T
    M = Tm.T @ sa.gbt_mass @ Tm
    G = sa.gbt_geometric
    if G is not None:
        Tq = np.eye(3 + m)
        Tq[3:, 3:] = Rm.T
        G = tuple(np.einsum("ai,bj,ije,ef->abf", Tq.T, Tq.T, X, T) for X in G)
    W = sa.gbt_moments
    W = None if W is None else Tm.T @ W
    Bz = getattr(sa, "gbt_brazier", None)
    if Bz is not None:
        # new mode amplitudes a_new = Rm a_old (old = Rm^T new)
        Tb, Qb = Bz
        Tb = np.einsum("ik,kce,ef->icf", Rm, Tb, T)
        Qb = np.einsum("ik,jl,klcd->ijcd", Rm, Rm, Qb)
        Bz = (Tb, Qb)
    return dataclasses.replace(sa, gbt_stiffness=0.5 * (K + K.T), gbt_mass=0.5 * (M + M.T), gbt_geometric=G,
                               gbt_moments=W, gbt_brazier=Bz)


def align_gbt_sections(per_element: Sequence, connectivity: npt.ArrayLike,
                       gbt_elements: Optional[Sequence[bool]] = None) -> Tuple[tuple, Dict[int, dict]]:
    """Turn the GBT bases of each element's section to its neighbour's (module docstring).

    ``per_element`` holds each element's ``SectionAnalysis`` (or ``None``), ``connectivity``
    its two nodes. Elements are visited in order; each is aligned to an earlier visited
    element sharing a node whose section differs, or kept as it is. Returns the aligned
    sections and, per element whose bases were turned, ``{"Rw", "Rm", "match"}``.
    """
    conn = np.asarray(connectivity, dtype=np.int64).reshape(-1, 2)
    n = len(per_element)
    use = [sa is not None and getattr(sa, "gbt_stiffness", None) is not None for sa in per_element]
    if gbt_elements is not None:
        use = [u and bool(g) for u, g in zip(use, gbt_elements)]
    out: List = list(per_element)
    info: Dict[int, dict] = {}
    at_node: Dict[int, List[int]] = {}
    sig_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def sig(i):
        if i not in sig_cache:
            sig_cache[i] = gbt_signature(out[i])
        return sig_cache[i]

    for e in range(n):
        if use[e]:
            nb = [f for nd in conn[e] for f in at_node.get(int(nd), []) if use[f]]
            same = next((f for f in nb if per_element[f] is per_element[e]), None)
            ref = next((f for f in nb if per_element[f] is not per_element[e]), None)
            if same is not None:                     # the same section: the same (turned) basis
                out[e] = out[same]
                if same in info:
                    info[e] = dict(info[same])
            elif ref is not None and _blocks(out[ref]) == _blocks(out[e]):
                Sw_r, Sm_r = sig(ref)
                Sw, Sm = sig(e)
                Rw, cw = _procrustes(Sw_r, Sw)
                Rm, cm = _procrustes(Sm_r, Sm)
                match = min(cw, cm)
                if match < MIN_MATCH:
                    log.warning("GBT bases of element %d do not continue those of element %d (match %.2f): "
                                "refine the elements or change the number of modes", e, ref, match)
                if not (np.allclose(Rw, np.eye(Rw.shape[0]), atol=1e-12)
                        and np.allclose(Rm, np.eye(Rm.shape[0]), atol=1e-12)):
                    out[e] = rotate_gbt_basis(out[e], Rw, Rm)
                    sig_cache.pop(e, None)
                info[e] = {"Rw": Rw, "Rm": Rm, "match": match, "reference": ref}
        for nd in conn[e]:
            at_node.setdefault(int(nd), []).append(e)
    return tuple(out), info
