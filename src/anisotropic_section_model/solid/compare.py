# anisotropic_section_model/solid/compare.py
"""Solid 2D section versus the thin-walled 6 x 6, and versus published matrices.

The thin-wall assumption is weakest at the root and in the cap region. This
module runs the solid 2D finite-element section (``SolidSectionAnalysis`` of a
``section_mesh_from_cut`` mesh) beside the thin-walled 6 x 6 of the same cut
and reports the difference. A published matrix is compared only when the
caller supplies one; missing or null fields stay ``absent``. VABS and BECAS
are not held and are not invented.

Named stiffness (``EA``, ``EI_flap``, ``EI_edge``, ``GJ``) follows the same
reduction as the existing solid-versus-thin-walled tests: classical 4 x 4
about the tension centre, ``GJ`` with shear free. Shear stiffness is the
Timoshenko 6 x 6 about the shear centre. A published ``K6`` is compared only
when it is already in this package's ``STRAINS`` order; a BeamDyn-order matrix
is not converted here.

The geometry kernel is not imported. Cuts are read by duck typing, as
``thin_walled.from_geometry`` and ``solid.from_geometry`` already do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from anisotropic_section_model.thin_walled.section_analysis import stiffness_about

__all__ = [
    "ABSENT",
    "NAMED_STIFFNESS",
    "ROLES",
    "StationComparison",
    "TermComparison",
    "compare_analyses",
    "compare_cut",
    "compare_thick_stations",
    "named_stiffness",
    "published_from_record",
]

ABSENT = "absent"
ROLES: Tuple[str, ...] = ("root", "cap")
#: stiffness names compared to a published station record when that record holds them
NAMED_STIFFNESS: Tuple[str, ...] = ("EA", "EI_flap", "EI_edge", "GJ", "GA_flap", "GA_edge")
_CENTRE_KEYS: Tuple[str, ...] = (
    "shear_centre_y",
    "shear_centre_z",
    "tension_centre_y",
    "tension_centre_z",
    "mass_centre_y",
    "mass_centre_z",
)
_STRAINS_ORDERS = frozenset({"onedfem", "strains"})


def _relative(value: float, reference: Optional[float]) -> Union[float, str]:
    """``(value - reference) / |reference|``, or ``absent`` when the reference is missing or ~0."""
    if reference is None:
        return ABSENT
    scale = abs(float(reference))
    if scale <= 1e-30:
        return ABSENT
    return (float(value) - float(reference)) / scale


def _held(record: Mapping[str, Any], key: str) -> Union[float, str]:
    if key not in record:
        return ABSENT
    value = record[key]
    if value is None or value == ABSENT:
        return ABSENT
    return float(value)


def named_stiffness(analysis) -> Dict[str, float]:
    """Frame-free named terms of a solid or thin-walled Saint-Venant analysis.

    ``EA``, ``EI_flap`` and ``EI_edge`` are the classical 4 x 4 about the
    tension centre (``EI_flap`` from ``kappa_y``, ``EI_edge`` from ``kappa_z``).
    ``GJ`` is the inverse of the free-shear twist flexibility. ``GA_flap`` and
    ``GA_edge`` are the Timoshenko shear stiffness about the shear centre.
    Centres and mass are included so a report can show them beside the 6 x 6;
    they are not a published matrix.
    """
    Kc = np.asarray(analysis.classical_stiffness, dtype=np.float64)
    yt, zt = (float(v) for v in analysis.tension_centre)
    ys, zs = (float(v) for v in analysis.shear_centre)
    ea = float(Kc[0, 0])
    Ks = stiffness_about(analysis.stiffness, ys, zs)
    out: Dict[str, float] = {
        "EA": ea,
        "EI_flap": float(Kc[1, 1] - ea * zt * zt),
        "EI_edge": float(Kc[2, 2] - ea * yt * yt),
        "GJ": float(1.0 / analysis.flexibility[5, 5]),
        "GA_edge": float(Ks[3, 3]),
        "GA_flap": float(Ks[4, 4]),
        "shear_centre_y": ys,
        "shear_centre_z": zs,
        "tension_centre_y": yt,
        "tension_centre_z": zt,
    }
    mass = getattr(analysis, "mass_properties", None)
    if mass:
        out["mass"] = float(mass["m"])
        out["mass_centre_y"] = float(mass["y_m"])
        out["mass_centre_z"] = float(mass["z_m"])
    return out


def published_from_record(record: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Named terms (and optional ``STRAINS``-order ``K6``) that a published record actually holds.

    Missing or null fields are omitted, so the comparison reports them as
    ``absent``. A ``K6`` is accepted only when ``K6_order`` / ``k6_order`` is
    ``onedfem`` or ``strains``: this package does not convert a BeamDyn-order
    matrix. VABS and BECAS keys are ignored; those gates are not held.
    """
    if not record:
        return {}
    out: Dict[str, Any] = {}
    for key in NAMED_STIFFNESS + ("mass",):
        value = _held(record, key)
        if value is not ABSENT:
            out[key] = value
    k6 = record.get("K6")
    order = record.get("K6_order", record.get("k6_order"))
    if k6 is not None and order in _STRAINS_ORDERS:
        K = np.asarray(k6, dtype=np.float64)
        if K.shape != (6, 6):
            raise ValueError(f"published K6 must be (6, 6), got {K.shape}")
        out["K6"] = K
    return out


def _has_caps(cut) -> Optional[bool]:
    walls_of = getattr(cut, "walls_of", None)
    if walls_of is None:
        return None
    return bool(walls_of("cap"))


@dataclass(frozen=True)
class TermComparison:
    """One named quantity: solid, thin-walled, and published when held."""

    name: str
    solid: float
    thin_walled: float
    difference_from_thin_walled: float
    relative_to_thin_walled: Union[float, str]
    published: Union[float, str]
    difference_from_published: Union[float, str]
    relative_to_published: Union[float, str]

    def as_dict(self) -> Dict[str, Union[str, float]]:
        return {
            "name": self.name,
            "solid": self.solid,
            "thin_walled": self.thin_walled,
            "difference_from_thin_walled": self.difference_from_thin_walled,
            "relative_to_thin_walled": self.relative_to_thin_walled,
            "published": self.published,
            "difference_from_published": self.difference_from_published,
            "relative_to_published": self.relative_to_published,
        }


@dataclass(frozen=True)
class StationComparison:
    """Solid 2D 6 x 6 versus thin-walled 6 x 6 (and published, if held) at one station."""

    role: str
    z: Optional[float]
    has_caps: Optional[bool]
    terms: Dict[str, TermComparison]
    solid_stiffness: npt.NDArray[np.float64]
    thin_walled_stiffness: npt.NDArray[np.float64]
    relative_frobenius_to_thin_walled: float
    published_stiffness: Union[npt.NDArray[np.float64], str]
    relative_frobenius_to_published: Union[float, str]
    vabs: str = ABSENT
    becas: str = ABSENT

    def as_dict(self) -> Dict[str, Any]:
        pub_k6 = self.published_stiffness
        return {
            "role": self.role,
            "z": self.z,
            "has_caps": self.has_caps,
            "terms": {name: term.as_dict() for name, term in self.terms.items()},
            "solid_stiffness": np.asarray(self.solid_stiffness).tolist(),
            "thin_walled_stiffness": np.asarray(self.thin_walled_stiffness).tolist(),
            "relative_frobenius_to_thin_walled": self.relative_frobenius_to_thin_walled,
            "published_stiffness": ABSENT if isinstance(pub_k6, str) else np.asarray(pub_k6).tolist(),
            "relative_frobenius_to_published": self.relative_frobenius_to_published,
            "vabs": self.vabs,
            "becas": self.becas,
        }


def _term(name: str, solid: float, thin: float, published: Union[float, str]) -> TermComparison:
    if published == ABSENT:
        pub: Union[float, str] = ABSENT
        d_pub: Union[float, str] = ABSENT
        r_pub: Union[float, str] = ABSENT
    else:
        pub = float(published)
        d_pub = solid - pub
        r_pub = _relative(solid, pub)
    return TermComparison(
        name,
        solid,
        thin,
        solid - thin,
        _relative(solid, thin),
        pub,
        d_pub,
        r_pub,
    )


def _frobenius_relative(left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]) -> float:
    denom = float(np.linalg.norm(right))
    if denom <= 1e-30:
        raise ValueError("cannot form a relative Frobenius difference against a zero matrix")
    return float(np.linalg.norm(left - right) / denom)


def compare_analyses(
    solid,
    thin_walled,
    published: Optional[Mapping[str, Any]] = None,
    *,
    role: str = "root",
    z: Optional[float] = None,
    has_caps: Optional[bool] = None,
) -> StationComparison:
    """Compare a ``SolidSectionAnalysis`` to a thin-walled analysis of the same section.

    ``published`` is a mapping from ``published_from_record`` (or the same
    keys). Omitted keys, including a missing ``K6``, stay ``absent``. ``vabs``
    and ``becas`` are always ``absent``.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    held = published_from_record(published)
    solid_terms = named_stiffness(solid)
    thin_terms = named_stiffness(thin_walled)
    names = list(dict.fromkeys([*NAMED_STIFFNESS, *_CENTRE_KEYS, "mass"]))
    terms = {
        name: _term(name, solid_terms[name], thin_terms[name], held.get(name, ABSENT))
        for name in names
        if name in solid_terms and name in thin_terms
    }
    Ks = np.asarray(solid.stiffness, dtype=np.float64)
    Kt = np.asarray(thin_walled.stiffness, dtype=np.float64)
    if Ks.shape != (6, 6) or Kt.shape != (6, 6):
        raise ValueError("both analyses must give a (6, 6) stiffness")
    pub_k6 = held.get("K6", ABSENT)
    if isinstance(pub_k6, str):
        rel_pub: Union[float, str] = ABSENT
    else:
        rel_pub = _frobenius_relative(Ks, np.asarray(pub_k6, dtype=np.float64))
    return StationComparison(
        role,
        None if z is None else float(z),
        has_caps,
        terms,
        Ks.copy(),
        Kt.copy(),
        _frobenius_relative(Ks, Kt),
        ABSENT if isinstance(pub_k6, str) else np.asarray(pub_k6, dtype=np.float64).copy(),
        rel_pub,
        ABSENT,
        ABSENT,
    )


def compare_cut(
    cut,
    laminates: Mapping[str, object],
    published: Optional[Mapping[str, Any]] = None,
    *,
    role: str = "root",
    element_size: Optional[float] = None,
    rows_per_ply: int = 1,
    thickness_tol: float = 0.01,
    origin: str = "pitch_axis",
) -> StationComparison:
    """Solid 2D mesh and thin-walled 6 x 6 of the same implicit-geometry cut.

    The cut is duck-typed (``section_mesh_from_cut`` / ``thin_walled_from_cut``).
    """
    from anisotropic_section_model.solid.from_geometry import section_mesh_from_cut
    from anisotropic_section_model.solid.section_analysis import SolidSectionAnalysis
    from anisotropic_section_model.thin_walled.from_geometry import thin_walled_from_cut

    mesh = section_mesh_from_cut(
        cut, laminates, element_size=element_size, rows_per_ply=rows_per_ply,
        thickness_tol=thickness_tol, origin=origin,
    )
    solid = SolidSectionAnalysis(mesh)
    thin = thin_walled_from_cut(
        cut, laminates, thickness_tol=thickness_tol, origin=origin,
    ).analysis()
    raw_z = getattr(cut, "z", None)
    z = None if raw_z is None else float(raw_z)
    return compare_analyses(
        solid, thin, published, role=role, z=z, has_caps=_has_caps(cut),
    )


def _station_parts(item) -> Tuple[str, Any, Optional[Mapping[str, Any]]]:
    if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) not in (2, 3):
        raise ValueError("each station is (role, cut) or (role, cut, published)")
    role, cut = item[0], item[1]
    published = item[2] if len(item) == 3 else None
    return str(role), cut, published


def compare_thick_stations(
    stations: Iterable,
    laminates: Mapping[str, object],
    published_by_role: Optional[Mapping[str, Mapping[str, Any]]] = None,
    **kwargs,
) -> Tuple[StationComparison, ...]:
    """Solid 2D versus thin-walled 6 x 6 at labelled root and cap stations.

    Parameters
    ----------
    stations : iterable of (role, cut) or (role, cut, published)
        ``role`` is ``root`` or ``cap``. A per-station ``published`` mapping
        overrides ``published_by_role[role]``.
    laminates, **kwargs
        Passed to ``compare_cut``.
    published_by_role : mapping, optional
        Published record for a role when the station tuple has no third item.
    """
    held_by_role = published_by_role or {}
    out = []
    for item in stations:
        role, cut, published = _station_parts(item)
        record = published if published is not None else held_by_role.get(role)
        out.append(compare_cut(cut, laminates, record, role=role, **kwargs))
    return tuple(out)
