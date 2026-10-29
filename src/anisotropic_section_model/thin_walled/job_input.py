# pre_processing/section_integration/thin_walled/job_input.py
"""Build section analyses from a job's ``laminate.txt`` and ``section_geometry.txt``."""

from __future__ import annotations

from typing import Dict

from pre_processing.parsing.laminate_parser import parse_laminate_file
from pre_processing.parsing.section_geometry_parser import parse_section_geometry_file, parse_section_mesh
from pre_processing.section_integration.thin_walled.section_analysis import (
    ThinWalledSectionAnalysis,
)


def section_analyses_from_files(
    laminate_path: str, geometry_path: str, hoop: str = "restrained", n_sub: int = 1
) -> Dict[int, ThinWalledSectionAnalysis]:
    """Return ``{section_id: ThinWalledSectionAnalysis}`` for every section in the files.

    ``n_sub`` is the default wall subdivision; a ``[Section_Mesh]`` block overrides it per section.
    """
    laminates = parse_laminate_file(laminate_path)
    sections = parse_section_geometry_file(geometry_path, laminates.keys())
    mesh = parse_section_mesh(geometry_path)
    return {
        sid: ThinWalledSectionAnalysis(sec, laminates, hoop=hoop, n_sub=mesh.get(sid, n_sub))
        for sid, sec in sections.items()
    }


LAMINATE_FILE = "laminate.txt"
SECTION_GEOMETRY_FILE = "section_geometry.txt"
ELEMENT_SECTION_FILE = "element_section.txt"
WALL_LOAD_FILE = "wall_load.txt"
SECTION_ANALYSIS_KEY = "section_analysis_per_element"
THIN_WALLED_KEY = "thin_walled_analysis_per_element"


def attach_section_analyses(job_dir: str, element_dictionary: dict, hoop: str = "restrained") -> dict:
    """Return ``element_dictionary`` with per-element ``SectionAnalysis`` entries when the job has thin-walled sections.

    Looks for ``laminate.txt``, ``section_geometry.txt`` and ``element_section.txt``
    in ``job_dir``. With none present the dictionary is returned unchanged. With
    some but not all present, raises ``FileNotFoundError``. Otherwise adds key
    ``"section_analysis_per_element"``: a tuple aligned with
    ``element_dictionary["ids"]`` holding a ``SectionAnalysis`` for each assigned
    element and ``None`` for the rest. Each section is analysed once. Key
    ``"thin_walled_analysis_per_element"`` holds the ``ThinWalledSectionAnalysis``
    behind each, for stress recovery.

    Rows of ``LinearGBTBeamElement3D`` (or rows with ``[modes]`` / ``[shear_modes]``
    columns) make their sections build GBT matrices with as many modes as any element
    asks for. The dictionary then gains ``"gbt_modes"``, ``"gbt_shear_modes"`` and
    ``"extra_dofs"`` (``n_sv + shear_modes + 2 modes`` for those rows, ``n_sv`` the
    section's Saint-Venant warping modes) for the mesh DOF layout.

    With ``wall_load.txt`` it also gains ``"wall_load_per_element"``: the generalised
    load per length of each loaded element (``None`` for the rest), in the order of the
    section's GBT mass for GBT rows, else ``[N, F_y, F_z, M_x, M_y, M_z, psi load]``.
    """
    import os

    import numpy as np

    from pre_processing.parsing.element_section_parser import parse_element_section_modes
    from pre_processing.section_integration.section_analysis import SectionAnalysis

    paths = [os.path.join(job_dir, f) for f in (LAMINATE_FILE, SECTION_GEOMETRY_FILE, ELEMENT_SECTION_FILE)]
    present = [os.path.exists(p) for p in paths]
    if not any(present):
        return element_dictionary
    if not all(present):
        missing = [os.path.basename(p) for p, ok in zip(paths, present) if not ok]
        raise FileNotFoundError(
            f"Thin-walled sections need laminate.txt, section_geometry.txt and element_section.txt; missing {missing}"
        )
    analyses = section_analyses_from_files(paths[0], paths[1], hoop=hoop)
    assignment_modes = parse_element_section_modes(paths[2])
    assignment = {e: v[0] for e, v in assignment_modes.items()}
    modes_of = {e: v[1] for e, v in assignment_modes.items()}
    shear_of = {e: v[2] for e, v in assignment_modes.items()}
    ids = [int(i) for i in np.asarray(element_dictionary["ids"]).ravel()]
    unknown_el = sorted(set(assignment) - set(ids))
    if unknown_el:
        raise ValueError(f"element_section.txt names unknown elements {unknown_el}")
    unknown_sec = sorted(set(assignment.values()) - set(analyses))
    if unknown_sec:
        raise ValueError(f"element_section.txt names unknown sections {unknown_sec}")
    types = element_dictionary.get("types")
    type_of = {} if types is None else {e: str(t) for e, t in zip(ids, np.asarray(types).ravel())}
    gbt_rows = {e for e in assignment if "GBT" in type_of.get(e, "") or modes_of[e] or shear_of[e]}
    not_gbt = sorted(e for e in gbt_rows if type_of and "GBT" not in type_of.get(e, ""))
    if not_gbt:
        raise ValueError(
            f"element_section.txt gives [modes] / [shear_modes] to elements that are not "
            f"LinearGBTBeamElement3D: {not_gbt}"
        )
    max_modes = {sid: 0 for sid in analyses}
    max_shear = {sid: 0 for sid in analyses}
    gbt_sections = {assignment[e] for e in gbt_rows}
    for e in gbt_rows:
        sid = assignment[e]
        max_modes[sid] = max(max_modes[sid], modes_of[e])
        max_shear[sid] = max(max_shear[sid], shear_of[e])
    per_section = {
        sid: SectionAnalysis.from_thin_walled(
            a, sid, gbt_modes=max_modes[sid], gbt_shear_modes=max_shear[sid], gbt=sid in gbt_sections
        )
        for sid, a in analyses.items()
    }
    per_element = tuple(per_section[assignment[e]] if e in assignment else None for e in ids)
    out = {
        **element_dictionary,
        SECTION_ANALYSIS_KEY: per_element,
        THIN_WALLED_KEY: tuple(analyses[assignment[e]] if e in assignment else None for e in ids),
    }
    if gbt_rows:
        out["gbt_modes"] = np.array([modes_of.get(e, 0) for e in ids], dtype=int)
        out["gbt_shear_modes"] = np.array([shear_of.get(e, 0) for e in ids], dtype=int)
        out["extra_dofs"] = np.array(
            [
                per_section[assignment[e]].gbt_sv_modes + shear_of[e] + 2 * modes_of[e] if e in gbt_rows else 0
                for e in ids
            ],
            dtype=int,
        )

    wl_path = os.path.join(job_dir, WALL_LOAD_FILE)
    if os.path.exists(wl_path):
        from pre_processing.parsing.wall_load_parser import parse_wall_load_file
        from pre_processing.section_integration.thin_walled.cross_section_modes import CrossSectionModes

        loads = parse_wall_load_file(wl_path)
        not_thin = sorted(set(loads) - set(assignment))
        if not_thin:
            raise ValueError(f"wall_load.txt loads elements without a thin-walled section: {not_thin}")
        modes_cache = {}
        per_load = []
        for e in ids:
            if e not in loads:
                per_load.append(None)
                continue
            sid = assignment[e]
            if e in gbt_rows:
                if sid not in modes_cache:
                    modes_cache[sid] = CrossSectionModes(analyses[sid])
                per_load.append(modes_cache[sid].wall_load(loads[e], max_modes[sid], max_shear[sid]))
            else:
                per_load.append(analyses[sid].wall_load_resultant(loads[e]))
        out["wall_load_per_element"] = tuple(per_load)
    return out
