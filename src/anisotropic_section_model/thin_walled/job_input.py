# pre_processing/section_integration/thin_walled/job_input.py
"""Build section analyses from a job's ``laminate.txt`` and ``section_geometry.txt``."""

from __future__ import annotations

from typing import Dict

from pre_processing.parsing.laminate_parser import parse_laminate_file
from pre_processing.parsing.section_geometry_parser import parse_section_geometry_file
from pre_processing.section_integration.thin_walled.section_analysis import (
    ThinWalledSectionAnalysis,
)


def section_analyses_from_files(
    laminate_path: str, geometry_path: str, hoop: str = "restrained", n_sub: int = 1
) -> Dict[int, ThinWalledSectionAnalysis]:
    """Return ``{section_id: ThinWalledSectionAnalysis}`` for every section in the files."""
    laminates = parse_laminate_file(laminate_path)
    sections = parse_section_geometry_file(geometry_path, laminates.keys())
    return {
        sid: ThinWalledSectionAnalysis(sec, laminates, hoop=hoop, n_sub=n_sub)
        for sid, sec in sections.items()
    }


LAMINATE_FILE = "laminate.txt"
SECTION_GEOMETRY_FILE = "section_geometry.txt"
ELEMENT_SECTION_FILE = "element_section.txt"
SECTION_ANALYSIS_KEY = "section_analysis_per_element"


def attach_section_analyses(job_dir: str, element_dictionary: dict, hoop: str = "restrained") -> dict:
    """Return ``element_dictionary`` with per-element ``SectionAnalysis`` entries when the job has thin-walled sections.

    Looks for ``laminate.txt``, ``section_geometry.txt`` and ``element_section.txt``
    in ``job_dir``. With none present the dictionary is returned unchanged. With
    some but not all present, raises ``FileNotFoundError``. Otherwise adds key
    ``"section_analysis_per_element"``: a tuple aligned with
    ``element_dictionary["ids"]`` holding a ``SectionAnalysis`` for each assigned
    element and ``None`` for the rest. Each section is analysed once.

    With a ``[modes]`` column, sections also get their GBT matrices (as many modes as
    any element asks for), and the dictionary gains ``"gbt_modes"`` and
    ``"extra_dofs"`` (``1 + 2 modes`` for those rows) for the mesh DOF layout.
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
    assignment = {e: sid for e, (sid, _) in assignment_modes.items()}
    modes_of = {e: m for e, (_, m) in assignment_modes.items()}
    ids = [int(i) for i in np.asarray(element_dictionary["ids"]).ravel()]
    unknown_el = sorted(set(assignment) - set(ids))
    if unknown_el:
        raise ValueError(f"element_section.txt names unknown elements {unknown_el}")
    unknown_sec = sorted(set(assignment.values()) - set(analyses))
    if unknown_sec:
        raise ValueError(f"element_section.txt names unknown sections {unknown_sec}")
    max_modes = {sid: 0 for sid in analyses}
    for e, sid in assignment.items():
        max_modes[sid] = max(max_modes[sid], modes_of[e])
    per_section = {
        sid: SectionAnalysis.from_thin_walled(a, sid, gbt_modes=max_modes[sid]) for sid, a in analyses.items()
    }
    per_element = tuple(per_section[assignment[e]] if e in assignment else None for e in ids)
    out = {**element_dictionary, SECTION_ANALYSIS_KEY: per_element}
    if any(modes_of.values()):
        gbt = np.array([modes_of.get(e, 0) for e in ids], dtype=int)
        out["gbt_modes"] = gbt
        out["extra_dofs"] = np.where(gbt > 0, 1 + 2 * gbt, 0)
    return out
