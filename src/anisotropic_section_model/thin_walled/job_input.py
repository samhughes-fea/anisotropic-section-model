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
