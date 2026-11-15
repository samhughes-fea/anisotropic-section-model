"""Section stages report on the shared logger."""

from __future__ import annotations

import pytest

from anisotropic_section_model.stage_log import (
    LOGGER_NAME,
    StageCollector,
    StageReport,
    collecting,
    fold_reports,
    stage,
)


def test_a_stage_records_seconds_and_an_error_keeps_its_time() -> None:
    with collecting() as sink:
        with stage("anisotropic_section_model", "kept"):
            pass
        with pytest.raises(RuntimeError):
            with stage("anisotropic_section_model", "failed"):
                raise RuntimeError("stage failed")
    folded = fold_reports(sink.take())
    assert folded["kept"] >= 0.0
    failed = folded["failed"]
    assert isinstance(failed, dict)
    assert failed["seconds"] >= 0.0
    assert failed["error"] == "stage failed"


def test_public_contract() -> None:
    import dataclasses
    import logging

    assert LOGGER_NAME == "blade.stage"
    assert [item.name for item in dataclasses.fields(StageReport)] == [
        "package", "stage", "seconds", "error", "depth", "started",
        "cpu_seconds", "measures"]
    assert callable(stage) and callable(fold_reports)
    assert issubclass(StageCollector, logging.Handler)


def test_section_analysis_reports_on_the_shared_logger() -> None:
    from anisotropic_section_model.thin_walled import ThinWalledSection, Wall, WallLaminate
    from anisotropic_section_model.thin_walled.section_analysis import ThinWalledSectionAnalysis

    points = {1: (0.0, 0.0), 2: (0.1, 0.0), 3: (0.1, 0.05), 4: (0.0, 0.05)}
    walls = [Wall(1, 1, 2, 1), Wall(2, 2, 3, 1), Wall(3, 3, 4, 1), Wall(4, 4, 1, 1)]
    section = ThinWalledSection(points, walls)
    laminate = WallLaminate.isotropic(70.0e9, 0.3, 0.001)
    with collecting() as sink:
        ThinWalledSectionAnalysis(section, {1: laminate}).stiffness
    reports = sink.take()
    assert reports[0].measures["stations"] == 1
    folded = fold_reports(reports)
    assert folded["section_analysis"] >= 0.0
