"""Section stages report on the shared logger."""

from __future__ import annotations

import pytest

from anisotropic_section_model.stage_log import collecting, fold_reports, stage


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
