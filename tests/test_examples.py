"""The worked examples re-check their committed results (``examples/*/data/results.json``).

Each example ends with checks against its references (closed forms, published values, shell
models) and raises if one is out of tolerance. Every example that keeps its results is run
here with ``--reuse``: it redraws its figures (into a temporary folder, so the committed ones
are left alone) and re-runs its checks, which takes seconds. The full solves run in CI's
``examples`` job (``python examples/run_all.py``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
REUSABLE = sorted(p.parent.parent.name for p in EXAMPLES.glob("[0-9][0-9]_*/data/results.json"))

pytest.importorskip("matplotlib")


@pytest.mark.parametrize("name", REUSABLE)
def test_example_rechecks_its_results(name, tmp_path):
    script = next((EXAMPLES / name).glob("[a-z]*.py"))
    env = dict(os.environ, MPLBACKEND="Agg", ASM_EXAMPLES_FIGURES=str(tmp_path))
    out = subprocess.run([sys.executable, str(script), "--reuse"], capture_output=True, text=True, env=env,
                         timeout=600)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    assert "checks passed" in out.stdout
