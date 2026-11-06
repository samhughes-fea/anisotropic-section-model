"""Find files in the private reference-blades catalogue.

Reference blade definitions (layups, aerofoils, materials) are not stored in this
public repository. Examples that need one read it from a local checkout of
github.com/samhughes-fea/reference-blades, following that repository's RULES.md:

    git clone https://github.com/samhughes-fea/reference-blades.git reference_blades

at the root of this repository (``reference_blades/`` is git-ignored), or anywhere
with ``REFERENCE_BLADES_DIR`` pointing at it. Without it those examples are skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class ReferenceBladesUnavailable(FileNotFoundError):
    """The private reference-blades checkout is not on this machine."""


def reference_blade_file(relpath: str, repo_root: Path = REPO_ROOT) -> Path:
    """``relpath`` (e.g. ``iea15/derived/sections/precomp/precomp_sections.json``) in the checkout."""
    env = os.environ.get("REFERENCE_BLADES_DIR")
    root = Path(env) if env else repo_root / "reference_blades"
    path = root / relpath
    if not path.is_file():
        raise ReferenceBladesUnavailable(
            f"{relpath} needs the private reference-blades catalogue: clone it to "
            f"{repo_root / 'reference_blades'} or set REFERENCE_BLADES_DIR")
    return path
