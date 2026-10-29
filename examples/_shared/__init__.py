"""Helpers shared by the example scripts (plot style, analytical references, membrane cell).

Importing this package makes ``anisotropic_section_model`` importable from a
plain clone by adding ``src/`` to the path when the package is not installed.
"""

import sys
from pathlib import Path

try:  # installed with `pip install -e .`
    import anisotropic_section_model  # noqa: F401
except ImportError:  # running straight from a clone
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
