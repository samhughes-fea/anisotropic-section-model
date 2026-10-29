import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from composite_laminate_model import create_default_library  # noqa: E402


@pytest.fixture(scope="session")
def library():
    return create_default_library()


@pytest.fixture(scope="session")
def aluminium(library):
    return library.get("Al-6061-T6")
