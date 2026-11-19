import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toy_data import make_toy


@pytest.fixture(scope="session")
def toy_df():
    return make_toy(n=2000)