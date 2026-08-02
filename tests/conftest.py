import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # headless: no display in CI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(autouse=True)
def close_figures():
    """Charts are created but never shown, so close them between tests."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
