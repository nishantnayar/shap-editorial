import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, no display needed for tests

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import shap_editorial as se
from shap_editorial._utils import ShapEditorialError
from _helpers import FakeExplanation


def _make_explanation(n_samples=40, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_samples, n_features))
    data = rng.uniform(size=(n_samples, n_features))
    names = [f"feat_{i}" for i in range(n_features)]
    return FakeExplanation(values, data, names)


def test_beeswarm_runs_without_error():
    exp = _make_explanation()
    fig, ax = se.beeswarm(exp, title="Test chart")
    assert fig is not None
    assert ax is not None


def test_beeswarm_respects_max_display():
    exp = _make_explanation(n_features=8)
    fig, ax = se.beeswarm(exp, max_display=3)
    # By default only the top max_display features are shown (no "other" row).
    assert len(ax.get_yticklabels()) == 3


def test_beeswarm_show_other_adds_bottom_row():
    exp = _make_explanation(n_features=8)
    fig, ax = se.beeswarm(exp, max_display=3, show_other=True)
    # 3 kept features + 1 collapsed "other" row = 4 y-ticks
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == 4
    # The aggregate row sits at the bottom (first y-tick).
    assert labels[0] == "5 other features"


def test_beeswarm_no_other_row_when_features_fit():
    exp = _make_explanation(n_features=3)
    fig, ax = se.beeswarm(exp, max_display=5)
    assert len(ax.get_yticklabels()) == 3


def test_beeswarm_raises_without_data():
    exp = FakeExplanation(np.zeros((5, 3)), data=None, feature_names=["a", "b", "c"])
    with pytest.raises(ValueError, match="no `.data`"):
        se.beeswarm(exp)


def test_beeswarm_raises_on_multiclass_input():
    exp = FakeExplanation(np.zeros((5, 3, 2)))
    with pytest.raises(ShapEditorialError, match="multiclass"):
        se.beeswarm(exp)


def test_beeswarm_draws_onto_given_axes():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    exp = _make_explanation()
    fig2, ax2 = se.beeswarm(exp, ax=ax)
    assert fig2 is fig
    assert ax2 is ax


def test_beeswarm_feature_order_largest_impact_first():
    # Construct values where feature 0 clearly has the largest mean |shap|.
    n = 30
    values = np.zeros((n, 3))
    values[:, 0] = 10.0  # huge impact
    values[:, 1] = 0.1
    values[:, 2] = 1.0
    data = np.random.default_rng(1).uniform(size=(n, 3))
    exp = FakeExplanation(values, data, ["big", "small", "medium"])

    fig, ax = se.beeswarm(exp, max_display=3)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    # Largest-impact feature should be at the top of the plot (last y-tick).
    assert labels[-1] == "big"
