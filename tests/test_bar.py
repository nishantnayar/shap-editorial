import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, no display needed for tests

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _helpers import FakeExplanation

import shap_editorial as se
from shap_editorial._utils import ShapEditorialError


def _make_explanation(n_samples=40, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_samples, n_features))
    names = [f"feat_{i}" for i in range(n_features)]
    return FakeExplanation(values, data=None, feature_names=names)


def test_bar_runs_without_error():
    fig, ax = se.bar(_make_explanation(), title="Importance")
    assert fig is not None
    assert ax is not None


def test_bar_does_not_require_data():
    # Unlike beeswarm, bar needs no `.data` (no colour-by-value).
    exp = FakeExplanation(
        np.random.default_rng(0).normal(size=(20, 4)),
        data=None,
        feature_names=list("abcd"),
    )
    fig, ax = se.bar(exp)
    assert len(ax.get_yticklabels()) == 4


def test_bar_respects_max_display():
    exp = _make_explanation(n_features=8)
    fig, ax = se.bar(exp, max_display=3)
    assert len(ax.get_yticklabels()) == 3  # top-N only by default


def test_bar_show_other_adds_bottom_row():
    exp = _make_explanation(n_features=8)
    fig, ax = se.bar(exp, max_display=3, show_other=True)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == 4
    assert labels[0] == "5 other features"


def test_bar_orders_most_important_on_top():
    n = 30
    values = np.zeros((n, 3))
    values[:, 0] = 5.0  # by far the largest mean |SHAP|
    values[:, 1] = 0.2
    values[:, 2] = 1.0
    exp = FakeExplanation(values, data=None, feature_names=["big", "small", "mid"])
    fig, ax = se.bar(exp, max_display=3)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels[-1] == "big"  # most important at the top


def test_bar_raises_on_multiclass():
    exp = FakeExplanation(np.zeros((5, 3, 2)))
    with pytest.raises(ShapEditorialError, match="multiclass"):
        se.bar(exp)


def test_bar_draws_onto_given_axes():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fig2, ax2 = se.bar(_make_explanation(), ax=ax)
    assert fig2 is fig
    assert ax2 is ax
