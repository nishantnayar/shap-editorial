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


def _make_single(n_features=6, seed=0, base=0.5):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=n_features)
    data = rng.uniform(size=n_features)
    names = [f"feat_{i}" for i in range(n_features)]
    return FakeExplanation(values, data, names, base_values=base)


def test_waterfall_runs_without_error():
    fig, ax = se.waterfall(_make_single(), title="One prediction")
    assert fig is not None
    assert ax is not None


def test_waterfall_max_display_aggregates_the_rest():
    exp = _make_single(n_features=8)
    fig, ax = se.waterfall(exp, max_display=3)
    # 3 features + 1 "other features" row = 4 rows
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == 4
    assert labels[0] == "5 other features"  # aggregate sits at the bottom


def test_waterfall_no_other_row_when_features_fit():
    exp = _make_single(n_features=3)
    fig, ax = se.waterfall(exp, max_display=5)
    assert len(ax.get_yticklabels()) == 3


def test_waterfall_show_other_false_omits_aggregate():
    exp = _make_single(n_features=8)
    fig, ax = se.waterfall(exp, max_display=3, show_other=False)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == 3  # only the top 3, no aggregate row
    assert all("other features" not in lbl for lbl in labels)


def test_waterfall_top_row_is_largest_contribution():
    values = np.array([0.05, 0.9, -0.1, 0.02])
    data = np.array([1.0, 2.0, 3.0, 4.0])
    exp = FakeExplanation(values, data, ["a", "big", "c", "d"], base_values=0.0)
    fig, ax = se.waterfall(exp)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    # Largest |contribution| is drawn at the top (last y-tick).
    assert labels[-1].startswith("big")


def test_waterfall_squeezes_single_row_2d():
    values = np.zeros((1, 4))
    data = np.zeros((1, 4))
    exp = FakeExplanation(values, data, ["a", "b", "c", "d"], base_values=0.1)
    fig, ax = se.waterfall(exp)
    assert len(ax.get_yticklabels()) == 4


def test_waterfall_raises_on_multiclass():
    exp = FakeExplanation(np.zeros((5, 3, 2)))
    with pytest.raises(ShapEditorialError, match="multiclass"):
        se.waterfall(exp)


def test_waterfall_raises_on_multiple_samples():
    exp = FakeExplanation(np.zeros((5, 3)), base_values=0.0)
    with pytest.raises(ShapEditorialError, match="single prediction"):
        se.waterfall(exp)


def test_waterfall_raises_without_base_values():
    exp = FakeExplanation(np.zeros(4), data=np.zeros(4), feature_names=list("abcd"))
    with pytest.raises(ShapEditorialError, match="base_values"):
        se.waterfall(exp)


def test_waterfall_labels_plain_by_default():
    values = np.array([0.3, -0.2, 0.1])
    data = np.array([12.0, 3.5, 7.0])
    exp = FakeExplanation(values, data, ["x", "y", "z"], base_values=0.0)
    fig, ax = se.waterfall(exp)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    # By default labels are just the feature name — no raw "= value".
    assert all("=" not in lbl for lbl in labels)


def test_waterfall_show_values_appends_feature_value():
    values = np.array([0.3, -0.2, 0.1])
    data = np.array([12.0, 3.5, 7.0])
    exp = FakeExplanation(values, data, ["x", "y", "z"], base_values=0.0)
    fig, ax = se.waterfall(exp, show_values=True)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert any("=" in lbl for lbl in labels)
