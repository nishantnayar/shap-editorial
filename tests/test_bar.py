import matplotlib.pyplot as plt
import numpy as np
import pytest
from _helpers import FakeExplanation, make_explanation

import shap_editorial as se
from shap_editorial._bar import _analysis_line
from shap_editorial._utils import ShapEditorialError


def fig_texts(fig):
    return [t.get_text() for t in fig.texts]


def make_bar_explanation(n_samples=40, n_features=6, seed=0):
    return make_explanation(n_samples, n_features, seed, with_data=False)


def test_bar_runs_without_error():
    fig, ax = se.bar(make_bar_explanation(), title="Importance")
    assert ax.get_figure() is fig
    assert len(ax.get_yticklabels()) == 6


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
    fig, ax = se.bar(make_bar_explanation(n_features=8), max_display=3)
    assert len(ax.get_yticklabels()) == 3  # top-N only by default


def test_bar_show_other_adds_bottom_row():
    exp = make_bar_explanation(n_features=8)
    fig, ax = se.bar(exp, max_display=3, show_other=True)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == 4
    assert labels[0] == "5 other features"


def test_bar_show_other_adds_nothing_when_features_fit_exactly():
    exp = make_bar_explanation(n_features=4)
    fig, ax = se.bar(exp, max_display=4, show_other=True)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == 4
    assert all("other features" not in lbl for lbl in labels)


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


def test_bar_raises_on_zero_max_display():
    with pytest.raises(ShapEditorialError, match="max_display must be at least 1"):
        se.bar(make_bar_explanation(), max_display=0)


def test_bar_draws_onto_given_axes():
    fig, ax = plt.subplots()
    fig2, ax2 = se.bar(make_bar_explanation(), ax=ax)
    assert fig2 is fig
    assert ax2 is ax


def test_bar_show_values_prints_numbers_by_default():
    fig, ax = se.bar(make_bar_explanation(n_features=3))
    assert len(ax.texts) == 3


def test_bar_show_values_false_prints_nothing():
    fig, ax = se.bar(make_bar_explanation(n_features=3), show_values=False)
    assert len(ax.texts) == 0


def test_bar_axis_label_shown_by_default():
    fig, ax = se.bar(make_bar_explanation())
    assert "Average impact on the model's output" in fig_texts(fig)


def test_bar_axis_label_none_omits_it():
    fig, ax = se.bar(make_bar_explanation(), axis_label=None)
    assert not any("Average impact" in t for t in fig_texts(fig))


def test_bar_max_display_one():
    fig, ax = se.bar(make_bar_explanation(n_features=5), max_display=1)
    assert len(ax.get_yticklabels()) == 1


def test_bar_all_equal_importance():
    values = np.tile(np.array([1.0, 1.0, 1.0]), (10, 1))
    exp = FakeExplanation(values, data=None, feature_names=["a", "b", "c"])
    fig, ax = se.bar(exp)
    assert len(ax.get_yticklabels()) == 3
    assert ax.get_xlim()[1] == pytest.approx(1.18)


def test_bar_all_zero_importance_falls_back_to_unit_xlim():
    exp = FakeExplanation(np.zeros((10, 3)), data=None, feature_names=["a", "b", "c"])
    fig, ax = se.bar(exp)
    assert ax.get_xlim()[1] == pytest.approx(1.18)


def test_bar_analysis_false_omits_line():
    fig, ax = se.bar(make_bar_explanation(), analysis=False)
    assert not any("most important feature" in t for t in fig_texts(fig))


def test_bar_custom_analysis_string_is_used():
    fig, ax = se.bar(make_bar_explanation(), analysis="Bespoke takeaway.")
    assert "Bespoke takeaway." in fig_texts(fig)


def test_bar_does_not_leak_rcparams():
    import matplotlib as mpl

    before = mpl.rcParams["axes.labelsize"]
    se.bar(make_bar_explanation(), transparent=True)
    assert mpl.rcParams["axes.labelsize"] == before
    assert mpl.rcParams["savefig.transparent"] is False


def test_analysis_line_none_when_all_importance_zero():
    assert _analysis_line(np.zeros(3), ["a", "b", "c"], np.array([0, 1, 2])) is None


def test_analysis_line_mentions_ratio_when_top_dominates():
    importance = np.array([4.0, 1.0])
    line = _analysis_line(importance, ["big", "small"], np.array([0, 1]))
    assert "big" in line
    assert "4.0×" in line


def test_analysis_line_omits_ratio_when_features_are_close():
    importance = np.array([1.0, 0.98])
    line = _analysis_line(importance, ["a", "b"], np.array([0, 1]))
    assert "×" not in line
