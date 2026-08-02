import matplotlib.pyplot as plt
import numpy as np
import pytest
from _helpers import FakeExplanation, make_single_explanation
from matplotlib.colors import to_rgba

import shap_editorial as se
from shap_editorial._theme import C_NEG, C_POS
from shap_editorial._utils import ShapEditorialError
from shap_editorial._waterfall import _fmt


def fig_texts(fig):
    return [t.get_text() for t in fig.texts]


def axes_texts(ax):
    return [t.get_text() for t in ax.texts]


def bar_colors(ax):
    return [tuple(p.get_facecolor()) for p in ax.patches]


def test_waterfall_runs_without_error():
    fig, ax = se.waterfall(make_single_explanation(), title="One prediction")
    assert ax.get_figure() is fig
    assert len(ax.get_yticklabels()) == 6


def test_waterfall_max_display_aggregates_the_rest():
    exp = make_single_explanation(n_features=8)
    fig, ax = se.waterfall(exp, max_display=3)
    # 3 features + 1 "other features" row = 4 rows
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == 4
    assert labels[0] == "5 other features"  # aggregate sits at the bottom


def test_waterfall_no_other_row_when_features_fit():
    fig, ax = se.waterfall(make_single_explanation(n_features=3), max_display=5)
    assert len(ax.get_yticklabels()) == 3


def test_waterfall_show_other_false_omits_aggregate():
    exp = make_single_explanation(n_features=8)
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


def test_waterfall_raises_on_zero_max_display():
    with pytest.raises(ShapEditorialError, match="max_display must be at least 1"):
        se.waterfall(make_single_explanation(), max_display=0)


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


def test_waterfall_show_values_without_data_stays_plain():
    exp = FakeExplanation(
        np.array([0.3, -0.2]), data=None, feature_names=["x", "y"], base_values=0.0
    )
    fig, ax = se.waterfall(exp, show_values=True)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert all("=" not in lbl for lbl in labels)


def test_waterfall_all_positive_contributions_are_red():
    exp = FakeExplanation(
        np.array([0.4, 0.3, 0.2]), feature_names=["a", "b", "c"], base_values=0.1
    )
    fig, ax = se.waterfall(exp, highlight=False)
    assert bar_colors(ax) == [to_rgba(C_POS)] * 3


def test_waterfall_all_negative_contributions_are_grey():
    exp = FakeExplanation(
        np.array([-0.4, -0.3, -0.2]), feature_names=["a", "b", "c"], base_values=0.9
    )
    fig, ax = se.waterfall(exp, highlight=False)
    assert bar_colors(ax) == [to_rgba(C_NEG)] * 3


def test_waterfall_all_zero_contributions_drops_duplicate_endpoint_label():
    # base == fx, so the two endpoint labels would print on top of each other.
    exp = FakeExplanation(np.zeros(3), feature_names=["a", "b", "c"], base_values=0.5)
    fig, ax = se.waterfall(exp)
    texts = axes_texts(ax)
    assert any("This prediction" in t for t in texts)
    assert not any("Average prediction" in t for t in texts)


def test_waterfall_distinct_endpoints_keep_both_labels():
    exp = FakeExplanation(
        np.array([0.5, 0.3]), feature_names=["a", "b"], base_values=0.1
    )
    fig, ax = se.waterfall(exp)
    texts = axes_texts(ax)
    assert any("This prediction" in t for t in texts)
    assert any("Average prediction" in t for t in texts)


def test_waterfall_max_display_one_draws_no_connectors():
    exp = make_single_explanation(n_features=5)
    fig, ax = se.waterfall(exp, max_display=1, show_other=False)
    assert len(ax.get_yticklabels()) == 1
    assert len(ax.lines) == 2  # the two endpoint reference lines, no connectors


def test_waterfall_analysis_false_omits_line():
    fig, ax = se.waterfall(make_single_explanation(), analysis=False)
    assert not any("largest effect" in t for t in fig_texts(fig))


def test_waterfall_custom_analysis_string_is_used():
    fig, ax = se.waterfall(make_single_explanation(), analysis="Bespoke takeaway.")
    assert "Bespoke takeaway." in fig_texts(fig)


def test_waterfall_accepts_base_values_as_array():
    exp = FakeExplanation(
        np.array([0.2, -0.1]), feature_names=["a", "b"], base_values=np.array([0.4])
    )
    fig, ax = se.waterfall(exp)
    assert any("0.5" in t for t in axes_texts(ax))  # 0.4 + 0.2 - 0.1


def test_waterfall_draws_onto_given_axes():
    fig, ax = plt.subplots()
    fig2, ax2 = se.waterfall(make_single_explanation(), ax=ax)
    assert fig2 is fig
    assert ax2 is ax


def test_waterfall_does_not_leak_rcparams():
    import matplotlib as mpl

    before = mpl.rcParams["axes.labelsize"]
    se.waterfall(make_single_explanation(), transparent=True)
    assert mpl.rcParams["axes.labelsize"] == before
    assert mpl.rcParams["savefig.transparent"] is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3.0, "3"),
        (-7.0, "-7"),
        (3.5, "3.5"),
        (0.000123456, "0.000123"),
        (1e7, "1e+07"),  # too large for the integer shortcut
    ],
)
def test_fmt_numeric(value, expected):
    assert _fmt(value) == expected


def test_fmt_passes_through_non_numeric():
    assert _fmt("categorical") == "categorical"


def test_fmt_passes_through_nan():
    assert _fmt(float("nan")) == "nan"
