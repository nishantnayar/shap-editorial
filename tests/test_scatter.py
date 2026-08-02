import matplotlib as mpl
import numpy as np
import pytest
from _helpers import FakeExplanation, make_explanation

import shap_editorial as se
from shap_editorial._scatter import _analysis_line, _jitter_width
from shap_editorial._utils import ShapEditorialError


def fig_texts(fig):
    return [t.get_text() for t in fig.texts]


def test_scatter_runs_and_plots_every_sample():
    expl = make_explanation(n_samples=40)
    fig, ax = se.scatter(expl, feature="feat_1")
    assert ax.collections[0].get_offsets().shape[0] == 40


def test_scatter_uses_the_named_feature_columns():
    values = np.array([[1.0, -5.0], [2.0, -6.0], [3.0, -7.0]])
    data = np.array([[10.0, 0.1], [20.0, 0.2], [30.0, 0.3]])
    expl = FakeExplanation(values, data, ["a", "b"])
    fig, ax = se.scatter(expl, feature="b", analysis=False)
    pts = ax.collections[0].get_offsets()
    np.testing.assert_allclose(pts[:, 0], data[:, 1])
    np.testing.assert_allclose(pts[:, 1], values[:, 1])


def test_scatter_accepts_integer_index():
    fig, ax = se.scatter(make_explanation(), feature=2, analysis=False)
    assert ax.get_xlabel() == "feat_2"


def test_scatter_defaults_to_top_driver():
    # feat_1 carries by far the largest mean |SHAP|, so it should be chosen.
    values = np.array([[0.1, 9.0], [-0.2, -8.0], [0.05, 7.5]])
    data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    expl = FakeExplanation(values, data, ["small", "big"])
    fig, ax = se.scatter(expl, analysis=False)
    assert ax.get_xlabel() == "big"


def test_scatter_unknown_feature_name():
    with pytest.raises(ShapEditorialError, match="No feature named"):
        se.scatter(make_explanation(), feature="nope")


def test_scatter_raises_without_data():
    expl = make_explanation(with_data=False)
    with pytest.raises(ShapEditorialError, match="no `.data`"):
        se.scatter(expl, feature=0)


def test_scatter_rejects_multiclass():
    expl = FakeExplanation(np.zeros((4, 3, 2)), np.zeros((4, 3)), ["a", "b", "c"])
    with pytest.raises(ShapEditorialError, match="multiclass"):
        se.scatter(expl)


def test_scatter_rejects_non_numeric_feature():
    data = np.array([["low"], ["high"], ["low"]], dtype=object)
    expl = FakeExplanation(np.array([[1.0], [2.0], [3.0]]), data, ["grade"])
    with pytest.raises(ShapEditorialError, match="non-numeric"):
        se.scatter(expl, feature="grade")


def test_scatter_colour_feature_maps_to_colours():
    expl = make_explanation(n_samples=20)
    fig, ax = se.scatter(expl, feature="feat_0", color="feat_3")
    # A colour array (one value per point) rather than a single flat colour.
    assert ax.collections[0].get_array() is not None
    assert "feat_3" in [a.get_title(loc="left") for a in fig.axes]


def test_scatter_without_colour_has_no_colour_key():
    expl = make_explanation()
    fig, ax = se.scatter(expl, feature="feat_0")
    assert ax.collections[0].get_array() is None
    assert len(fig.axes) == 1  # no extra colorbar axes


def test_scatter_unknown_colour_feature():
    with pytest.raises(ShapEditorialError, match="No feature named"):
        se.scatter(make_explanation(), feature=0, color="nope")


def test_scatter_auto_analysis_reports_direction():
    x = np.linspace(0, 10, 30)
    values = np.column_stack([x - 5])
    data = np.column_stack([x])
    expl = FakeExplanation(values, data, ["rising"])
    fig, ax = se.scatter(expl, feature="rising")
    assert any("pushes the prediction higher" in t for t in fig_texts(fig))


def test_scatter_custom_analysis_string_is_used():
    fig, ax = se.scatter(make_explanation(), feature=0, analysis="My own take.")
    assert "My own take." in fig_texts(fig)


def test_scatter_analysis_false_omits_line():
    fig, ax = se.scatter(make_explanation(), feature=0, analysis=False, title="T")
    assert fig_texts(fig) == ["T"]


def test_scatter_direction_labels_present_and_customizable():
    fig, ax = se.scatter(make_explanation(), feature=0, analysis=False)
    axes_texts = [t.get_text() for t in ax.texts]
    assert any("higher" in t for t in axes_texts)

    fig, ax = se.scatter(
        make_explanation(),
        feature=0,
        analysis=False,
        direction_labels=("up is good", "down is bad"),
    )
    assert [t.get_text() for t in ax.texts] == ["up is good", "down is bad"]


def test_scatter_direction_labels_false_omits_them():
    fig, ax = se.scatter(make_explanation(), feature=0, direction_labels=False)
    assert len(ax.texts) == 0


def test_scatter_constant_feature_gives_no_analysis_line():
    values = np.column_stack([np.linspace(0, 1, 10)])
    data = np.column_stack([np.full(10, 3.0)])
    expl = FakeExplanation(values, data, ["flat"])
    fig, ax = se.scatter(expl, feature="flat", title="T")
    assert fig_texts(fig) == ["T"]


def test_scatter_all_nan_feature_column_still_renders():
    values = np.column_stack([np.linspace(0, 1, 10)])
    data = np.column_stack([np.full(10, np.nan)])
    expl = FakeExplanation(values, data, ["missing"])
    fig, ax = se.scatter(expl, feature="missing")
    assert ax.get_xlabel() == "missing"


def test_scatter_transparent_background():
    fig, ax = se.scatter(make_explanation(), feature=0, transparent=True)
    assert fig.get_facecolor()[3] == 0


def test_scatter_does_not_leak_rcparams():
    before = mpl.rcParams["axes.labelsize"]
    se.scatter(make_explanation(), feature=0, transparent=True)
    assert mpl.rcParams["axes.labelsize"] == before
    assert mpl.rcParams["savefig.transparent"] is False


def test_jitter_applied_to_low_cardinality_only():
    many = np.linspace(0, 1, 100)
    assert _jitter_width(many, None, 100) == 0.0

    few = np.repeat([1.0, 2.0, 3.0], 20)
    assert _jitter_width(few, None, 60) > 0.0


def test_jitter_can_be_disabled_and_forced():
    few = np.repeat([1.0, 2.0, 3.0], 20)
    assert _jitter_width(few, 0, 60) == 0.0

    many = np.linspace(0, 1, 100)
    assert _jitter_width(many, 0.5, 100) > 0.0


def test_jitter_width_handles_all_nan():
    assert _jitter_width(np.full(5, np.nan), None, 5) == 0.0


def test_analysis_line_none_when_too_few_points():
    assert _analysis_line(np.array([1.0, 2.0]), np.array([1.0, 2.0]), "x") is None


def test_analysis_line_reports_mixed_for_uncorrelated():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    line = _analysis_line(x, y, "noise")
    assert "mixed" in line


def test_analysis_line_negative_direction():
    x = np.linspace(0, 10, 30)
    line = _analysis_line(x, -x, "falling")
    assert "pushes the prediction lower" in line
