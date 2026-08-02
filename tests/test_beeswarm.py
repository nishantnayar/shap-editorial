import matplotlib.pyplot as plt
import numpy as np
import pytest
from _helpers import FakeExplanation, make_explanation

import shap_editorial as se
from shap_editorial._beeswarm import _analysis_line, _norm
from shap_editorial._utils import ShapEditorialError


def fig_texts(fig):
    return [t.get_text() for t in fig.texts]


def test_beeswarm_runs_without_error():
    fig, ax = se.beeswarm(make_explanation(), title="Test chart")
    assert ax.get_figure() is fig
    assert len(ax.get_yticklabels()) == 6


def test_beeswarm_respects_max_display():
    fig, ax = se.beeswarm(make_explanation(n_features=8), max_display=3)
    # By default only the top max_display features are shown (no "other" row).
    assert len(ax.get_yticklabels()) == 3


def test_beeswarm_show_other_adds_bottom_row():
    exp = make_explanation(n_features=8)
    fig, ax = se.beeswarm(exp, max_display=3, show_other=True)
    # 3 kept features + 1 collapsed "other" row = 4 y-ticks
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == 4
    # The aggregate row sits at the bottom (first y-tick).
    assert labels[0] == "5 other features"


def test_beeswarm_no_other_row_when_features_fit():
    fig, ax = se.beeswarm(make_explanation(n_features=3), max_display=5)
    assert len(ax.get_yticklabels()) == 3


def test_beeswarm_raises_without_data():
    exp = FakeExplanation(np.zeros((5, 3)), data=None, feature_names=["a", "b", "c"])
    with pytest.raises(ShapEditorialError, match="no `.data`"):
        se.beeswarm(exp)


def test_beeswarm_raises_on_multiclass_input():
    exp = FakeExplanation(np.zeros((5, 3, 2)))
    with pytest.raises(ShapEditorialError, match="multiclass"):
        se.beeswarm(exp)


def test_beeswarm_raises_on_zero_max_display():
    with pytest.raises(ShapEditorialError, match="max_display must be at least 1"):
        se.beeswarm(make_explanation(), max_display=0)


def test_beeswarm_draws_onto_given_axes():
    fig, ax = plt.subplots()
    fig2, ax2 = se.beeswarm(make_explanation(), ax=ax)
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


def test_beeswarm_analysis_false_omits_line():
    fig, ax = se.beeswarm(make_explanation(), analysis=False)
    assert not any("strongest driver" in t for t in fig_texts(fig))


def test_beeswarm_custom_analysis_string_is_used():
    fig, ax = se.beeswarm(make_explanation(), analysis="Bespoke takeaway.")
    assert "Bespoke takeaway." in fig_texts(fig)


def test_beeswarm_direction_labels_false_omits_them():
    fig, ax = se.beeswarm(make_explanation(), direction_labels=False)
    assert not any("pushes prediction" in t for t in fig_texts(fig))


def test_beeswarm_custom_direction_labels():
    fig, ax = se.beeswarm(
        make_explanation(), direction_labels=("← toward benign", "toward malignant →")
    )
    texts = fig_texts(fig)
    assert "← toward benign" in texts
    assert "toward malignant →" in texts


def test_beeswarm_highlight_false_draws_no_band():
    fig, ax = se.beeswarm(make_explanation(), highlight=False)
    assert len(ax.patches) == 0  # the highlight band is the only patch beeswarm adds


def test_beeswarm_highlight_true_draws_one_band():
    fig, ax = se.beeswarm(make_explanation(), highlight=True)
    assert len(ax.patches) == 1


def test_beeswarm_single_sample():
    exp = make_explanation(n_samples=1, n_features=4)
    fig, ax = se.beeswarm(exp)
    assert len(ax.get_yticklabels()) == 4


def test_beeswarm_single_feature():
    exp = make_explanation(n_samples=20, n_features=1)
    fig, ax = se.beeswarm(exp)
    assert len(ax.get_yticklabels()) == 1


def test_beeswarm_all_nan_feature_column_still_renders():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(20, 3))
    data = rng.uniform(size=(20, 3))
    data[:, 0] = np.nan  # a feature whose raw values are entirely missing
    exp = FakeExplanation(values, data, ["nan_feat", "b", "c"])
    fig, ax = se.beeswarm(exp)
    assert len(ax.get_yticklabels()) == 3


def test_beeswarm_constant_feature_column_still_renders():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(20, 2))
    data = np.ones((20, 2))  # no spread at all in the raw feature values
    exp = FakeExplanation(values, data, ["a", "b"])
    fig, ax = se.beeswarm(exp)
    assert len(ax.get_yticklabels()) == 2


def test_beeswarm_transparent_background():
    fig, ax = se.beeswarm(make_explanation(), transparent=True)
    assert fig.get_facecolor()[3] == 0.0  # fully transparent alpha channel


def test_beeswarm_does_not_leak_rcparams():
    import matplotlib as mpl

    before = mpl.rcParams["axes.labelsize"]
    se.beeswarm(make_explanation(), transparent=True)
    assert mpl.rcParams["axes.labelsize"] == before
    assert mpl.rcParams["savefig.transparent"] is False


def test_norm_scales_to_unit_range():
    np.testing.assert_allclose(_norm(np.array([0.0, 5.0, 10.0])), [0.0, 0.5, 1.0])


def test_norm_all_nan_column_returns_midscale():
    assert np.all(_norm(np.full(5, np.nan)) == 0.5)


def test_norm_constant_column_returns_midscale():
    assert np.all(_norm(np.full(5, 3.0)) == 0.5)


def test_analysis_line_none_for_flat_feature():
    values = np.zeros((10, 2))
    data = np.ones((10, 2))  # no variance, so no honest direction to report
    assert _analysis_line(values, data, ["a", "b"], np.array([0])) is None


def test_analysis_line_reports_positive_direction():
    n = 30
    fv = np.linspace(0, 1, n)
    values = np.column_stack([fv * 2.0, np.zeros(n)])  # SHAP tracks feature value
    data = np.column_stack([fv, np.zeros(n)])
    line = _analysis_line(values, data, ["driver", "b"], np.array([0]))
    assert "driver" in line
    assert "higher" in line
