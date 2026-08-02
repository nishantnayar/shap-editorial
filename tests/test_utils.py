import numpy as np
import pytest
from _helpers import FakeExplanation

from shap_editorial._utils import (
    ShapEditorialError,
    extract_explanation,
    extract_single_explanation,
    normalize_column,
    resolve_feature,
    top_feature_order,
)


def test_extract_explanation_basic():
    values = np.array([[1.0, -2.0], [0.5, 1.5]])
    data = np.array([[10, 20], [11, 21]])
    exp = FakeExplanation(values, data, ["a", "b"])

    v, d, names = extract_explanation(exp)
    assert v.shape == (2, 2)
    assert names == ["a", "b"]
    np.testing.assert_array_equal(d, data)


def test_extract_explanation_missing_names_generates_defaults():
    values = np.zeros((3, 2))
    exp = FakeExplanation(values)
    v, d, names = extract_explanation(exp)
    assert names == ["Feature 0", "Feature 1"]
    assert d is None


def test_extract_explanation_name_override():
    values = np.zeros((2, 2))
    exp = FakeExplanation(values, feature_names=["x", "y"])
    _, _, names = extract_explanation(exp, feature_names=["custom_a", "custom_b"])
    assert names == ["custom_a", "custom_b"]


def test_extract_explanation_rejects_non_explanation():
    with pytest.raises(ShapEditorialError):
        extract_explanation([1, 2, 3])


def test_extract_explanation_rejects_multiclass():
    values = np.zeros((5, 3, 2))  # (samples, features, classes)
    exp = FakeExplanation(values)
    with pytest.raises(ShapEditorialError, match="multiclass"):
        extract_explanation(exp)


def test_extract_explanation_rejects_1d_values():
    exp = FakeExplanation(np.zeros(4))
    with pytest.raises(ShapEditorialError, match="2D array"):
        extract_explanation(exp)


def test_extract_explanation_rejects_wrong_name_count():
    values = np.zeros((2, 3))
    exp = FakeExplanation(values, feature_names=["only_one"])
    with pytest.raises(ShapEditorialError, match="feature_names has"):
        extract_explanation(exp)


# -- extract_single_explanation ----------------------------------------


def test_extract_single_1d():
    values = np.array([0.3, -0.2, 0.1])
    data = np.array([1.0, 2.0, 3.0])
    exp = FakeExplanation(values, data, ["a", "b", "c"], base_values=0.4)

    v, d, base, names = extract_single_explanation(exp)
    assert v.shape == (3,)
    assert d.shape == (3,)
    assert base == pytest.approx(0.4)
    assert names == ["a", "b", "c"]


def test_extract_single_squeezes_2d():
    exp = FakeExplanation(
        np.array([[0.3, -0.2]]),
        data=np.array([[1.0, 2.0]]),
        feature_names=["a", "b"],
        base_values=0.1,
    )
    v, d, _, _ = extract_single_explanation(exp)
    assert v.shape == (2,)
    assert d.shape == (2,)


def test_extract_single_base_as_scalar():
    exp = FakeExplanation(np.zeros(3), base_values=0.25)
    _, _, base, _ = extract_single_explanation(exp)
    assert isinstance(base, float)
    assert base == pytest.approx(0.25)


def test_extract_single_base_as_size_one_array():
    exp = FakeExplanation(np.zeros(3), base_values=np.array([0.25]))
    _, _, base, _ = extract_single_explanation(exp)
    assert base == pytest.approx(0.25)


def test_extract_single_generates_default_names():
    exp = FakeExplanation(np.zeros(2), base_values=0.0)
    _, _, _, names = extract_single_explanation(exp)
    assert names == ["Feature 0", "Feature 1"]


def test_extract_single_rejects_non_explanation():
    with pytest.raises(ShapEditorialError):
        extract_single_explanation([1, 2, 3])


def test_extract_single_rejects_multiclass():
    exp = FakeExplanation(np.zeros((5, 3, 2)))
    with pytest.raises(ShapEditorialError, match="multiclass"):
        extract_single_explanation(exp)


def test_extract_single_rejects_multi_sample():
    exp = FakeExplanation(np.zeros((5, 3)), base_values=0.0)
    with pytest.raises(ShapEditorialError, match="single prediction"):
        extract_single_explanation(exp)


def test_extract_single_rejects_missing_base_values():
    exp = FakeExplanation(np.zeros(3))
    with pytest.raises(ShapEditorialError, match="base_values"):
        extract_single_explanation(exp)


def test_extract_single_rejects_multi_output_base_values():
    exp = FakeExplanation(np.zeros(3), base_values=np.array([0.3, 0.7]))
    with pytest.raises(ShapEditorialError, match="multiple outputs"):
        extract_single_explanation(exp)


def test_extract_single_rejects_wrong_name_count():
    exp = FakeExplanation(np.zeros(3), feature_names=["only_one"], base_values=0.0)
    with pytest.raises(ShapEditorialError, match="feature_names has"):
        extract_single_explanation(exp)


# -- top_feature_order -------------------------------------------------


def test_top_feature_order_sorts_by_mean_abs_descending():
    # feature 0: small impact, feature 1: large impact, feature 2: medium
    values = np.array(
        [
            [0.1, 5.0, 1.0],
            [-0.1, -4.0, -1.5],
        ]
    )
    kept, other = top_feature_order(values, max_display=2)
    assert list(kept) == [1, 2]
    assert list(other) == [0]


def test_top_feature_order_max_display_larger_than_features():
    values = np.array([[1.0, 2.0], [1.0, 2.0]])
    kept, other = top_feature_order(values, max_display=10)
    assert len(kept) == 2
    assert len(other) == 0


def test_top_feature_order_single_feature():
    kept, other = top_feature_order(np.array([[1.0], [2.0]]), max_display=1)
    assert list(kept) == [0]
    assert len(other) == 0


def test_top_feature_order_zero_max_display():
    # The chart functions reject max_display=0 before reaching here, but the
    # helper itself should still partition cleanly rather than misbehave.
    kept, other = top_feature_order(np.array([[1.0, 2.0]]), max_display=0)
    assert len(kept) == 0
    assert len(other) == 2


def test_normalize_column_scales_to_unit_range():
    scaled = normalize_column(np.array([0.0, 5.0, 10.0]))
    np.testing.assert_allclose(scaled, [0.0, 0.5, 1.0])


def test_normalize_column_all_nan_returns_midscale():
    assert np.all(normalize_column(np.full(5, np.nan)) == 0.5)


def test_normalize_column_constant_returns_midscale():
    assert np.all(normalize_column(np.full(5, 3.0)) == 0.5)


def test_resolve_feature_by_name_and_index():
    names = ["age", "income", "score"]
    assert resolve_feature(names, "income") == 1
    assert resolve_feature(names, 2) == 2
    assert resolve_feature(names, -1) == 2


def test_resolve_feature_unknown_name():
    with pytest.raises(ShapEditorialError, match="No feature named"):
        resolve_feature(["age", "income"], "salary")


def test_resolve_feature_index_out_of_range():
    with pytest.raises(ShapEditorialError, match="out of range"):
        resolve_feature(["age", "income"], 5)


def test_resolve_feature_duplicate_name():
    with pytest.raises(ShapEditorialError, match="matches 2 columns"):
        resolve_feature(["age", "age", "income"], "age")


def test_resolve_feature_rejects_bool():
    # bool is a subclass of int, so True must not silently mean column 1.
    with pytest.raises(ShapEditorialError, match="No feature named"):
        resolve_feature(["age", "income"], True)
