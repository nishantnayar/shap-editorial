import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shap_editorial._utils import (
    ShapEditorialError,
    extract_explanation,
    top_feature_order,
)
from _helpers import FakeExplanation


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


def test_extract_explanation_rejects_wrong_name_count():
    values = np.zeros((2, 3))
    exp = FakeExplanation(values, feature_names=["only_one"])
    with pytest.raises(ShapEditorialError, match="feature_names has"):
        extract_explanation(exp)


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
