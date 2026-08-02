"""Helpers for pulling plain arrays out of a shap.Explanation object.

We deliberately don't import shap at module load time — this package
only needs duck-typed access to `.values`, `.data`, and `.feature_names`,
so it works against any object shaped like a shap.Explanation without
forcing a hard dependency at import time.
"""

from __future__ import annotations

import numpy as np


class ShapEditorialError(ValueError):
    """Raised when the input doesn't look like a usable SHAP explanation."""


def extract_explanation(shap_values, feature_names=None):
    """Return (values, data, feature_names) as plain numpy arrays.

    Parameters
    ----------
    shap_values : shap.Explanation (or duck-typed equivalent)
        Must expose `.values` and, ideally, `.data` and `.feature_names`.
    feature_names : list[str] | None
        Overrides names found on the explanation object, if given.

    Notes
    -----
    For multiclass explanations (values.ndim == 3), the last axis is
    assumed to be the class axis, and the caller must slice a class
    before calling this — we raise a clear error instead of guessing.
    """
    if not hasattr(shap_values, "values"):
        raise ShapEditorialError(
            "Expected a shap.Explanation object (or something exposing "
            "`.values`), got: " + type(shap_values).__name__
        )

    values = np.asarray(shap_values.values)

    if values.ndim == 3:
        raise ShapEditorialError(
            "Got a multiclass explanation (values has 3 dimensions: "
            f"shape={values.shape}). Select one class first, e.g. "
            "`shap_values[..., class_index]`, then pass that in."
        )
    if values.ndim != 2:
        raise ShapEditorialError(
            f"Expected a 2D array of shape (n_samples, n_features), got "
            f"shape={values.shape}."
        )

    data = getattr(shap_values, "data", None)
    data = np.asarray(data) if data is not None else None

    names = feature_names
    if names is None:
        names = getattr(shap_values, "feature_names", None)
    if names is None:
        names = [f"Feature {i}" for i in range(values.shape[1])]
    names = list(names)

    if len(names) != values.shape[1]:
        raise ShapEditorialError(
            f"feature_names has {len(names)} entries but values has "
            f"{values.shape[1]} columns."
        )

    return values, data, names


def top_feature_order(values: np.ndarray, max_display: int):
    """Return (kept_idx, other_idx) sorted by mean |SHAP value|, descending.

    `kept_idx` has at most `max_display` entries. Everything else goes
    in `other_idx`, to be collapsed into a single "N other features" row.
    """
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]  # descending
    kept_idx = order[:max_display]
    other_idx = order[max_display:]
    return kept_idx, other_idx
