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
    """Return (values, data, feature_names) as plain numpy arrays from a
    duck-typed shap.Explanation. `feature_names` overrides the object's own.
    Multiclass (ndim 3) and non-2D inputs raise ShapEditorialError rather than
    being guessed at."""
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


def extract_single_explanation(shap_values, feature_names=None):
    """Return (values, data, base_value, names) for a *single* prediction.

    The waterfall chart explains one instance, so this expects a 1-D `.values`
    (n_features,) — or a single-row 2-D array, which is squeezed. It also needs
    `.base_values` (E[f(x)]), the starting point the contributions build from.

    Raises `ShapEditorialError` (never guesses) for multiclass explanations, for
    multi-sample inputs, or when `.base_values` is missing / multi-output.
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
    if values.ndim == 2:
        if values.shape[0] == 1:
            values = values[0]
        else:
            raise ShapEditorialError(
                "waterfall explains a single prediction, but got "
                f"{values.shape[0]} samples (shape={values.shape}). Select one "
                "first, e.g. `explanation[0]`."
            )
    if values.ndim != 1:
        raise ShapEditorialError(
            f"Expected a 1D array of shape (n_features,), got shape={values.shape}."
        )

    n_features = values.shape[0]

    data = getattr(shap_values, "data", None)
    if data is not None:
        data = np.asarray(data)
        if data.ndim == 2 and data.shape[0] == 1:
            data = data[0]

    base = getattr(shap_values, "base_values", None)
    if base is None:
        raise ShapEditorialError(
            "waterfall needs `.base_values` (E[f(x)]) — the value the "
            "contributions build from. Pass an Explanation from calling the "
            "explainer on your data, e.g. `explainer(X)[0]`."
        )
    base = np.asarray(base)
    if base.ndim >= 1:
        if base.size == 1:
            base = base.reshape(-1)[0]
        else:
            raise ShapEditorialError(
                "`.base_values` has multiple outputs "
                f"(shape={base.shape}); select one class first."
            )
    base = float(base)

    names = feature_names
    if names is None:
        names = getattr(shap_values, "feature_names", None)
    if names is None:
        names = [f"Feature {i}" for i in range(n_features)]
    names = list(names)

    if len(names) != n_features:
        raise ShapEditorialError(
            f"feature_names has {len(names)} entries but values has "
            f"{n_features} features."
        )

    return values, data, base, names


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
