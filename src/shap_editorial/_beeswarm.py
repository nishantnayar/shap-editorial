"""Publication-ready beeswarm plot for a shap.Explanation object."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

from ._finalize import finalize
from ._theme import C_GRID, C_HIGH, C_LOW, C_MID, C_OTHER_BAR, C_SPINE, set_theme
from ._utils import extract_explanation, top_feature_order

_CMAP = LinearSegmentedColormap.from_list("shap_editorial", [C_LOW, C_MID, C_HIGH])


def _row_jitter(n, row_height=0.7, rng=None):
    rng = rng or np.random.default_rng(0)
    if n == 0:
        return np.zeros(0)
    # Slight vertical jitter so overlapping points fan out into a "beeswarm"
    # shape rather than a flat line, without needing full swarm packing.
    return rng.uniform(-row_height / 2, row_height / 2, size=n)


def beeswarm(
    shap_values,
    *,
    max_display: int = 10,
    title: str | None = None,
    subtitle: str | None = "SHAP value (impact on model output)",
    source: str | None = None,
    feature_names=None,
    figsize=(8, 5.5),
    ax=None,
):
    """Render an editorial-style beeswarm plot of SHAP values.

    Parameters
    ----------
    shap_values : shap.Explanation
        The result of calling a shap Explainer on your data, e.g.
        `explainer(X_test)`. Must be a single-output (binary/regression)
        explanation — for multiclass models, slice a class first.
    max_display : int
        Number of individual features to show before collapsing the
        remainder into a single "N other features" row.
    title, subtitle, source : str | None
        Editorial title stack. `subtitle` defaults to a plain-language
        description of what the x-axis means; pass None to omit it.
    feature_names : list[str] | None
        Overrides names on the explanation object, if provided.
    figsize : tuple
        Only used if `ax` is None (a new figure is created).
    ax : matplotlib.axes.Axes | None
        Draw onto an existing axes instead of creating a new figure.

    Returns
    -------
    (fig, ax) : the created or given figure and axes.
    """
    set_theme()

    values, data, names = extract_explanation(shap_values, feature_names)
    n_samples, n_features = values.shape

    if data is None:
        raise ValueError(
            "This explanation has no `.data` (the original feature values), "
            "which the beeswarm plot needs for its colour scale. Pass an "
            "Explanation created from calling the explainer on your data, "
            "e.g. `explainer(X_test)`, not just raw SHAP value arrays."
        )

    kept_idx, other_idx = top_feature_order(values, max_display)
    n_rows = len(kept_idx) + (1 if len(other_idx) else 0)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    rng = np.random.default_rng(0)

    # Normalize each feature's raw values to [0, 1] independently so the
    # colour scale is meaningful per-feature, matching SHAP's own convention.
    def _norm(col):
        col = col.astype(float)
        lo, hi = np.nanmin(col), np.nanmax(col)
        if hi - lo < 1e-12:
            return np.full_like(col, 0.5)
        return (col - lo) / (hi - lo)

    row_labels = []
    for row, feat_idx in enumerate(kept_idx[::-1]):  # largest impact at top
        y = row
        v = values[:, feat_idx]
        colour_val = _norm(data[:, feat_idx])
        jitter = _row_jitter(n_samples, rng=rng)
        ax.scatter(
            v,
            np.full(n_samples, y) + jitter,
            c=colour_val,
            cmap=_CMAP,
            vmin=0,
            vmax=1,
            s=14,
            linewidths=0,
            alpha=0.85,
        )
        row_labels.append(names[feat_idx])

    if len(other_idx):
        y = len(kept_idx)
        # Sum (not mean) of the excluded features' SHAP values per sample,
        # so the row still reflects each sample's true net contribution
        # from everything not individually displayed.
        other_sum = values[:, other_idx].sum(axis=1)
        jitter = _row_jitter(n_samples, rng=rng)
        ax.scatter(
            other_sum,
            np.full(n_samples, y) + jitter,
            c=C_OTHER_BAR,
            s=12,
            linewidths=0,
            alpha=0.6,
        )
        row_labels.append(f"{len(other_idx)} other features")

    ax.axvline(0, color=C_SPINE, linewidth=0.9, zorder=0)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xlabel("")
    ax.grid(axis="x", color=C_GRID, linewidth=0.7, zorder=-1)
    ax.set_axisbelow(True)

    # Slim colour legend: Low -> High feature value.
    sm = ScalarMappable(cmap=_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, aspect=30)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value", fontsize=8.5)
    cbar.outline.set_visible(False)

    fig.subplots_adjust(top=0.80, left=0.28, right=0.90, bottom=0.10)
    finalize(fig, ax, title=title, subtitle=subtitle, source=source)

    return fig, ax
