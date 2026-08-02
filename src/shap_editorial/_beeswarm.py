"""Publication-ready beeswarm plot for a shap.Explanation object."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

from ._finalize import ROW_HEIGHT_IN, finalize, highlight_row, title_block_height
from ._theme import (
    C_GRID,
    C_HIGH,
    C_LABEL_MUTED,
    C_LOW,
    C_MID,
    C_OTHER_BAR,
    C_ROW_GUIDE,
    C_ZERO,
    set_theme,
)
from ._utils import ShapEditorialError, extract_explanation, top_feature_order

# Weighted stops: grey holds through the low half of the scale, and red is
# reserved for the top ~20% — so only genuinely high feature values pop and
# everything else recedes to grey.
_CMAP = LinearSegmentedColormap.from_list(
    "shap_editorial",
    [(0.0, C_LOW), (0.5, C_LOW), (0.8, C_MID), (1.0, C_HIGH)],
)


def _analysis_line(values, data, names, kept_idx):
    """One-sentence takeaway derived from the top driver's SHAP pattern.

    Reports the strongest feature's direction of effect from the correlation
    between each sample's feature value and its SHAP value — narrating the
    colour<->position pattern already in the plot. Returns None rather than
    guess when the feature is flat or degenerate.
    """
    if len(kept_idx) == 0:
        return None
    top = kept_idx[0]
    fv = np.asarray(data[:, top], dtype=float)
    sv = np.asarray(values[:, top], dtype=float)
    mask = np.isfinite(fv) & np.isfinite(sv)
    if mask.sum() < 3 or np.std(fv[mask]) < 1e-12 or np.std(sv[mask]) < 1e-12:
        return None
    r = float(np.corrcoef(fv[mask], sv[mask])[0, 1])
    if not np.isfinite(r):
        return None
    name = names[top]
    if abs(r) < 0.15:
        return (
            f"“{name}” is the strongest driver, though its effect direction is mixed."
        )
    direction = "higher" if r > 0 else "lower"
    return (
        f"“{name}” is the strongest driver: "
        f"higher values push the prediction {direction}."
    )


def _row_jitter(n, rng):
    # Fans overlapping points out into a "beeswarm" shape without paying for
    # full swarm packing.
    return rng.uniform(-0.35, 0.35, size=n)


def _norm(col):
    """Scale one feature's raw values to [0, 1] for the colour map.

    Per-feature (not global) normalization, matching SHAP's own convention.
    Constant and all-NaN columns collapse to mid-scale rather than producing
    NaN colours.
    """
    col = col.astype(float)
    finite = np.isfinite(col)
    if not finite.any():
        return np.full_like(col, 0.5)
    lo, hi = col[finite].min(), col[finite].max()
    if hi - lo < 1e-12:
        return np.full_like(col, 0.5)
    return (col - lo) / (hi - lo)


def beeswarm(
    shap_values,
    *,
    max_display: int = 10,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    feature_names=None,
    figsize=None,
    show_other: bool = False,
    analysis: bool | str = True,
    highlight: bool = True,
    direction_labels: bool | tuple[str, str] = True,
    transparent: bool = False,
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
        Number of top features (by mean |SHAP|) to show. Must be at least 1.
    show_other : bool
        If True, collapse every feature beyond `max_display` into a single
        "N other features" row at the bottom (the per-sample sum of their
        SHAP values, preserving the additive property). Defaults to False:
        just show the top `max_display` features. The aggregate row can't
        carry the feature-value colour scale (it sums across features), so
        it renders in a flat grey — kept as an opt-in for completeness.
    analysis : bool | str
        Editorial takeaway line under the title. True (default) auto-generates
        a one-sentence insight from the top driver's SHAP pattern; pass a
        string to supply your own; pass False to omit it.
    highlight : bool
        If True (default), draw a dotted outline around the top-driver row and
        bold its label, so the eye lands on the strongest feature.
    direction_labels : bool | tuple[str, str]
        Small labels under the x-axis telling the reader what each side means.
        True (default) shows the generic "← pushes prediction lower /
        pushes prediction higher →". Pass a (left, right) tuple for
        domain-specific wording (e.g. ("← toward benign", "toward malignant →")),
        or False to omit them.
    title, subtitle, source : str | None
        Editorial title stack. `subtitle` defaults to None — the analysis line
        and directional axis labels now describe the x-axis, so a "SHAP value"
        subtitle is redundant; pass one explicitly if you want the metric named.
    feature_names : list[str] | None
        Overrides names on the explanation object, if provided.
    figsize : tuple | None
        Only used if `ax` is None. Defaults to None, in which case the height is
        chosen automatically from the number of rows (so few- and many-feature
        charts both look tight); pass an explicit (w, h) to override.
    transparent : bool
        If True, render (and save) with a transparent background instead of
        white — useful for coloured slides or dark web pages. Save to a format
        with an alpha channel (PNG, SVG, PDF); JPEG has no transparency and will
        flatten the background. Ignored when drawing onto an existing `ax`.
    ax : matplotlib.axes.Axes | None
        Draw onto an existing axes instead of creating a new figure. Intended
        for a single-axes figure you manage: the call still adjusts figure
        margins and adds figure-level elements (the red tab, title block,
        colour key, and axis labels) positioned for that layout, so it is not
        suitable for embedding as one panel in a multi-panel figure.

    Returns
    -------
    (fig, ax) : the created or given figure and axes.

    Notes
    -----
    Returning `(fig, ax)` means you can save in any matplotlib format:
    `fig.savefig("chart.png" | ".jpg" | ".svg" | ".pdf", dpi=200,
    bbox_inches="tight")`. Use SVG/PDF for resolution-independent print output.

    The editorial theme is applied inside a matplotlib rc context, so your own
    global rcParams survive the call untouched.
    """
    if max_display < 1:
        raise ShapEditorialError(f"max_display must be at least 1, got {max_display}.")

    values, data, names = extract_explanation(shap_values, feature_names)
    n_samples = values.shape[0]

    if data is None:
        raise ShapEditorialError(
            "This explanation has no `.data` (the original feature values), "
            "which the beeswarm plot needs for its colour scale. Pass an "
            "Explanation created from calling the explainer on your data, "
            "e.g. `explainer(X_test)`, not just raw SHAP value arrays."
        )

    kept_idx, other_idx = top_feature_order(values, max_display)
    has_other = show_other and len(other_idx) > 0
    n_rows = len(kept_idx) + (1 if has_other else 0)

    # Resolve the takeaway text and directional labels up front, so the title
    # block and bottom margin can be sized before the figure is created.
    analysis_text = None
    if analysis:
        analysis_text = (
            analysis
            if isinstance(analysis, str)
            else _analysis_line(values, data, names, kept_idx)
        )
    dir_labels = None
    if direction_labels:
        dir_labels = (
            ("← pushes prediction lower", "pushes prediction higher →")
            if direction_labels is True
            else tuple(direction_labels)
        )

    # Inch-based margins: reserve room for the title block on top and the tick /
    # directional / source labels below, then give each data row a fixed height.
    top_in = title_block_height(title=title, subtitle=subtitle, analysis=analysis_text)
    bottom_in = 0.9 if dir_labels else 0.5

    with mpl.rc_context():
        set_theme(transparent=transparent)

        if ax is None:
            if figsize is None:
                figsize = (8.0, top_in + ROW_HEIGHT_IN * n_rows + bottom_in)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        rng = np.random.default_rng(0)

        base = 1 if has_other else 0  # bottom row belongs to the aggregate
        row_labels = [""] * n_rows

        for i, feat_idx in enumerate(kept_idx[::-1]):  # largest impact ends up on top
            y = base + i
            v = values[:, feat_idx]
            colour_val = _norm(data[:, feat_idx])
            jitter = _row_jitter(n_samples, rng)
            # Draw low-impact points first so the points that matter most sit on
            # top instead of being buried under the dense cluster near zero.
            order = np.argsort(np.abs(v))
            ax.scatter(
                v[order],
                np.full(n_samples, y) + jitter[order],
                c=colour_val[order],
                cmap=_CMAP,
                vmin=0,
                vmax=1,
                s=14,
                linewidths=0,
                alpha=0.6,
            )
            row_labels[y] = names[feat_idx]

        if has_other:
            # Sum (not mean) of the excluded features per sample, so the row
            # still reflects each sample's true net contribution from everything
            # not individually displayed.
            other_sum = values[:, other_idx].sum(axis=1)
            jitter = _row_jitter(n_samples, rng)
            order = np.argsort(np.abs(other_sum))
            ax.scatter(
                other_sum[order],
                jitter[order],
                c=C_OTHER_BAR,
                s=12,
                linewidths=0,
                alpha=0.5,
            )
            row_labels[0] = f"{len(other_idx)} other features"

        # Faint leader lines tie each label to its cloud of points, restoring the
        # label<->row connection lost when the tick dashes were removed. Drawn
        # behind the points so they're occluded where the points are dense.
        for r in range(n_rows):
            ax.axhline(r, color=C_ROW_GUIDE, linewidth=0.8, zorder=-2)

        # Just under the points, so it marks x=0 without slashing across the data.
        ax.axvline(0, color=C_ZERO, linewidth=1.0, zorder=0.5)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels)
        ax.set_ylim(-0.6, n_rows - 0.4)
        ax.grid(axis="x", color=C_GRID, linewidth=0.6, zorder=-1)
        ax.set_axisbelow(True)

        # Convert the inch-based margins to figure fractions for this height.
        height = fig.get_figheight()
        fig.subplots_adjust(
            top=1.0 - top_in / height,
            bottom=bottom_in / height,
            left=0.28,
            right=0.95,
        )

        if highlight and len(kept_idx):
            highlight_row(fig, ax, n_rows - 1)

        # Horizontal colour key near the top-right. A horizontal bar with
        # horizontal labels reads better than a vertical colorbar with a rotated
        # axis label (which forces a head-tilt).
        sm = ScalarMappable(cmap=_CMAP)
        sm.set_array([])
        cax = fig.add_axes([0.77, 1.0 - 0.55 / height, 0.18, 0.11 / height])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"])
        cbar.ax.tick_params(length=0, labelsize=8.5, colors=C_LABEL_MUTED)
        cbar.outline.set_visible(False)
        cax.set_title(
            "Feature value", fontsize=8.5, color=C_LABEL_MUTED, loc="left", pad=4
        )

        # The single clearest cue for "which side means what".
        if dir_labels:
            left_txt, right_txt = dir_labels
            y = 0.42 / height
            fig.text(
                0.28,
                y,
                left_txt,
                ha="left",
                va="bottom",
                fontsize=9,
                color=C_LABEL_MUTED,
            )
            fig.text(
                0.95,
                y,
                right_txt,
                ha="right",
                va="bottom",
                fontsize=9,
                color=C_LABEL_MUTED,
            )

        finalize(
            fig,
            ax,
            title=title,
            subtitle=subtitle,
            source=source,
            analysis=analysis_text,
        )

    return fig, ax
