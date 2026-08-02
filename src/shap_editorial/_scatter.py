"""Publication-ready dependence scatter plot for a shap.Explanation object."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable

from ._finalize import finalize, title_block_height
from ._theme import (
    C_ECON_RED,
    C_GRID,
    C_LABEL_MUTED,
    C_ZERO,
    FEATURE_CMAP,
    set_theme,
)
from ._utils import (
    ShapEditorialError,
    extract_explanation,
    normalize_column,
    resolve_feature,
    top_feature_order,
)

# Plot area height in inches, excluding the title block and bottom labels. The
# row-based charts size themselves by row count; a scatter has no rows, so it
# gets a fixed, roughly landscape plot area instead.
_PLOT_HEIGHT_IN = 4.2

# A feature with at most this many distinct values reads as categorical or
# integer-coded, and its points would stack into vertical stripes without jitter.
_LOW_CARDINALITY = 10


def _analysis_line(x, y, name):
    """One-sentence takeaway from the feature's direction of effect.

    Reports the sign of the correlation between the feature's values and its
    SHAP values - narrating the slope already visible in the plot. Returns None
    rather than guess when the feature is flat or degenerate.

    Deliberately says nothing about *where* the effect crosses zero: a crossing
    point would be a fitted quantity, not something read off the points.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return None
    r = float(np.corrcoef(x[mask], y[mask])[0, 1])
    if not np.isfinite(r):
        return None
    if abs(r) < 0.15:
        return (
            f"“{name}” has a mixed effect: it doesn’t move the prediction "
            "in one consistent direction."
        )
    direction = "higher" if r > 0 else "lower"
    return f"Higher “{name}” pushes the prediction {direction}."


def _jitter_width(x, jitter, n_samples):
    """Horizontal spread for low-cardinality features, in x-axis units."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 0.0
    n_unique = np.unique(finite).size
    if jitter is None:
        # Only worth it when repeated values actually stack up.
        auto = n_unique <= _LOW_CARDINALITY and n_unique < n_samples / 2
        jitter = 0.35 if auto else 0.0
    if not jitter:
        return 0.0
    span = finite.max() - finite.min()
    spacing = span / (n_unique - 1) if n_unique > 1 and span > 0 else 1.0
    return jitter * spacing


def scatter(
    shap_values,
    *,
    feature=None,
    color=None,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    feature_names=None,
    figsize=None,
    analysis: bool | str = True,
    direction_labels: bool | tuple[str, str] = True,
    jitter: float | None = None,
    transparent: bool = False,
    ax=None,
):
    """Render an editorial-style dependence scatter for one feature.

    Each point is one sample: its raw value for `feature` on the x-axis against
    that feature's SHAP value on the y-axis. Where `beeswarm` shows *which*
    features matter, this shows *how* one of them behaves.

    Parameters
    ----------
    shap_values : shap.Explanation
        The result of calling a shap Explainer on your data, e.g.
        `explainer(X_test)`. Must be a single-output (binary/regression)
        explanation - for multiclass models, slice a class first. Needs `.data`
        (the original feature values), which forms the x-axis.
    feature : str | int | None
        Which feature to plot, by name or column index (negative indexes count
        from the end). Defaults to None, meaning the top driver by mean |SHAP|.
        Passing a pre-sliced 1-D explanation is not supported: a 1-D array is
        ambiguous between one sample and one feature, so name the column here.
    color : str | int | None
        Optional second feature, by name or index, whose value colours the
        points on the grey->red scale - the standard way to expose an
        interaction. Defaults to None (a single house colour). There is no
        "pick the strongest interaction for me" mode on purpose: that would
        mean computing interaction strength, which is outside this package's
        remit of styling values you already have.
    analysis : bool | str
        Editorial takeaway line under the title. True (default) auto-generates
        a one-sentence insight from the feature's direction of effect; pass a
        string to supply your own; pass False to omit it. Note the auto line
        reads a monotonic direction, so a U-shaped effect is reported as mixed.
    direction_labels : bool | tuple[str, str]
        Small labels at the top and bottom of the y-axis telling the reader
        what each side means. True (default) shows the generic "↑ pushes
        prediction higher / ↓ pushes prediction lower". Pass a (top, bottom)
        tuple for domain-specific wording, or False to omit them.
    jitter : float | None
        Horizontal spread, as a fraction of the gap between distinct values,
        for categorical or integer-coded features that would otherwise stack
        into vertical stripes. Defaults to None: applied automatically only
        when the feature has few distinct values. Pass 0 to disable.
    title, subtitle, source : str | None
        Editorial title stack. `subtitle` defaults to None - the analysis line
        and directional labels already describe the axes.
    feature_names : list[str] | None
        Overrides names on the explanation object, if provided.
    figsize : tuple | None
        Only used if `ax` is None. Defaults to None, in which case the height
        is derived from the title block plus a fixed plot area; pass an
        explicit (w, h) to override.
    transparent : bool
        If True, render (and save) with a transparent background instead of
        white. Save to a format with an alpha channel (PNG, SVG, PDF).
        Ignored when drawing onto an existing `ax`.
    ax : matplotlib.axes.Axes | None
        Draw onto an existing axes instead of creating a new figure. Intended
        for a single-axes figure you manage: the call still adjusts figure
        margins and adds figure-level elements (the red tab, title block, and
        colour key), so it is not suitable as one panel of a multi-panel figure.

    Returns
    -------
    (fig, ax) : the created or given figure and axes.

    Notes
    -----
    The editorial theme is applied inside a matplotlib rc context, so your own
    global rcParams survive the call untouched.
    """
    values, data, names = extract_explanation(shap_values, feature_names)
    n_samples = values.shape[0]

    if data is None:
        raise ShapEditorialError(
            "This explanation has no `.data` (the original feature values), "
            "which the scatter plot needs for its x-axis. Pass an Explanation "
            "created from calling the explainer on your data, e.g. "
            "`explainer(X_test)`, not just raw SHAP value arrays."
        )

    if feature is None:
        kept_idx, _ = top_feature_order(values, 1)
        idx = int(kept_idx[0])
    else:
        idx = resolve_feature(names, feature)
    name = names[idx]

    try:
        x = np.asarray(data[:, idx], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ShapEditorialError(
            f"Feature “{name}” has non-numeric values, so it can't go on a "
            "continuous x-axis. Encode it numerically first."
        ) from exc
    y = np.asarray(values[:, idx], dtype=float)

    colour_idx = None if color is None else resolve_feature(names, color)
    if colour_idx is not None:
        try:
            colour_val = normalize_column(data[:, colour_idx])
        except (TypeError, ValueError) as exc:
            raise ShapEditorialError(
                f"Colour feature “{names[colour_idx]}” has non-numeric values, "
                "so it can't drive the colour scale. Encode it numerically first."
            ) from exc

    analysis_text = None
    if analysis:
        analysis_text = (
            analysis if isinstance(analysis, str) else _analysis_line(x, y, name)
        )
    dir_labels = None
    if direction_labels:
        dir_labels = (
            ("↑ pushes prediction higher", "↓ pushes prediction lower")
            if direction_labels is True
            else tuple(direction_labels)
        )

    top_in = title_block_height(title=title, subtitle=subtitle, analysis=analysis_text)
    bottom_in = 0.95

    with mpl.rc_context():
        set_theme(transparent=transparent)

        if ax is None:
            if figsize is None:
                figsize = (8.0, top_in + _PLOT_HEIGHT_IN + bottom_in)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        width = _jitter_width(x, jitter, n_samples)
        x_plot = x
        if width:
            rng = np.random.default_rng(0)
            x_plot = x + rng.uniform(-width / 2, width / 2, size=n_samples)

        if colour_idx is None:
            ax.scatter(
                x_plot, y, color=C_ECON_RED, s=18, linewidths=0, alpha=0.55, zorder=3
            )
        else:
            ax.scatter(
                x_plot,
                y,
                c=colour_val,
                cmap=FEATURE_CMAP,
                vmin=0,
                vmax=1,
                s=18,
                linewidths=0,
                alpha=0.7,
                zorder=3,
            )

        # Marks the line between lowering and raising the prediction, sitting
        # just under the points rather than slashing across them.
        ax.axhline(0, color=C_ZERO, linewidth=1.0, zorder=0.5)
        ax.set_xlabel(name)
        ax.grid(axis="y", color=C_GRID, linewidth=0.6, zorder=-1)
        ax.set_axisbelow(True)

        # Pad the y range so the directional labels sit in empty bands above and
        # below the data instead of colliding with points.
        finite_y = y[np.isfinite(y)]
        if finite_y.size:
            lo, hi = float(finite_y.min()), float(finite_y.max())
            span = (hi - lo) or 1.0
            pad = span * (0.16 if dir_labels else 0.06)
            ax.set_ylim(lo - pad, hi + pad)

        height = fig.get_figheight()
        fig.subplots_adjust(
            top=1.0 - top_in / height,
            bottom=bottom_in / height,
            left=0.11,
            right=0.95,
        )

        if colour_idx is not None:
            sm = ScalarMappable(cmap=FEATURE_CMAP)
            sm.set_array([])
            cax = fig.add_axes([0.77, 1.0 - 0.55 / height, 0.18, 0.11 / height])
            cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["Low", "High"])
            cbar.ax.tick_params(length=0, labelsize=8.5, colors=C_LABEL_MUTED)
            cbar.outline.set_visible(False)
            cax.set_title(
                names[colour_idx],
                fontsize=8.5,
                color=C_LABEL_MUTED,
                loc="left",
                pad=4,
            )

        if dir_labels:
            top_txt, bottom_txt = dir_labels
            for text, y_frac, va in (
                (top_txt, 0.985, "top"),
                (bottom_txt, 0.015, "bottom"),
            ):
                ax.text(
                    0.012,
                    y_frac,
                    text,
                    transform=ax.transAxes,
                    ha="left",
                    va=va,
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
