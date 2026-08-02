"""Publication-ready global feature-importance bar chart."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ._finalize import ROW_HEIGHT_IN, finalize, highlight_row, title_block_height
from ._theme import (
    C_GRID,
    C_HIGH,
    C_LABEL_MUTED,
    C_LOW,
    C_MID,
    C_OTHER_BAR,
    set_theme,
)
from ._utils import ShapEditorialError, extract_explanation, top_feature_order

# Bars are shaded by importance on the grey→red scale: the most important
# feature is red and pops, lower ones recede to grey — matching the beeswarm's
# "redder = higher" convention instead of a flat wall of red.
_CMAP = LinearSegmentedColormap.from_list("shap_editorial_bar", [C_LOW, C_MID, C_HIGH])


def _analysis_line(importance, names, kept_idx):
    """One-sentence takeaway: the most important feature, and by how much."""
    if len(kept_idx) == 0 or importance[kept_idx[0]] <= 0:
        return None
    top = kept_idx[0]
    msg = f"“{names[top]}” is the most important feature overall"
    if len(kept_idx) > 1 and importance[kept_idx[1]] > 0:
        ratio = importance[top] / importance[kept_idx[1]]
        if ratio >= 1.15:
            msg += f" — about {ratio:.1f}× the next feature's impact"
    return msg + "."


def _value_label(ax, x, y):
    ax.text(
        x,
        y,
        f"  {x:.3g}",
        va="center",
        ha="left",
        fontsize=8.5,
        color=C_LABEL_MUTED,
        zorder=4,
    )


def bar(
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
    show_values: bool = True,
    axis_label: str | None = "Average impact on the model's output",
    transparent: bool = False,
    ax=None,
):
    """Render an editorial-style global feature-importance bar chart.

    Each bar is a feature's mean absolute SHAP value across all samples — a
    single, direction-free measure of how much the feature moves the model's
    output on average. Bars are sorted with the most important feature at top.

    Parameters
    ----------
    shap_values : shap.Explanation
        The result of calling a shap Explainer on your data. Must be a
        single-output (binary/regression) explanation — for multiclass models,
        slice a class first. Unlike `beeswarm`, `.data` is not required.
    max_display : int
        Number of top features (by mean |SHAP|) to show. Must be at least 1.
    show_other : bool
        If True, collapse the remaining features into a single "N other
        features" bar at the bottom (the summed importance of the rest).
        Defaults to False — just the top `max_display`.
    analysis : bool | str
        Takeaway line under the title. True (default) names the most important
        feature; pass a string to override, or False to omit.
    highlight : bool
        If True (default), highlight the most-important row.
    show_values : bool
        If True (default), print each bar's importance value at its end.
    axis_label : str | None
        Plain-language caption under the x-axis explaining what the bars
        measure. Pass None to omit.
    title, subtitle, source : str | None
        Editorial title stack (subtitle defaults to None).
    feature_names : list[str] | None
        Overrides names on the explanation object.
    figsize : tuple | None
        Defaults to None → height chosen from the number of rows.
    transparent : bool
        If True, render/save with a transparent background instead of white.
        Save to a format with an alpha channel (PNG, SVG, PDF); JPEG has no
        transparency and will flatten the background.
    ax : matplotlib.axes.Axes | None
        Draw onto an existing axes instead of creating a new figure. Intended
        for a single-axes figure you manage: the call still adjusts figure
        margins and adds figure-level elements (the red tab, title block, and
        axis caption) positioned for that layout, so it is not suitable for
        embedding as one panel in a multi-panel figure.

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

    values, _data, names = extract_explanation(shap_values, feature_names)
    importance = np.abs(values).mean(axis=0)

    kept_idx, other_idx = top_feature_order(values, max_display)
    has_other = show_other and len(other_idx) > 0
    n_rows = len(kept_idx) + (1 if has_other else 0)

    analysis_text = None
    if analysis:
        analysis_text = (
            analysis
            if isinstance(analysis, str)
            else _analysis_line(importance, names, kept_idx)
        )

    top_in = title_block_height(title=title, subtitle=subtitle, analysis=analysis_text)
    bottom_in = 0.85 if axis_label else 0.5

    with mpl.rc_context():
        set_theme(transparent=transparent)

        if ax is None:
            if figsize is None:
                figsize = (8.0, top_in + ROW_HEIGHT_IN * n_rows + bottom_in)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        base = 1 if has_other else 0  # bottom row belongs to the aggregate
        row_labels = [""] * n_rows

        # kept_idx is already sorted descending, so the first entry sets both the
        # colour scale and the x-limit.
        vmax = float(importance[kept_idx[0]]) if len(kept_idx) else 0.0
        for i, idx in enumerate(kept_idx[::-1]):  # most important ends up on top
            y = base + i
            frac = importance[idx] / vmax if vmax > 0 else 0.0
            ax.barh(
                y,
                importance[idx],
                height=0.68,
                color=_CMAP(frac),
                linewidth=0,
                zorder=3,
            )
            if show_values:
                _value_label(ax, importance[idx], y)
            row_labels[y] = names[idx]

        xlim_max = vmax
        if has_other:
            other_imp = float(importance[other_idx].sum())
            ax.barh(0, other_imp, height=0.68, color=C_OTHER_BAR, linewidth=0, zorder=3)
            if show_values:
                _value_label(ax, other_imp, 0)
            row_labels[0] = f"{len(other_idx)} other features"
            xlim_max = max(xlim_max, other_imp)

        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels)
        ax.set_ylim(-0.6, n_rows - 0.4)
        ax.set_xlim(0, (xlim_max or 1.0) * 1.18)
        ax.grid(axis="x", color=C_GRID, linewidth=0.6, zorder=-1)
        ax.set_axisbelow(True)

        height = fig.get_figheight()
        fig.subplots_adjust(
            top=1.0 - top_in / height,
            bottom=bottom_in / height,
            left=0.28,
            right=0.95,
        )

        if highlight and len(kept_idx):
            highlight_row(fig, ax, n_rows - 1)

        if axis_label:
            fig.text(
                0.615,  # centre of the plot area, between left=0.28 and right=0.95
                0.42 / height,
                axis_label,
                ha="center",
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
