"""Publication-ready waterfall plot for a single-prediction shap.Explanation."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ._finalize import ROW_HEIGHT_IN, finalize, highlight_row, title_block_height
from ._theme import (
    C_GRID,
    C_LABEL,
    C_LABEL_MUTED,
    C_NEG,
    C_POS,
    C_ZERO,
    set_theme,
)
from ._utils import ShapEditorialError, extract_single_explanation


def _fmt(v):
    """Compact, human-readable formatting of a feature value or number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return str(v)
    if f.is_integer() and abs(f) < 1e6:
        return str(int(f))
    return f"{f:.3g}"


def _analysis_line(values, names, kept_idx):
    """One-sentence takeaway: the feature that moved this prediction the most."""
    if len(kept_idx) == 0:
        return None
    top = kept_idx[0]
    c = float(values[top])
    if abs(c) < 1e-12:
        return None
    direction = "up" if c > 0 else "down"
    return (
        f"For this prediction, “{names[top]}” had the largest effect "
        f"({c:+.3g}), pushing the output {direction}."
    )


def waterfall(
    shap_values,
    *,
    max_display: int = 10,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    feature_names=None,
    figsize=None,
    show_other: bool = True,
    analysis: bool | str = True,
    highlight: bool = True,
    show_values: bool = False,
    transparent: bool = False,
    ax=None,
):
    """Render an editorial-style waterfall plot for a single prediction.

    A waterfall shows how each feature moves the model output from the average
    prediction to this instance's prediction: red bars push the output up, grey
    bars push it down, and they sum (with the baseline) to this prediction.

    Parameters
    ----------
    shap_values : shap.Explanation
        A *single-instance* explanation, e.g. `explainer(X)[0]`. Must expose
        `.values` (1-D), `.base_values` (E[f(x)]), and ideally `.data`.
        Multiclass or multi-sample inputs raise `ShapEditorialError`.
    max_display : int
        Number of top features (by |SHAP|) to show individually. At least 1.
    show_other : bool
        If True (default), collapse the remaining features into a single
        "N other features" bar so the bars reconcile from the average
        prediction to this prediction - the whole point of a waterfall. Set
        False to show only the top `max_display` features; the bars then stop
        short of "This prediction" (the gap is the hidden contributions).
        Defaults to True here, unlike `beeswarm`, precisely to keep the
        additive reconciliation intact.
    title, subtitle, source : str | None
        Editorial title stack (subtitle defaults to None).
    analysis : bool | str
        Takeaway line under the title. True (default) auto-generates it from the
        largest contribution; pass a string to override, or False to omit.
    highlight : bool
        If True (default), highlight the largest-contribution row.
    show_values : bool
        If True, append this instance's feature value to each row label
        ("name = value"). Defaults to False - a raw, unitless feature value
        next to the contribution tends to confuse more than it helps.
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
        margins and adds figure-level elements (the red tab and title block)
        positioned for that layout, so it is not suitable for embedding as one
        panel in a multi-panel figure.

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

    values, data, base, names = extract_single_explanation(shap_values, feature_names)
    fx = base + float(values.sum())

    order = np.argsort(np.abs(values))[::-1]  # largest impact first
    kept_idx = order[:max_display]
    other_idx = order[max_display:]
    has_other = show_other and len(other_idx) > 0
    n_rows = len(kept_idx) + (1 if has_other else 0)

    analysis_text = None
    if analysis:
        analysis_text = (
            analysis
            if isinstance(analysis, str)
            else _analysis_line(values, names, kept_idx)
        )

    top_in = title_block_height(title=title, subtitle=subtitle, analysis=analysis_text)
    bottom_in = 0.55  # x ticks only (endpoint labels sit at the top)

    with mpl.rc_context():
        set_theme(transparent=transparent)

        if ax is None:
            if figsize is None:
                figsize = (8.0, top_in + ROW_HEIGHT_IN * n_rows + bottom_in)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Rows from the bottom up: the aggregate "other" bar first (smallest
        # contributions), then features in ascending impact so the largest sits
        # at the top and its bar lands on f(x).
        rows = []
        if has_other:
            rows.append((values[other_idx].sum(), f"{len(other_idx)} other features"))
        for idx in kept_idx[::-1]:
            if show_values and data is not None:
                label = f"{names[idx]} = {_fmt(data[idx])}"
            else:
                label = names[idx]
            rows.append((float(values[idx]), label))

        running = base
        starts, ends = [], []
        for contrib, _ in rows:
            starts.append(running)
            running += contrib
            ends.append(running)

        row_labels = []
        for y, (contrib, label) in enumerate(rows):
            left = min(starts[y], ends[y])
            color = C_POS if contrib >= 0 else C_NEG
            ax.barh(y, abs(contrib), left=left, height=0.62, color=color, zorder=3)
            # Value label just past the outer end of the bar.
            ax.text(
                ends[y],
                y,
                f"  {contrib:+.3g}" if contrib >= 0 else f"{contrib:+.3g}  ",
                va="center",
                ha="left" if contrib >= 0 else "right",
                fontsize=8.5,
                color=C_LABEL_MUTED,
                zorder=4,
            )
            row_labels.append(label)

        # Thin connectors linking each bar's cumulative boundary to the next row.
        for i in range(n_rows - 1):
            ax.plot(
                [ends[i], ends[i]],
                [i + 0.31, i + 1 - 0.31],
                color=C_ZERO,
                linewidth=0.8,
                zorder=2,
            )

        ax.axvline(base, color=C_ZERO, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.axvline(fx, color=C_LABEL, linewidth=1.2, zorder=1)

        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels)
        ax.set_ylim(-0.7, n_rows - 0.3)
        ax.grid(axis="x", color=C_GRID, linewidth=0.6, zorder=-1)
        ax.set_axisbelow(True)

        # Comfortable x-limits with room for the value labels at the bar ends.
        span_lo = min([base, fx, *starts, *ends])
        span_hi = max([base, fx, *starts, *ends])
        span = span_hi - span_lo
        pad = 0.18 * (span or 1.0)
        ax.set_xlim(span_lo - pad, span_hi + pad)

        # Plain-language endpoint labels above their reference lines (at the top,
        # so they never collide with the x-axis ticks). No "E[f(x)]" jargon.
        # When the two endpoints nearly coincide the labels would overprint, and
        # the baseline is redundant anyway since both read the same number.
        if span > 0 and abs(fx - base) / span > 0.08:
            ax.annotate(
                f"Average prediction: {base:.3g}",
                xy=(base, n_rows - 0.34),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=C_LABEL_MUTED,
                annotation_clip=False,
            )
        ax.annotate(
            f"This prediction: {fx:.3g}",
            xy=(fx, n_rows - 0.34),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=C_LABEL,
            annotation_clip=False,
        )

        height = fig.get_figheight()
        fig.subplots_adjust(
            top=1.0 - top_in / height,
            bottom=bottom_in / height,
            left=0.28,
            right=0.95,
        )

        if highlight and len(kept_idx):
            highlight_row(fig, ax, n_rows - 1)

        finalize(
            fig,
            ax,
            title=title,
            subtitle=subtitle,
            source=source,
            analysis=analysis_text,
        )

    return fig, ax
