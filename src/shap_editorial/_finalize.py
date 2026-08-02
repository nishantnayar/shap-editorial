"""Shared title/subtitle/source-line finishing touches for all chart types."""

from __future__ import annotations

from matplotlib.patches import Rectangle

from ._theme import C_ECON_RED, C_LABEL, C_LABEL_MUTED, C_SOURCE


def finalize(fig, ax, *, title=None, subtitle=None, source=None, analysis=None):
    """Add the Economist-style title block above the axes and a source line below.

    Draws the signature red corner tab, then a bold left-aligned title, a muted
    subtitle, and an optional darker analytical takeaway line beneath it, then
    an optional source line at the bottom left. Positions are figure-relative so
    this behaves consistently regardless of figure size. Call this last, after
    all plotting is done.

    The title block is set flush-left with the whole graphic (Economist
    convention), not indented to the plot area, so it sits above the y-axis
    category labels rather than to the right of them.
    """
    left = 0.025

    # Economist signature: a small red tab in the top-left corner.
    tab_w, tab_h = 0.045, 0.018
    tab_y = 0.965
    fig.add_artist(
        Rectangle(
            (left, tab_y),
            tab_w,
            tab_h,
            transform=fig.transFigure,
            facecolor=C_ECON_RED,
            edgecolor="none",
            clip_on=False,
            zorder=5,
        )
    )

    y = tab_y - 0.028
    if title:
        fig.text(
            left,
            y,
            title,
            fontsize=15,
            fontweight="bold",
            color=C_LABEL,
            ha="left",
            va="top",
        )
        y -= 0.052

    if subtitle:
        fig.text(
            left,
            y,
            subtitle,
            fontsize=10.5,
            color=C_LABEL_MUTED,
            ha="left",
            va="top",
        )

    if source:
        fig.text(
            left,
            0.01,
            source,
            fontsize=8.5,
            color=C_SOURCE,
            ha="left",
            va="bottom",
        )
