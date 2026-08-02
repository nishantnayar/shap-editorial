"""Shared title/subtitle/source-line finishing touches for all chart types."""

from __future__ import annotations

from matplotlib.patches import Rectangle

from ._theme import C_ECON_RED, C_LABEL, C_LABEL_MUTED, C_SOURCE

# Title-block vertical layout, in inches from the top of the figure. Inch-based
# (not figure-fraction) so text keeps a constant physical size and never
# overlaps, whatever the figure's height. `_beeswarm` reserves matching top
# margin via `title_block_height()` so the block never collides with the plot.
_TAB_TOP_IN = 0.14  # gap above the red tab
_TAB_H_IN = 0.10  # tab height
_GAP_AFTER_TAB_IN = 0.16
_TITLE_STEP_IN = 0.34  # advance after the (14pt bold) title
_LINE_STEP_IN = 0.28  # advance after a subtitle / analysis line
_SOURCE_UP_IN = 0.12  # source line, measured up from the bottom edge
_TAB_W_FRAC = 0.045  # tab width (horizontal, so figure-fraction is fine)
_LEFT = 0.025  # flush-left with the whole graphic


def title_block_height(*, title=None, subtitle=None, analysis=None) -> float:
    """Inches of vertical space the title block needs for the given content.

    `_beeswarm` uses this to reserve top margin so the plot starts just below
    the block. Kept in sync with `finalize`'s own cursor arithmetic.
    """
    h = _TAB_TOP_IN + _TAB_H_IN + _GAP_AFTER_TAB_IN
    if title:
        h += _TITLE_STEP_IN
    if subtitle:
        h += _LINE_STEP_IN
    if analysis:
        h += _LINE_STEP_IN
    return h + 0.12  # small gap between the block and the plot


def finalize(fig, ax, *, title=None, subtitle=None, source=None, analysis=None):
    """Add the Economist-style title block above the axes and a source line below.

    Draws the signature red corner tab, then a bold left-aligned title, a muted
    subtitle, and an optional darker analytical takeaway line beneath it, then
    an optional source line at the bottom left. Vertical positions are measured
    in inches from the figure edges, so the block keeps a constant physical size
    and spacing regardless of figure height. Call this last, after all plotting.

    The title block is set flush-left with the whole graphic (Economist
    convention), not indented to the plot area, so it sits above the y-axis
    category labels rather than to the right of them.
    """
    h = fig.get_figheight()

    def y_from_top(inches):
        return 1.0 - inches / h

    # Economist signature: a small red tab in the top-left corner.
    fig.add_artist(
        Rectangle(
            (_LEFT, y_from_top(_TAB_TOP_IN + _TAB_H_IN)),
            _TAB_W_FRAC,
            _TAB_H_IN / h,
            transform=fig.transFigure,
            facecolor=C_ECON_RED,
            edgecolor="none",
            clip_on=False,
            zorder=5,
        )
    )

    cur = _TAB_TOP_IN + _TAB_H_IN + _GAP_AFTER_TAB_IN
    if title:
        fig.text(
            _LEFT,
            y_from_top(cur),
            title,
            fontsize=15,
            fontweight="bold",
            color=C_LABEL,
            ha="left",
            va="top",
        )
        cur += _TITLE_STEP_IN

    if subtitle:
        fig.text(
            _LEFT,
            y_from_top(cur),
            subtitle,
            fontsize=10.5,
            color=C_LABEL_MUTED,
            ha="left",
            va="top",
        )
        cur += _LINE_STEP_IN

    if analysis:
        # The analytical takeaway sits below the subtitle, in a darker weight
        # than the subtitle so it reads as the insight, not metadata.
        fig.text(
            _LEFT,
            y_from_top(cur),
            analysis,
            fontsize=10.5,
            color=C_LABEL,
            ha="left",
            va="top",
        )
        cur += _LINE_STEP_IN

    if source:
        fig.text(
            _LEFT,
            _SOURCE_UP_IN / h,
            source,
            fontsize=8.5,
            color=C_SOURCE,
            ha="left",
            va="bottom",
        )
