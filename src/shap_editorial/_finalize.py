"""Shared title/subtitle/source-line finishing touches for all chart types."""

from __future__ import annotations

from ._theme import C_LABEL, C_LABEL_MUTED, C_SOURCE


def finalize(fig, ax, *, title=None, subtitle=None, source=None):
    """Add a left-aligned title stack above the axes and a source line below.

    Positions are figure-relative so this behaves consistently regardless
    of figure size. Call this last, after all plotting is done, since it
    reads the current axes position to place text sensibly.
    """
    pos = ax.get_position()
    left = pos.x0

    y = 0.98
    if title:
        fig.text(
            left,
            y,
            title,
            fontsize=14,
            fontweight="bold",
            color=C_LABEL,
            ha="left",
            va="top",
        )
        y -= 0.055

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
