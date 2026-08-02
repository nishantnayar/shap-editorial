"""Editorial theme: colours, typography, and layout constants.

This is a small, self-contained styling layer — no dependency on any
other charting theme package. It exists to make SHAP output look like
something you'd publish, not something you'd screenshot apologetically.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# -- Palette -----------------------------------------------------------
# A diverging low->high feature-value scale, distinct from SHAP's default
# red/blue so charts don't look like an unstyled shap.summary_plot().
C_LOW = "#3B6E8F"     # muted steel blue: low feature value
C_HIGH = "#C24D3B"    # muted brick red: high feature value
C_MID = "#B7BEC4"     # neutral grey, used for the zero-impact midpoint

C_SPINE = "#4A4A4A"
C_GRID = "#E3E3E3"
C_LABEL = "#2B2B2B"
C_LABEL_MUTED = "#6B6B6B"
C_SOURCE = "#8A8A8A"
C_BG = "#FFFFFF"
C_OTHER_BAR = "#B7BEC4"  # colour for the collapsed "N other features" row

FONT_STACK = ["IBM Plex Sans", "Helvetica Neue", "Arial", "DejaVu Sans"]


def set_theme() -> None:
    """Apply the editorial rcParams globally to matplotlib."""
    plt.rcParams.update(
        {
            "figure.facecolor": C_BG,
            "savefig.facecolor": C_BG,
            "axes.facecolor": C_BG,
            "axes.edgecolor": C_SPINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.labelcolor": C_LABEL,
            "axes.labelsize": 10,
            "text.color": C_LABEL,
            "xtick.color": C_LABEL_MUTED,
            "ytick.color": C_LABEL,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": 10,
        }
    )
