"""Editorial theme: colours, typography, and layout constants.

The look is modelled on *The Economist*'s data-journalism style — a
self-contained styling layer, no dependency on any other charting-theme
package. It exists to make SHAP output look like something you'd publish,
not something you'd screenshot apologetically.

Signature Economist cues implemented here and in `_finalize.py`:
- the small red brand tab in the top-left corner,
- a bold, left-aligned title with a muted subtitle beneath it,
- the Economist blue/red palette,
- a bottom-left source line.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# -- Palette -----------------------------------------------------------
# The Economist's house colours. The low->high feature-value scale runs
# from Economist blue through a light neutral to Economist red.
C_ECON_RED = "#E3120B"   # the brand red — also the high end of the scale
# Grey -> red sequential feature-value scale: "redder = higher value". Low
# values stay neutral grey and recede; high values pop in Economist red. This
# separates by both hue and lightness, so it stays colour-blind safe (unlike a
# red/green scale) and reuses the brand red for a cohesive look.
C_LOW = "#DBDBDB"        # neutral light grey: low feature value (recedes)
C_MID = "#EA9A8B"        # muted salmon: mid feature value
C_HIGH = "#E3120B"       # Economist red: high feature value (draws the eye)

C_SPINE = "#2B2B2B"
C_ZERO = "#8C8C8C"        # muted grey zero reference — visible but subordinate to the data
C_GRID = "#E8E8E8"        # faint value gridlines
C_ROW_GUIDE = "#F0F0F0"   # fainter still: per-row leader lines
C_LABEL = "#121317"        # near-black, Economist body text
C_LABEL_MUTED = "#5B6770"  # muted slate for subtitles
C_SOURCE = "#8A8A8A"
C_HIGHLIGHT = "#FBEBE8"   # faint warm tint behind the highlighted top-driver row
C_BG = "#FFFFFF"
C_OTHER_BAR = "#AEB6BB"    # colour for the collapsed "N other features" row

# Economist headline/body faces aren't redistributable, so we lead with
# their names and fall back to a clean, widely-available sans.
FONT_STACK = ["Econ Sans Cnd", "Officina Sans", "Helvetica Neue", "Arial", "DejaVu Sans"]


def set_theme(*, transparent: bool = False) -> None:
    """Apply the editorial rcParams globally to matplotlib.

    Parameters
    ----------
    transparent : bool
        If True, the figure and axes backgrounds are transparent and saved
        figures keep that transparency — useful for dropping a chart onto a
        coloured slide or a dark web page. Defaults to a white background.
    """
    bg = "none" if transparent else C_BG
    plt.rcParams.update(
        {
            "figure.facecolor": bg,
            "savefig.facecolor": bg,
            "savefig.transparent": transparent,
            "axes.facecolor": bg,
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
            # Economist dot/bar charts let the category labels stand alone —
            # no y-axis tick dashes.
            "ytick.major.size": 0,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": 10,
        }
    )
