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
C_ECON_RED = "#E3120B"   # the brand red — used for the corner tab
C_LOW = "#006BA2"        # Economist blue: low feature value
C_HIGH = "#E3120B"       # Economist red: high feature value
C_MID = "#D9D9D9"        # neutral light grey: mid feature value

C_SPINE = "#2B2B2B"
C_GRID = "#D7D7D7"
C_LABEL = "#121317"        # near-black, Economist body text
C_LABEL_MUTED = "#5B6770"  # muted slate for subtitles
C_SOURCE = "#8A8A8A"
C_BG = "#FFFFFF"
C_OTHER_BAR = "#AEB6BB"    # colour for the collapsed "N other features" row

# Economist headline/body faces aren't redistributable, so we lead with
# their names and fall back to a clean, widely-available sans.
FONT_STACK = ["Econ Sans Cnd", "Officina Sans", "Helvetica Neue", "Arial", "DejaVu Sans"]


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
