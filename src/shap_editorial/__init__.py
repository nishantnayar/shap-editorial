"""shap-editorial: publication-ready charts for SHAP explanations."""

from ._beeswarm import beeswarm
from ._theme import set_theme
from ._utils import ShapEditorialError
from ._waterfall import waterfall

__version__ = "0.1.0"
__all__ = ["beeswarm", "waterfall", "set_theme", "ShapEditorialError"]
