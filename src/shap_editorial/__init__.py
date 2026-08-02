"""shap-editorial: publication-ready charts for SHAP explanations."""

from ._bar import bar
from ._beeswarm import beeswarm
from ._theme import set_theme
from ._utils import ShapEditorialError
from ._waterfall import waterfall

__version__ = "0.1.1"
__all__ = ["beeswarm", "waterfall", "bar", "set_theme", "ShapEditorialError"]
