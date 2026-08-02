"""shap-editorial: publication-ready charts for SHAP explanations."""

from ._beeswarm import beeswarm
from ._theme import set_theme
from ._utils import ShapEditorialError

__version__ = "0.1.0"
__all__ = ["beeswarm", "set_theme", "ShapEditorialError"]
