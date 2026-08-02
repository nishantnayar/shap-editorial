# shap-editorial

Publication-ready charts for SHAP explanations. A thin styling/layout layer
on top of `shap.Explanation` objects — not a new interpretability method, not
a SHAP computation library.

> Your SHAP values, in a chart good enough to publish — one function call, no
> manual rework.

## Install

```bash
uv pip install -e ".[dev]"      # dev (pytest)
uv pip install -e ".[example]"  # to run the real end-to-end example (shap + sklearn)
```

## Usage

```python
import shap_editorial as se

# `explanation` is any shap.Explanation-shaped object (needs .values;
# ideally .data and .feature_names too)
fig = se.beeswarm(explanation, title="Feature impact on churn")
fig.savefig("beeswarm.png", dpi=200, bbox_inches="tight")
```

Set a global look once:

```python
se.set_theme()
```

Multiclass explanations (`values.ndim == 3`) are rejected with a clear error —
slice a class first, e.g. `shap_values[..., class_index]`.

## Public API

- `beeswarm()` — global feature-impact summary
- `set_theme()` — apply the editorial matplotlib theme
- `ShapEditorialError` — raised on unsupported input

## Development

```bash
uv sync --extra dev
python -m pytest tests/ -q
```

## License

MIT
