# Contributing

Thanks for wanting to improve `shap-editorial`. This package is a thin
styling/layout layer on top of `shap.Explanation` objects - not a new
interpretability method, and not a SHAP computation library. Keep changes
inside that remit.

## Getting started

```bash
uv sync --extra dev
uv run pre-commit install
uv run python -m pytest tests/ -q
```

Day-to-day commands, Ruff, and the three-layer testing strategy live in
**[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** and
**[docs/TESTING.md](docs/TESTING.md)**.

## Adding a chart type

New chart types should follow the `beeswarm` pattern:

1. Take a `shap.Explanation`-shaped object.
2. Extract arrays via `_utils.py`.
3. Render with the shared theme (`_theme.py`).
4. Finish with `_finalize.finalize()` so every chart looks consistent.

Public surface is the chart function itself (`beeswarm`, `waterfall`, `bar`,
`scatter`). Everything else stays private (leading underscore).

## Scope to avoid

- Real-time / production model monitoring
- Drift or bias detection
- New SHAP computation methods or approximators
- Regulatory / compliance-specific outputs

## Pull requests

- Keep the unit suite green (`pytest tests/`).
- After theming, layout, or figure-lifecycle changes, also run the example
  scripts and (when relevant) the visual-diff procedure in
  [docs/TESTING.md](docs/TESTING.md).
- Match the existing style: concise Python, NumPy-style docstrings on public
  chart functions only.
