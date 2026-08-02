# Development

Setup, day-to-day commands, and where the deeper docs live.

```bash
uv sync --extra dev
uv run python -m pytest tests/ -q
```

## Testing

What this package ships is **rendered pixels**, but almost everything convenient
to assert on is in-memory state. Those two can disagree, so testing is in three
layers, each catching what the one above it cannot see.

| Layer | Run | Speed | Catches | Blind to |
|---|---|---|---|---|
| 1. Unit | `pytest tests/` | ~5 s | Validation, ordering, aggregation, error types | Anything only visible once rendered |
| 2. Integration | `examples/*.py` | ~90 s | Breakage against *real* `shap.Explanation` objects and models | Whether the result looks right |
| 3. Visual regression | `tools/visual_diff.py` | ~3 min | Fonts, layout, spacing, colour, clipping | Correctness of the numbers |

The unit suite (149 tests) uses a lightweight `FakeExplanation` stand-in rather
than importing the real `shap`, so it stays fast and dependency-light, and runs
headless via matplotlib's `Agg` backend. It cannot, however, see anything that
only appears once a figure is rendered - a chart can pass every assertion while
coming out in the wrong typeface - which is why layers 2 and 3 exist.

See **[TESTING.md](TESTING.md)** for the full strategy: per-layer conventions,
the visual-regression procedure, what a new chart type needs, and the
pre-release checklist.

## Before you commit

```bash
uv run --extra dev python -m pytest tests/ -q
uv run ruff format --check .
uv run ruff check .
```

Before a release, or after touching theming, layout, or figure lifecycle, add
layers 2 and 3 as described in [TESTING.md](TESTING.md).

## Code style

Formatting and linting use [Ruff](https://docs.astral.sh/ruff/) (config in
`pyproject.toml`), enforced via a pre-commit hook. Set it up once:

```bash
uv run pre-commit install     # run Ruff automatically on every commit
```

Run it manually anytime:

```bash
uv run ruff format .          # format
uv run ruff check --fix .     # lint + import-sort (isort), autofixing what it can
```

## Related docs

- [API reference](API.md)
- [Testing strategy](TESTING.md)
- [Examples gallery](../examples/README.md)
- [Contributing](../CONTRIBUTING.md)
