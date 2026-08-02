# shap-editorial

Publication-ready charts for SHAP explanations. A thin styling/layout
layer on top of `shap.Explanation` objects — not a new interpretability
method, not a SHAP computation library.

## Why this exists (read before changing scope)

SHAP's own plotting functions (`summary_plot`, `beeswarm`, `waterfall`,
`force_plot`) are mathematically correct but visually "default
matplotlib, made by a researcher, for a researcher": cramped labels,
jargon axis titles (`E[f(x)]`, `phi`), inconsistent style across plot
types. There is no existing PyPI package that takes SHAP's actual output
objects and renders them as something publishable without manual rework
in another tool. Confirmed via search: `shapiq` is about computation
(approximators, interaction indices), `shapper` is just an R port of the
same plots, and every tutorial uses `shap.summary_plot()` unmodified.

**Target user:** ML practitioners who already use SHAP and need the
output somewhere public-facing — a blog post, a paper, a deck, a
portfolio project. Not a compliance/regulatory tool. Not a replacement
for enterprise MRM platforms (Fiddler, Arthur, Truera) — those solve
production monitoring at a different scale and price point; this solves
"make this one chart look good," which is a much smaller and more
personal problem.

**Selling proposition:** "Your SHAP values, in a chart good enough to
publish — one function call, no manual rework."

## Scope discipline

- This package **wraps and styles** SHAP output. It does not compute
  SHAP values, does not add new interpretability methods, and does not
  do production monitoring, drift detection, or bias analysis. Resist
  scope creep in those directions.
- New chart types should follow the same pattern as `beeswarm`: take a
  `shap.Explanation`-shaped object, extract arrays via `_utils.py`,
  render with the shared theme, finish with `_finalize.finalize()`.
- Chart functions are the public API surface (`beeswarm`, and future
  `waterfall`, `bar`, etc.) — everything else is a private module
  (leading underscore) and not guaranteed stable.

## Architecture

```
src/shap_editorial/
  __init__.py     Public API: beeswarm, set_theme, ShapEditorialError
  _theme.py       Colour palette, font stack, set_theme() rcParams
  _utils.py       extract_explanation(), top_feature_order() — duck-typed
                  extraction from shap.Explanation-like objects
  _finalize.py    Shared title/subtitle/source-line layout, used by every
                  chart type so they look consistent with each other
  _beeswarm.py    The beeswarm chart itself
```

Design choices worth knowing:

- **Duck-typed input, no hard `shap` dependency at import time.**
  `extract_explanation()` only requires `.values` (and, ideally, `.data`
  and `.feature_names`). This means tests don't need the real `shap`
  package (which is heavy — pulls in numba, cloudpickle, sklearn-style
  deps). Tests use `tests/_helpers.py::FakeExplanation`, a minimal stand-in.
- **Multiclass explanations are rejected, not guessed.** `values.ndim == 3`
  raises `ShapEditorialError` with an explicit message telling the caller
  to slice a class first (e.g. `shap_values[..., class_index]`). Do not
  add "helpful" auto-slicing — silently picking a class is more likely to
  mislead than help.
- **Top-N by default; the "other features" row is opt-in (`show_other=True`).**
  By default the beeswarm shows only the top `max_display` features — a
  summed-SHAP-across-many-features quantity isn't something most readers
  can interpret, and the aggregate row can't carry the feature-value
  colour scale (so it renders flat grey, inconsistent with the rest).
  When enabled, it renders at the **bottom** in subdued grey — a residual,
  never above the named features.
- **When shown, the "other features" row uses `.sum(axis=1)` per sample,
  not mean.** This preserves the additive property (per-sample total still
  adds up to something meaningful) — an earlier draft used
  `mean(|values|) * sign(mean(values))`, which is not a meaningful
  quantity and was a real bug caught during initial testing. Don't
  reintroduce a mean-based aggregation here.
- **Points are drawn in ascending |SHAP| order** so the highest-impact
  points render on top instead of being buried under the dense low-impact
  cluster near zero. Opacity is low (~0.6) so that central cluster reads
  as a tone, not a solid mass.
- **Colour scale is per-feature min-max normalized** (`_norm()` inside
  `_beeswarm.py`), matching SHAP's own convention, but is a **grey → red
  sequential** scale ("redder = higher feature value"): low values stay
  neutral grey and recede, high values pop in Economist red. This was a
  deliberate move away from SHAP's blue/red (blue overwhelmed the plot)
  and away from a red/green scale (colour-blind unsafe); grey→red
  separates by both hue and lightness. Don't reintroduce blue here.
- **Theme is modelled on *The Economist*'s data-journalism style** and is
  self-contained — no dependency on any other chart-styling package.
  Palette, font stack, and the signature red corner tab live in
  `_theme.py` / `_finalize.py` only. Every chart type gets the tab and
  title block for free via `finalize()`.

## Testing conventions

- `matplotlib.use("Agg")` at the top of any test file that plots —
  headless, no display available in CI or this sandbox.
- Use `tests/_helpers.py::FakeExplanation` instead of importing `shap`
  in unit tests. Reserve real `shap` + a real trained model for the
  standalone example scripts (see `examples/beeswarm_quickstart.py`), not
  for the pytest suite — keeps `pip install -e .[dev]` fast.
- Run tests from the project root: `python3 -m pytest tests/ -q`
- Current status: 16 tests passing (`test_utils.py`, `test_beeswarm.py`).

## Running the real end-to-end examples

```
pip install shap scikit-learn --break-system-packages
python3 examples/beeswarm_quickstart.py   # the hero chart
python3 examples/beeswarm_gallery.py       # many datasets / tasks
```

`beeswarm_quickstart.py` trains a `RandomForestClassifier` on `sklearn`'s
breast cancer dataset, computes real SHAP values via `shap.TreeExplainer`,
and saves `examples/images/beeswarm/hero.png`. `beeswarm_gallery.py` sweeps
several datasets/tasks into `examples/images/beeswarm/`. Use these to
sanity-check any change visually — the unit tests check
structure/correctness, not appearance.

Examples are namespaced by chart type: scripts are prefixed with the chart
(`beeswarm_*.py`) and outputs go to `images/<chart>/`, so new chart types
(`waterfall`, `bar`) slot in without reorganising.

## Naming

- PyPI/import name `shap-editorial` / `shap_editorial` — confirmed
  available on PyPI (checked directly, not from memory).
- Public function names should read as verbs/nouns a SHAP user already
  expects: `beeswarm()`, not `render_beeswarm_chart()` or similar.

## Not yet built (roadmap, in likely order)

1. `waterfall()` — single-prediction explanation (second most requested
   after beeswarm; natural next chart type, same architecture)
2. `bar()` — simple global feature importance bar chart
3. Packaging: `pyproject.toml`, `LICENSE`, `README.md` for actual PyPI
   publish (not yet created — currently a source tree only)
4. CI (GitHub Actions): run pytest on push; do not add release
   automation until there's an actual v0.1 published manually first

## Explicitly out of scope (don't build these here)

- Real-time/production model monitoring
- Drift or bias detection
- New SHAP computation methods or approximators
- Regulatory/compliance-specific outputs (reason codes, adverse-action
  letters) — that was a considered and *rejected* direction; this
  package is deliberately positioned as a general publishing tool, not
  a compliance tool
