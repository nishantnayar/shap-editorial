<h1 align="center">shap-editorial</h1>

<p align="center">
  <strong>Turn model explanations into charts you can actually publish.</strong><br>
  One line of code turns SHAP's output into a clean, self-explaining chart for a report, a slide, or a paper. No design work, no manual rework.
</p>

<p align="center">
  <a href="https://github.com/nishantnayar/shap-editorial/actions/workflows/ci.yml">
    <img src="https://github.com/nishantnayar/shap-editorial/actions/workflows/ci.yml/badge.svg" alt="CI status">
  </a>
  <a href="https://pypi.org/project/shap-editorial/">
    <img src="https://img.shields.io/pypi/v/shap-editorial" alt="PyPI version">
  </a>
  <img src="https://img.shields.io/badge/python-3.10--3.13-blue" alt="Python 3.10 to 3.13">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="Alpha">
</p>

<p align="center">
  <a href="https://github.com/nishantnayar/shap-editorial/blob/main/docs/API.md">API</a> ·
  <a href="https://github.com/nishantnayar/shap-editorial/tree/main/examples">Examples</a> ·
  <a href="https://github.com/nishantnayar/shap-editorial/blob/main/docs/DEVELOPMENT.md">Development</a> ·
  <a href="https://github.com/nishantnayar/shap-editorial/blob/main/docs/TESTING.md">Testing</a> ·
  <a href="https://github.com/nishantnayar/shap-editorial/blob/main/CONTRIBUTING.md">Contributing</a>
</p>

---

<p align="center">
  <img src="https://raw.githubusercontent.com/nishantnayar/shap-editorial/main/examples/images/beeswarm/hero.png" alt="Editorial beeswarm plot of SHAP values" width="720">
</p>

<p align="center"><em>One <code>beeswarm()</code> call - <em>The Economist</em>-style title block, an auto-generated takeaway line, a highlighted top driver, directional axis cues, and an explainable grey→red colour scale, ready to publish.</em></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/nishantnayar/shap-editorial/main/examples/images/waterfall/hero.png" alt="Editorial waterfall plot of a single SHAP prediction" width="720">
</p>

<p align="center"><em>…and one <code>waterfall()</code> call to explain a single prediction - same red tab and grey→red palette, plain-language endpoints (<em>Average prediction → This prediction</em>, no <code>E[f(x)]</code>/<code>f(x)</code> jargon), and an auto takeaway.</em></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/nishantnayar/shap-editorial/main/examples/images/bar/hero.png" alt="Editorial bar chart of global SHAP feature importance" width="720">
</p>

<p align="center"><em>…and one <code>bar()</code> call for a clean importance ranking. Every chart type shares the same look.</em></p>

## What it is

Machine-learning models predict well but explain poorly: they tell you *what*
they decided, not *why*. [SHAP](https://github.com/shap/shap) is the widely used
tool that closes that gap, measuring how much each input feature pushed a
prediction. `shap-editorial` takes that analysis and turns it into a clean,
self-explaining chart in **one line of code**.

Out of the box, SHAP's own charts look like raw research output: cramped labels,
mathematical axis titles, and no headline. This package restyles them for a
human audience, so the result is ready for a report, a slide deck, a blog post,
or a paper without a trip through a design tool.

- **Already use SHAP?** It is a drop-in styling layer: your existing values go
  in, a polished chart comes out. It does not compute SHAP values or add new
  interpretability methods; it only makes the output look good.
- **New to SHAP?** The charts are built to be read at a glance: a plain-language
  title, a one-sentence takeaway, and clear cues for which way each feature
  pushes the prediction, instead of jargon.

### What each chart answers

| Chart | The question it answers |
|---|---|
| **beeswarm** | Across all your data, which features matter most, and which way do they push the prediction? |
| **waterfall** | For one specific case, why did the model land on this prediction? |
| **bar** | A simple ranking: which features matter most overall? |
| **scatter** | For one feature, how does its value relate to its effect on the prediction? |

### See the difference

The same SHAP output, drawn by SHAP's default plot (left) and by
`shap-editorial` (right):

<p align="center">
  <img src="https://raw.githubusercontent.com/nishantnayar/shap-editorial/main/examples/images/beeswarm/comparison.png" alt="Before and after: stock SHAP plot versus shap-editorial" width="900">
</p>

<details>
<summary><strong>Feature-by-feature comparison with SHAP's built-in plots</strong></summary>

|                        | `shap.summary_plot()`         | `shap_editorial.beeswarm()`               |
| ---------------------- | ----------------------------- | ----------------------------------------- |
| Typography             | matplotlib defaults           | Economist-style font stack, sized hierarchy |
| Title / subtitle       | none / jargon axis label      | Red corner tab, bold flush-left title, plain-language subtitle |
| Analysis               | none                          | Auto takeaway line + highlighted top-driver row |
| Direction cues         | reader must infer             | "← pushes prediction lower / higher →" under the axis |
| Colour scale           | SHAP red/blue                 | Explainable grey → red ("redder = higher"), high values pop |
| Colour key             | vertical bar, rotated label   | Horizontal key, horizontal labels           |
| Source / attribution   | none                          | Optional source line                      |
| Cross-chart consistency| varies by plot type           | Shared theme + finalize layer             |

</details>

## Install

```bash
pip install shap-editorial
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add shap-editorial
```

**Runtime dependencies are only `matplotlib` and `numpy`.** `shap` itself is
*not* required to use this package - input is duck-typed (see below).

For local development from a clone:

```bash
uv sync --extra dev              # core + test tooling
uv sync --extra example          # adds shap + scikit-learn for the demos
# or: pip install -e ".[dev,example]"
```

## Quickstart

```python
import shap
import shap_editorial as se

# 1. Compute SHAP values however you normally would
explainer = shap.TreeExplainer(model)
explanation = explainer(X_test)  # a shap.Explanation object

# 2. Render it, publication-ready - an auto takeaway line, a highlighted
#    top driver, and directional axis labels come for free.
fig, ax = se.beeswarm(
    explanation,
    title="What drives the churn prediction",
    source="Source: internal model v3 · n = 2,000 held-out customers",
    max_display=10,
)

# 3. Save at whatever DPI you need
fig.savefig("beeswarm.png", dpi=200, bbox_inches="tight")
```

`beeswarm()` returns the `(fig, ax)` pair, so you keep full matplotlib control
for any further tweaks.

> **Tips**
> - **Any format.** Because you get the `fig` back, save wherever matplotlib
>   can: `fig.savefig("chart.png" | ".jpg" | ".svg" | ".pdf", dpi=200,
>   bbox_inches="tight")`. Use **SVG/PDF** for crisp, resolution-independent
>   figures in print/decks; raster (`.png`/`.jpg`) for the web.
> - **Transparent backgrounds:** pass `transparent=True` for slides or dark
>   pages. Save to a format with an alpha channel - **PNG, SVG, or PDF** -
>   since **JPEG has no transparency** and will flatten the background.

### Run the real end-to-end examples

```bash
uv run --extra example python examples/beeswarm_quickstart.py   # the hero chart
uv run --extra example python examples/beeswarm_gallery.py       # many datasets/tasks
uv run --extra example python examples/waterfall_quickstart.py   # single-prediction waterfall
uv run --extra example python examples/bar_quickstart.py         # global importance bar chart
uv run --extra example python examples/scatter_quickstart.py     # one feature's effect in depth
uv run --extra example python examples/beeswarm_comparison.py    # side-by-side before/after vs stock SHAP
```

See [`examples/`](https://github.com/nishantnayar/shap-editorial/tree/main/examples) for the full gallery and how to run it.

## API

| Object | Description |
| --- | --- |
| `beeswarm(...)` | Global feature-impact summary across all samples. Returns `(fig, ax)`. |
| `waterfall(...)` | Single-prediction explanation. Returns `(fig, ax)`. |
| `bar(...)` | Global feature-importance ranking (mean \|SHAP\|). Returns `(fig, ax)`. |
| `scatter(...)` | One feature's value vs its SHAP value, optionally coloured by another. Returns `(fig, ax)`. |
| `set_theme(...)` | Apply the editorial matplotlib theme globally (optional; charts use it via `rc_context`). |
| `ShapEditorialError` | Raised when the input isn't a usable SHAP explanation. |

Signatures, parameters, and chart-specific notes:
**[docs/API.md](https://github.com/nishantnayar/shap-editorial/blob/main/docs/API.md)**.

```python
fig, ax = se.waterfall(expl[..., 0][0], title="Why this case was predicted malignant")
fig, ax = se.bar(explanation, title="Which features matter most")
fig, ax = se.scatter(explanation, feature="worst radius", color="mean texture")
```

## Interpreting direction (read this)

A beeswarm shows the direction of effect for **one class** - whichever one your
`Explanation` holds. `shap-editorial` styles what you give it; it **cannot know
your class coding**, so *you* are responsible for framing direction correctly.

For a binary classifier, `TreeExplainer` returns values of shape
`(n, features, 2)`, and you must slice a class before plotting. Which class you
slice flips the sign of every effect:

```python
# sklearn breast cancer: target is coded 0 = malignant, 1 = benign
expl = explainer(X)  # shape (n, features, 2)

se.beeswarm(
    expl[..., 0],  # explains P(malignant)
    title="What drives the malignancy prediction",
    direction_labels=("← toward benign", "toward malignant →"),
)

se.beeswarm(
    expl[..., 1],  # explains P(benign) - every effect flips!
    title="What drives the benign prediction",
)
```

If a takeaway ever reads "backwards" (e.g. *"higher 'worst concave points'
pushes the prediction lower"* under a **malignancy** title), you've almost
certainly sliced the opposite class. Use `direction_labels=(left, right)` to
state your framing explicitly - it's the clearest guard against this.

## Design principles

- **Duck-typed input, no hard `shap` dependency.** Anything shaped like a
  `shap.Explanation` works. This keeps the install light and makes testing fast.
- **Multiclass is rejected, not guessed.** A 3-D `values` array raises
  `ShapEditorialError` telling you to slice a class first
  (`shap_values[..., class_index]`). Silently picking a class misleads more
  than it helps.
- **Self-contained theme.** Palette and font stack live in one module; no
  dependency on any external charting-style package.
- **Readability over convention.** A grey→red scale ("redder = higher value")
  replaces SHAP's blue/red so the low-value mass recedes and high values pop;
  points are drawn in |impact| order so the ones that matter aren't buried; the
  aggregate "other features" row sits subdued at the bottom.
- **Public API is the chart functions.** Everything else is a private module
  (leading underscore) and not guaranteed stable.

## Roadmap

- [x] `beeswarm()` - global feature-impact summary
- [x] Packaging, CI, LICENSE
- [x] `waterfall()` - single-prediction explanation
- [x] `bar()` - global feature-importance bar chart
- [x] `scatter()` - single-feature dependence plot
- [x] First PyPI release (`v0.1.0`; current: `v0.1.1`)

**Out of scope** (by design): real-time model monitoring, drift/bias
detection, new SHAP computation methods, and regulatory/compliance outputs.

## Development

```bash
uv sync --extra dev
uv run python -m pytest tests/ -q
```

Full setup, Ruff, commit checklist, and the three-layer testing strategy:
**[docs/DEVELOPMENT.md](https://github.com/nishantnayar/shap-editorial/blob/main/docs/DEVELOPMENT.md)** and
**[docs/TESTING.md](https://github.com/nishantnayar/shap-editorial/blob/main/docs/TESTING.md)**.

## Contributing

See **[CONTRIBUTING.md](https://github.com/nishantnayar/shap-editorial/blob/main/CONTRIBUTING.md)**. New chart types should follow the
`beeswarm` pattern: extract via `_utils.py`, render with the shared theme, and
finish with `_finalize.finalize()`.

## License

[MIT](https://github.com/nishantnayar/shap-editorial/blob/main/LICENSE) © 2026 Nishant Nayar
