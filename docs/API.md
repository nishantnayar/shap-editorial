# API reference

Install: `pip install shap-editorial` ·
Package: [`shap-editorial` on PyPI](https://pypi.org/project/shap-editorial/) ·
Source: [GitHub](https://github.com/nishantnayar/shap-editorial)

Public surface: the chart functions, `set_theme()`, and `ShapEditorialError`.
Everything else is a private module (leading underscore) and not guaranteed
stable.

| Object | Description |
| --- | --- |
| `beeswarm(...)` | Global feature-impact summary across all samples. Returns `(fig, ax)`. |
| `waterfall(...)` | Single-prediction explanation: how each feature moves the output from the average prediction to this one. Returns `(fig, ax)`. |
| `bar(...)` | Global feature-importance ranking (mean \|SHAP\| per feature). Returns `(fig, ax)`. |
| `scatter(...)` | Dependence plot for one feature: its value vs its SHAP value, optionally coloured by a second feature. Returns `(fig, ax)`. |
| `set_theme(transparent=False)` | Apply the Economist-style matplotlib `rcParams` globally. Chart functions apply the theme internally via `rc_context()` without touching your global settings; call `set_theme()` only if you want the style to persist across your own plots. Pass `transparent=True` for a no-background theme. |
| `ShapEditorialError` | Raised when the input isn't a usable SHAP explanation. Subclass of `ValueError`. |

For framing which class you sliced and how direction labels should read, see
[Interpreting direction](../README.md#interpreting-direction-read-this) in the
README.

---

## `beeswarm`

```python
beeswarm(
    shap_values,
    *,
    max_display: int = 10,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    feature_names=None,
    figsize=None,
    show_other: bool = False,
    analysis: bool | str = True,
    highlight: bool = True,
    direction_labels: bool | tuple[str, str] = True,
    transparent: bool = False,
    ax=None,
)
```

- **`shap_values`** - a `shap.Explanation` (or any object exposing `.values`,
  `.data`, and optionally `.feature_names`). Must be single-output
  (binary classification or regression).
- **`max_display`** - number of top features (by mean |SHAP|) to show.
- **`show_other`** - when `True`, collapse the remaining features into a single
  "N other features" row at the bottom (per-sample sum, preserving the additive
  property). Defaults to `False` - just the top `max_display`. That aggregate
  row can't carry the colour scale (it sums across features), so it renders flat
  grey; it's opt-in for completeness.
- **`analysis`** - the editorial takeaway line under the title. `True` (default)
  auto-generates a one-sentence insight from the top driver's SHAP pattern
  (e.g. *"'worst concave points' is the strongest driver: higher values push the
  prediction lower"*); pass a string for your own, or `False` to omit. The
  auto text only narrates the pattern already in the plot - no new computation.
- **`highlight`** - when `True` (default), draw a dotted outline around the
  top-driver row and bold its label so the eye lands on the strongest feature.
- **`direction_labels`** - small labels under the x-axis for "which side means
  what". `True` (default) shows the generic *"← pushes prediction lower /
  pushes prediction higher →"*; pass a `(left, right)` tuple for domain-specific
  wording (e.g. `("← toward benign", "toward malignant →")`), or `False` to omit.
- **`title` / `subtitle` / `source`** - the editorial text stack. Pass `None`
  to omit any of them.
- **`feature_names`** - override names on the explanation object.
- **`transparent`** - render and save with a transparent background instead of
  white, for coloured slides or dark web pages. The saved PNG keeps the
  transparency:

  ```python
  fig, ax = se.beeswarm(explanation, title="…", transparent=True)
  fig.savefig("beeswarm.png", dpi=200, bbox_inches="tight")  # transparent
  ```
- **`ax`** - draw onto an existing axes instead of creating a new figure.

---

## `waterfall` - explain one prediction

Where `beeswarm` summarises the whole dataset, `waterfall` explains a **single
prediction**: red bars push the output up, grey bars push it down, and they sum
(with the baseline) from the average prediction to this one - the endpoints are
labelled in plain language, not `E[f(x)]`/`f(x)`.

```python
expl = explainer(X)  # a shap.Explanation
fig, ax = se.waterfall(
    expl[..., 0][0],  # class 0, instance 0 → 1-D values + base value
    title="Why this case was predicted malignant",
)
```

```python
waterfall(
    shap_values,
    *,
    max_display: int = 10,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    feature_names=None,
    figsize=None,
    show_other: bool = True,
    analysis: bool | str = True,
    highlight: bool = True,
    show_values: bool = False,
    transparent: bool = False,
    ax=None,
)
```

- **`shap_values`** - a **single-instance** explanation (e.g. `explainer(X)[0]`)
  exposing `.values` (1-D), `.base_values` (the average prediction), and ideally
  `.data`. Multiclass, multi-sample, or missing-`base_values` inputs raise
  `ShapEditorialError` - it never guesses.
- **`max_display`** - top features shown individually.
- **`show_other`** - collapse the rest into a single "N other features" bar so
  the bars reconcile from the average prediction to this one. Defaults to `True`
  here (unlike `beeswarm`'s `False`), because that reconciliation *is* the
  waterfall - set `False` to show only the top features (the bars then stop
  short of "This prediction", the gap being the hidden contributions).
- **`show_values`** - append this instance's feature value to each label
  (`name = value`). Off by default (a raw, unitless value next to the
  contribution tends to confuse).
- **`analysis` / `highlight` / `transparent` / `title` / `subtitle` / `source`** -
  behave as in `beeswarm`. The auto takeaway names the largest contribution and
  its direction for this instance.

---

## `bar` - global importance ranking

Each bar is a feature's **mean absolute SHAP value** across all samples, a
single direction-free measure of importance. Use it when you want a clean
ranking rather than the beeswarm's full distribution. Unlike `beeswarm`, `bar`
does not need `.data`.

```python
fig, ax = se.bar(explanation, title="Which features matter most")
```

```python
bar(
    shap_values,
    *,
    max_display: int = 10,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    feature_names=None,
    figsize=None,
    show_other: bool = False,
    analysis: bool | str = True,
    highlight: bool = True,
    show_values: bool = True,
    axis_label: str | None = "Average impact on the model's output",
    transparent: bool = False,
    ax=None,
)
```

- **`shap_values`** - a single-output explanation (slice a class for
  multiclass). `.data` is optional here.
- **`show_values`** - print each bar's importance value at its end (default on).
- **`axis_label`** - plain-language caption under the x-axis; pass `None` to omit.
- **`max_display` / `show_other` / `analysis` / `highlight` / `transparent` /
  `title` / `source`** - behave as in `beeswarm`.

---

## `scatter` - one feature in depth

Where `beeswarm` shows *which* features matter, `scatter` shows *how* one of
them behaves: each point is a sample, plotting the feature's value against its
SHAP value. Colour it by a second feature to expose an interaction.

```python
# defaults to the top feature; name one with feature=..., colour by another
fig, ax = se.scatter(explanation, feature="worst radius", color="mean texture")
```

```python
scatter(
    shap_values,
    *,
    feature=None,
    color=None,
    title: str | None = None,
    subtitle: str | None = None,
    source: str | None = None,
    feature_names=None,
    figsize=None,
    analysis: bool | str = True,
    direction_labels: bool | tuple[str, str] = True,
    jitter: float | None = None,
    transparent: bool = False,
    ax=None,
)
```

- **`feature`** - which feature to plot, by name or column index. Defaults to
  the top driver by mean |SHAP|. Passing a pre-sliced 1-D explanation is not
  supported: name or index the column here instead.
- **`color`** - optional second feature whose value colours the points, the
  standard way to reveal an interaction. There is no auto-pick of the strongest
  interaction feature.
- **`jitter`** - horizontal spread for categorical or integer-coded features
  that would otherwise stack into vertical stripes; applied automatically for
  low-cardinality features, pass `0` to disable.
- **`analysis` / `direction_labels` / `transparent` / `title` / `source`** -
  behave as in `beeswarm` (the direction labels sit on the y-axis here).
