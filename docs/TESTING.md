# Testing strategy

This package turns `shap.Explanation` objects into charts. That makes it
unusual to test: the thing we actually ship is **rendered pixels**, but almost
everything convenient to assert on is in-memory object state. Those two can
disagree, and have.

The strategy is therefore three layers, each catching a class of bug the layer
above it cannot see.

| Layer | What it runs | Speed | Catches | Blind to |
|---|---|---|---|---|
| 1. Unit | `pytest tests/` | ~5 s | Input validation, ordering, aggregation, which artists exist, error types | Anything only visible once rendered |
| 2. Integration | `examples/*.py` | ~90 s | Breakage against *real* `shap.Explanation` objects, real models, real dtypes | Whether the result looks right |
| 3. Visual regression | `tools/visual_diff.py` | ~3 min | Fonts, layout, spacing, colour, clipping, anything that reaches the PNG | Semantic correctness of the numbers |

**All three are required before a release.** Layer 1 alone is not sufficient
and there is a concrete precedent for that, documented at the bottom.

---

## Layer 1 - Unit tests

```bash
uv run --extra dev python -m pytest tests/ -q
```

149 tests, no `shap` dependency, no display. Current shape:

| File | Tests | Covers |
|---|---|---|
| `tests/test_utils.py` | 30 | `extract_explanation`, `extract_single_explanation`, `top_feature_order`, `resolve_feature`, `normalize_column` - every validation and error path |
| `tests/test_theme.py` | 18 | Font resolution, rcParams isolation, transparency; parametrised over all four chart types |
| `tests/test_beeswarm.py` | 23 | Chart structure, options, `_analysis_line` |
| `tests/test_waterfall.py` | 30 | Chart structure, options, bar sign colours, endpoint labels, `_fmt` |
| `tests/test_bar.py` | 22 | Chart structure, options, `_analysis_line` |
| `tests/test_scatter.py` | 26 | Feature resolution, colour key, jitter, `_analysis_line` |

### Conventions

- **`tests/conftest.py` owns all shared setup**: the `Agg` backend, putting
  `src` on `sys.path`, and closing figures after each test. Never repeat any of
  this in an individual test file.
- **Never import `shap` in unit tests.** Use `tests/_helpers.py::FakeExplanation`,
  a duck-typed stand-in exposing `.values`, `.data`, `.feature_names`, and
  `.base_values`. The real `shap` pulls in numba/llvmlite and makes
  `pip install -e .[dev]` slow and fragile; keep it in Layer 2 only.
- **Build inputs with the shared factories**, `make_explanation()` and
  `make_single_explanation()`, rather than hand-rolling arrays per file.
- **Unit-test private helpers that contain real logic** directly -
  `normalize_column`, `_fmt`, each module's `_analysis_line`. Exercising them
  only through a full chart render means a failure reports "the chart is wrong"
  instead of naming the broken function.
- **Assert something that can actually fail.** `assert fig is not None` is
  noise; `plt.subplots()` never returns `None`. Assert on tick labels, artist
  counts, colours, or text content.
- **Test the error type, not just the message.** Every user-facing failure
  should be `ShapEditorialError`. Because it subclasses `ValueError`, a test
  written as `pytest.raises(ValueError)` will pass even when the wrong type is
  raised - which is exactly how one such bug survived in `beeswarm`.

### What a new chart type needs

Match the existing files. At minimum: renders without error; respects
`max_display`; orders most-important-first; `show_other` adds or omits the
aggregate row correctly; rejects multiclass, bad `max_display`, and missing
required fields; honours `analysis` / `highlight` / `transparent`; draws onto a
supplied `ax`; and does not leak rcParams. Then add it to the parametrised
fixtures in `tests/test_theme.py`.

---

## Layer 2 - Integration against real SHAP

The unit suite deliberately never imports `shap`, so the example scripts are
the **only** thing verifying that duck-typed extraction still works against
genuine `shap.Explanation` objects, real trained models, and real dtypes.

```bash
uv run --extra example python examples/beeswarm_quickstart.py
```

If dependency resolution picks an ancient numba that will not build (common on
newer Pythons), use the isolated env instead - this is the reliable path:

```bash
uv run --no-project --python 3.12 --with-editable . \
    --with shap --with scikit-learn --with matplotlib --with "numpy<2.1" \
    python examples/beeswarm_gallery.py
```

The eight scripts (`<chart>_quickstart.py`, `<chart>_gallery.py` for beeswarm,
waterfall, bar, and scatter) cover binary classification, regression, multiclass
with few and with many features, the `show_other` toggle, transparent
backgrounds, a second model family, and - for scatter - feature selection and
interaction colouring. Case file names are kept parallel across chart types so
the same scenario can be compared between them. `beeswarm_comparison.py`
additionally renders the stock-SHAP-vs-editorial before/after asset.

---

## Layer 3 - Visual regression

Layers 1 and 2 both pass while the chart renders in the wrong typeface, with
overlapping labels, or with a clipped colour key. Layer 3 is the only thing
that sees the actual output.

The method is to render the **same examples from the previous commit in the
same environment** and diff pixel-for-pixel. Using a git worktree keeps the
working tree untouched, and rendering both sides in one env means any
difference is attributable to the code rather than to a library upgrade.

```bash
# 1. Baseline: previous committed code, in a throwaway worktree
git worktree add _baseline HEAD
cd _baseline
uv run --no-project --python 3.12 --with-editable . \
    --with shap --with scikit-learn --with matplotlib --with "numpy<2.1" \
    python examples/beeswarm_gallery.py    # ...and the other five scripts
cd ..

# 2. Current: your working tree, identical env
uv run --no-project --python 3.12 --with-editable . \
    --with shap --with scikit-learn --with matplotlib --with "numpy<2.1" \
    python examples/beeswarm_gallery.py    # ...and the other five scripts

# 3. Diff
uv run --extra dev python tools/visual_diff.py \
    _baseline/examples/images examples/images

# 4. Clean up
git worktree remove _baseline --force
```

### Reading the result

`tools/visual_diff.py` reports `IDENTICAL`, `DIFFERS` (with changed-pixel count
and max channel delta), or `SIZE`, and exits non-zero if anything changed.

- **A refactor should report every image identical.** Anything else means you
  changed rendering, whether you meant to or not. Note the tool walks whichever
  PNGs are present, so clear out scratch images first or they show up too.
- **A deliberate visual change should differ only in the charts you touched.**
  If a waterfall tweak also moves every beeswarm, something is shared that you
  did not intend to share.
- **`SIZE` on many files at once is a red flag.** The examples save with
  `bbox_inches="tight"`, so the canvas is sized to the text. Widespread width
  changes almost always mean text metrics moved, which usually means the font
  changed.

### When the diff is legitimate

Commit the regenerated PNGs along with the code change. When it is not
legitimate, fix the code - do not regenerate to make the diff go away.

---

## Why Layer 3 is not optional: the DejaVu incident

Chart functions were changed to apply the theme inside `mpl.rc_context()` so a
call would stop mutating the caller's global `rcParams`. The entire unit suite
passed. The examples ran clean. Every rendered PNG silently changed typeface.

A matplotlib `Text` artist copies `font.family` when it is **created**, but if
that value is the generic `"sans-serif"` the real face is only resolved against
`font.sans-serif` when the artist is **drawn**. Drawing happens at
`fig.savefig(...)` - after the chart function has returned and its rc context
has exited. So every chart fell back to DejaVu Sans:

```
inside ctx  resolved font : arial.ttf
after  ctx  resolved font : DejaVuSans.ttf
```

No assertion on in-memory state could see this, because in-memory state was
correct. Only the pixels were wrong. The fix was for `_theme.py` to set
`font.family` to the concrete `FONT_STACK` rather than the generic name, so
each artist carries real font names and resolves identically on either side of
the context boundary. `tests/test_theme.py` now guards it.

Two lessons worth keeping:

1. **Anything resolved lazily at draw time escapes the unit suite.** Fonts are
   the obvious case; `savefig.*` params are another. When you touch theming or
   figure lifecycle, Layer 3 is mandatory.
2. **Verify a new regression test actually fails.** Revert the fix, confirm the
   test goes red, restore the fix. A test written after the fact that never saw
   the bug is not proven to catch it.

---

## Checklists

**Before every commit**

```bash
uv run --extra dev python -m pytest tests/ -q
uv run ruff format --check .
uv run ruff check .
```

**Before a release, or after touching theming, layout, or figure lifecycle**

Everything above, plus Layer 2 (all six example scripts run clean) and Layer 3
(visual diff reviewed, and regenerated PNGs committed if the change was
intended).

## Housekeeping

The isolated `uv run --no-project --with ...` invocations unpack wheels into the
shared uv cache, which grows quickly. Reclaim it with:

```bash
uv cache prune     # removes unused entries only; safe for other projects
```

Prefer `prune` over `uv cache clean`, which wipes the cache for every project on
the machine.
