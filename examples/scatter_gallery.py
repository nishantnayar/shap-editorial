"""Gallery for `scatter`: how one feature's effect behaves, across datasets.

Exercises the full range - binary classification, regression, multiclass (one
class sliced), few vs many features, a low-cardinality feature that needs
jitter, a transparent background, and a second model - and saves one PNG per
case into `examples/images/scatter/`.

File names are kept parallel with the other galleries
(01_binary_classification, 02_regression, 03_multiclass_few_features,
04_multiclass_many_features, 06_transparent, 07_gradient_boosting) so the same
case can be compared across chart types. Slot 05 is each chart's distinctive
option: `show_other` for beeswarm, low-cardinality jitter here.

Run it (needs the optional `example` deps: shap + scikit-learn):

    uv run --extra example python examples/scatter_gallery.py

On Python 3.13, where shap/numba wheels lag, use an isolated 3.12 env:

    uv run --no-project --python 3.12 --with-editable . \
        --with shap --with scikit-learn --with matplotlib --with "numpy<2.1" \
        python examples/scatter_gallery.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import shap
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)

import shap_editorial as se

OUT = Path(__file__).resolve().parent / "images" / "scatter"
OUT.mkdir(parents=True, exist_ok=True)


def save(name, fig):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def rf(X, y, **kwargs):
    model = RandomForestClassifier(random_state=0, **kwargs).fit(X, y)
    return shap.TreeExplainer(model)(X)


# 01. Binary classification - breast cancer, malignant class (index 0), with a
# second feature colouring the points to expose an interaction.
bc = load_breast_cancer(as_frame=True)
e = rf(bc.data, bc.target, n_estimators=200)
fig, _ = se.scatter(
    e[..., 0],
    feature="worst concave points",
    color="worst radius",
    title="How “worst concave points” drives malignancy",
    source="Data: sklearn breast cancer · Model: Random Forest (200 trees)",
    direction_labels=("↑ toward malignant", "↓ toward benign"),
)
save("01_binary_classification", fig)

# 02. Regression - diabetes, letting `feature=None` pick the top driver.
db = load_diabetes(as_frame=True)
reg = RandomForestRegressor(n_estimators=200, random_state=0).fit(db.data, db.target)
e_reg = shap.TreeExplainer(reg)(db.data)
fig, _ = se.scatter(
    e_reg,
    color="age",
    title="The strongest driver of disease progression",
    source="Data: sklearn diabetes · Model: Random Forest regressor",
    direction_labels=("↑ higher progression", "↓ lower progression"),
)
save("02_regression", fig)

# 03. Multiclass, few features - iris, virginica class (only 4 features).
ir = load_iris(as_frame=True)
e = rf(ir.data, ir.target, n_estimators=200)
fig, _ = se.scatter(
    e[..., 2],
    title="What makes an iris read as virginica",
    source="Data: sklearn iris · Model: Random Forest · class = virginica",
    direction_labels=("↑ toward virginica", "↓ away from virginica"),
)
save("03_multiclass_few_features", fig)

# 04. Multiclass, many features - digits, the digit "8" (64 pixel features).
dg = load_digits(as_frame=True)
model = RandomForestClassifier(n_estimators=150, random_state=0).fit(dg.data, dg.target)
e = shap.TreeExplainer(model)(dg.data.iloc[:400])
fig, _ = se.scatter(
    e[..., 8],
    title="The pixel that most decides whether a digit reads as “8”",
    source="Data: sklearn digits · Model: Random Forest · class = 8",
    direction_labels=("↑ toward 8", "↓ away from 8"),
)
save("04_multiclass_many_features", fig)

# 05. Low-cardinality feature - diabetes "sex" takes two values, so the points
# would stack into two vertical lines without the automatic jitter.
fig, _ = se.scatter(
    e_reg,
    feature="sex",
    # The auto line phrases everything as "higher X", which reads oddly for a
    # two-valued feature, so this one supplies its own wording.
    analysis="The two groups sit on opposite sides of zero.",
    title="A two-valued feature, spread out by automatic jitter",
    source="Data: sklearn diabetes · Model: Random Forest regressor",
    direction_labels=("↑ higher progression", "↓ lower progression"),
)
save("05_low_cardinality", fig)

# 06. Transparent background (for dark slides / coloured backgrounds).
wn = load_wine(as_frame=True)
e = rf(wn.data, wn.target, n_estimators=150)
fig, _ = se.scatter(
    e[..., 0],
    title="Wine class 0 (transparent background)",
    source="Data: sklearn wine · Model: Random Forest",
    transparent=True,
)
save("06_transparent", fig)

# 07. Different model - gradient boosting (binary, single 2-D output).
gb = GradientBoostingClassifier(random_state=0).fit(bc.data, bc.target)
e = shap.TreeExplainer(gb)(bc.data)
expl = e[..., 0] if e.values.ndim == 3 else e
fig, _ = se.scatter(
    expl,
    feature="worst concave points",
    color="worst radius",
    title="Breast cancer, explained by gradient boosting",
    source="Data: sklearn breast cancer · Model: Gradient Boosting",
    direction_labels=("↑ toward malignant", "↓ toward benign"),
)
save("07_gradient_boosting", fig)

print(f"\nGallery written to {OUT}")
