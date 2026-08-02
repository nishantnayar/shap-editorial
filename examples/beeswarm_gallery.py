"""Gallery for `beeswarm`: global feature-impact across datasets and tasks.

Exercises the full range — binary classification, multiclass (one class
sliced), regression, few vs many features, the opt-in aggregate row, a
transparent background, and a second model — and saves one PNG per case into
`examples/images/beeswarm/`.

File names are kept parallel with the `waterfall` gallery
(01_binary_classification, 02_regression, 03_multiclass_few_features,
04_multiclass_many_features, 05_show_other, 06_transparent, ...) so the same
case can be compared across chart types.

Run it (needs the optional `example` deps: shap + scikit-learn):

    uv run --extra example python examples/beeswarm_gallery.py

On Python 3.13, where shap/numba wheels lag, use an isolated 3.12 env:

    uv run --no-project --python 3.12 --with-editable . \
        --with shap --with scikit-learn --with matplotlib --with "numpy<2.1" \
        python examples/beeswarm_gallery.py
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

OUT = Path(__file__).resolve().parent / "images" / "beeswarm"
OUT.mkdir(parents=True, exist_ok=True)


def save(name, fig):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def rf(X, y, **kwargs):
    model = RandomForestClassifier(random_state=0, **kwargs).fit(X, y)
    return shap.TreeExplainer(model)(X)


# 01. Binary classification — breast cancer, malignant class (index 0).
bc = load_breast_cancer(as_frame=True)
e = rf(bc.data, bc.target, n_estimators=200)
fig, _ = se.beeswarm(
    e[..., 0],
    title="What drives the malignancy prediction",
    source="Data: sklearn breast cancer · Model: Random Forest (200 trees)",
    direction_labels=("← toward benign", "toward malignant →"),
)
save("01_binary_classification", fig)

# 02. Regression — diabetes disease progression (values are 2-D, no class axis).
db = load_diabetes(as_frame=True)
reg = RandomForestRegressor(n_estimators=200, random_state=0).fit(db.data, db.target)
e = shap.TreeExplainer(reg)(db.data)
fig, _ = se.beeswarm(
    e,
    title="What drives predicted disease progression",
    source="Data: sklearn diabetes · Model: Random Forest regressor",
    direction_labels=("← lower progression", "higher progression →"),
)
save("02_regression", fig)

# 03. Multiclass, few features — iris, virginica class (only 4 features).
ir = load_iris(as_frame=True)
e = rf(ir.data, ir.target, n_estimators=200)
fig, _ = se.beeswarm(
    e[..., 2],
    title="What drives an iris being classified virginica",
    source="Data: sklearn iris · Model: Random Forest · class = virginica",
    direction_labels=("← away from virginica", "toward virginica →"),
)
save("03_multiclass_few_features", fig)

# 04. Multiclass, many features — digits, the digit "8" (64 pixel features).
dg = load_digits(as_frame=True)
model = RandomForestClassifier(n_estimators=150, random_state=0).fit(dg.data, dg.target)
e = shap.TreeExplainer(model)(dg.data.iloc[:400])
fig, _ = se.beeswarm(
    e[..., 8],
    title="Which pixels drive a digit being read as “8”",
    source="Data: sklearn digits · Model: Random Forest · class = 8",
    direction_labels=("← away from 8", "toward 8 →"),
)
save("04_multiclass_many_features", fig)

# 05. The opt-in aggregate row — same digits case with show_other=True.
fig, _ = se.beeswarm(
    e[..., 8],
    title="Digit “8”, with the remaining pixels summed",
    source="Data: sklearn digits · Model: Random Forest · class = 8",
    show_other=True,
    direction_labels=("← away from 8", "toward 8 →"),
)
save("05_show_other", fig)

# 06. Transparent background (for dark slides / coloured backgrounds).
wn = load_wine(as_frame=True)
e = rf(wn.data, wn.target, n_estimators=150)
fig, _ = se.beeswarm(
    e[..., 0],
    title="Wine class 0 (transparent background)",
    source="Data: sklearn wine · Model: Random Forest",
    transparent=True,
)
save("06_transparent", fig)

# 07. Different model — gradient boosting (binary, single 2-D output).
gb = GradientBoostingClassifier(random_state=0).fit(bc.data, bc.target)
e = shap.TreeExplainer(gb)(bc.data)
expl = e[..., 0] if e.values.ndim == 3 else e
fig, _ = se.beeswarm(
    expl,
    title="Breast cancer, explained by gradient boosting",
    source="Data: sklearn breast cancer · Model: Gradient Boosting",
    direction_labels=("← toward benign", "toward malignant →"),
)
save("07_gradient_boosting", fig)

print(f"\nGallery written to {OUT}")
