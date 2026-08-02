"""Gallery for `bar`: global feature importance across datasets and tasks.

File names are kept parallel with the beeswarm and waterfall galleries
(01_binary_classification, 02_regression, 03_multiclass_few_features,
04_multiclass_many_features, 05_show_other, 06_transparent, 07_gradient_boosting)
so the same case can be compared across chart types.

    uv run --extra example python examples/bar_gallery.py

On Python 3.13, use an isolated 3.12 env:

    uv run --no-project --python 3.12 --with-editable . \
        --with shap --with scikit-learn --with matplotlib --with "numpy<2.1" \
        python examples/bar_gallery.py
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

OUT = Path(__file__).resolve().parent / "images" / "bar"
OUT.mkdir(parents=True, exist_ok=True)


def save(name, fig):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def rf(X, y, **kwargs):
    model = RandomForestClassifier(random_state=0, **kwargs).fit(X, y)
    return shap.TreeExplainer(model)(X)


# 01. Binary classification.
bc = load_breast_cancer(as_frame=True)
e = rf(bc.data, bc.target, n_estimators=200)
fig, _ = se.bar(
    e[..., 0],
    title="Which features matter most for the malignancy model",
    source="Data: sklearn breast cancer · Model: Random Forest (200 trees)",
)
save("01_binary_classification", fig)

# 02. Regression.
db = load_diabetes(as_frame=True)
reg = RandomForestRegressor(n_estimators=200, random_state=0).fit(db.data, db.target)
fig, _ = se.bar(
    shap.TreeExplainer(reg)(db.data),
    title="Which features drive predicted disease progression",
    source="Data: sklearn diabetes · Model: Random Forest regressor",
)
save("02_regression", fig)

# 03. Multiclass, few features.
ir = load_iris(as_frame=True)
e = rf(ir.data, ir.target, n_estimators=200)
fig, _ = se.bar(
    e[..., 2],
    title="Which features matter most for the virginica class",
    source="Data: sklearn iris · Model: Random Forest · class = virginica",
)
save("03_multiclass_few_features", fig)

# 04. Multiclass, many features.
dg = load_digits(as_frame=True)
model = RandomForestClassifier(n_estimators=150, random_state=0).fit(dg.data, dg.target)
e = shap.TreeExplainer(model)(dg.data.iloc[:400])
fig, _ = se.bar(
    e[..., 8],
    title="Which pixels matter most for reading a digit as “8”",
    source="Data: sklearn digits · Model: Random Forest · class = 8",
)
save("04_multiclass_many_features", fig)

# 05. The opt-in aggregate bar.
fig, _ = se.bar(
    e[..., 8],
    title="Digit “8” importance, with the remaining pixels summed",
    source="Data: sklearn digits · Model: Random Forest · class = 8",
    show_other=True,
)
save("05_show_other", fig)

# 06. Transparent background.
wn = load_wine(as_frame=True)
e = rf(wn.data, wn.target, n_estimators=150)
fig, _ = se.bar(
    e[..., 0],
    title="Wine class 0 importance (transparent background)",
    source="Data: sklearn wine · Model: Random Forest",
    transparent=True,
)
save("06_transparent", fig)

# 07. Different model — gradient boosting.
gb = GradientBoostingClassifier(random_state=0).fit(bc.data, bc.target)
e = shap.TreeExplainer(gb)(bc.data)
expl = e[..., 0] if e.values.ndim == 3 else e
fig, _ = se.bar(
    expl,
    title="Breast cancer importance, via gradient boosting",
    source="Data: sklearn breast cancer · Model: Gradient Boosting",
)
save("07_gradient_boosting", fig)

print(f"\nGallery written to {OUT}")
