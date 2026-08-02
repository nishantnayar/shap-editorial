"""Gallery for `waterfall`: single-prediction explanations across datasets.

Exercises the range the chart is built for — binary classification, regression,
multiclass, few vs many features, the show_other toggle, a transparent
background, and a second model — saving one PNG per case into
`examples/images/waterfall/`.

File names are kept parallel with the `beeswarm` gallery
(01_binary_classification, 02_regression, 03_multiclass_few_features,
04_multiclass_many_features, 05_show_other, 06_transparent, ...) so the same
case can be compared across chart types.

Run it (needs the optional `example` deps: shap + scikit-learn):

    uv run --extra example python examples/waterfall_gallery.py

On Python 3.13, where shap/numba wheels lag, use an isolated 3.12 env:

    uv run --no-project --python 3.12 --with-editable . \
        --with shap --with scikit-learn --with matplotlib --with "numpy<2.1" \
        python examples/waterfall_gallery.py
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
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)

import shap_editorial as se

OUT = Path(__file__).resolve().parent / "images" / "waterfall"
OUT.mkdir(parents=True, exist_ok=True)


def save(name, fig):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def first_index(y, value):
    return int((y == value).to_numpy().argmax())


# Breast cancer (binary): explain class 0 = malignant.
bc = load_breast_cancer(as_frame=True)
rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(bc.data, bc.target)
e_bc = shap.TreeExplainer(rf)(bc.data)[..., 0]  # P(malignant)

# 01. Binary classification — a malignant case (contributions push up, red).
fig, _ = se.waterfall(
    e_bc[first_index(bc.target, 0)],
    title="Why this case was predicted malignant",
    source="Data: sklearn breast cancer · Model: Random Forest",
)
save("01_binary_classification", fig)

# 02. Regression — diabetes progression for one patient.
db = load_diabetes(as_frame=True)
reg = RandomForestRegressor(n_estimators=200, random_state=0).fit(db.data, db.target)
e_db = shap.TreeExplainer(reg)(db.data)
fig, _ = se.waterfall(
    e_db[0],
    title="What drives this patient's predicted progression",
    source="Data: sklearn diabetes · Model: Random Forest regressor",
)
save("02_regression", fig)

# 03. Multiclass, few features — iris (4 features), virginica class.
ir = load_iris(as_frame=True)
rf_ir = RandomForestClassifier(n_estimators=200, random_state=0).fit(ir.data, ir.target)
e_ir = shap.TreeExplainer(rf_ir)(ir.data)[..., 2]
fig, _ = se.waterfall(
    e_ir[first_index(ir.target, 2)],
    title="Why this flower was classified virginica",
    source="Data: sklearn iris · Model: Random Forest · class = virginica",
)
save("03_multiclass_few_features", fig)

# 04. Multiclass, many features — digits (64 pixels), the digit "8".
dg = load_digits(as_frame=True)
rf_dg = RandomForestClassifier(n_estimators=150, random_state=0).fit(dg.data, dg.target)
e_dg = shap.TreeExplainer(rf_dg)(dg.data.iloc[:200])[..., 8]
fig, _ = se.waterfall(
    e_dg[0],
    title="Which pixels made this digit read as “8”",
    source="Data: sklearn digits · Model: Random Forest · class = 8",
)
save("04_multiclass_many_features", fig)

# 05. The show_other toggle — top features only (bars stop short; the gap is
#     the hidden contributions).
fig, _ = se.waterfall(
    e_bc[first_index(bc.target, 0)],
    max_display=6,
    show_other=False,
    title="Top 6 drivers only (show_other=False)",
    source="Data: sklearn breast cancer · Model: Random Forest",
)
save("05_show_other", fig)

# 06. Transparent background (for dark slides / coloured backgrounds).
fig, _ = se.waterfall(
    e_bc[first_index(bc.target, 0)],
    title="Malignant case (transparent background)",
    source="Data: sklearn breast cancer · Model: Random Forest",
    transparent=True,
)
save("06_transparent", fig)

# 07. Different model — gradient boosting (binary, single 2-D output).
gb = GradientBoostingClassifier(random_state=0).fit(bc.data, bc.target)
e_gb = shap.TreeExplainer(gb)(bc.data)
e_gb = e_gb[..., 0] if e_gb.values.ndim == 3 else e_gb
fig, _ = se.waterfall(
    e_gb[first_index(bc.target, 0)],
    title="Malignant case, explained by gradient boosting",
    source="Data: sklearn breast cancer · Model: Gradient Boosting",
)
save("07_gradient_boosting", fig)

print(f"\nGallery written to {OUT}")
