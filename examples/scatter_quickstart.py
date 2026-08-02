"""End-to-end example: train a real model, compute real SHAP values,
render an editorial dependence scatter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

import shap_editorial as se

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

model = RandomForestClassifier(n_estimators=200, random_state=0)
model.fit(X, y)

explainer = shap.TreeExplainer(model)
explanation = explainer(X)

# Class 0 = malignant in sklearn's coding, so positive SHAP values push the
# prediction toward malignant - which is what the direction labels below say.
if explanation.values.ndim == 3:
    explanation = explanation[..., 0]

# "worst concave points" is the top driver in the beeswarm; this chart shows
# *how* it acts, with "worst radius" colouring the points to expose their
# interaction.
fig, ax = se.scatter(
    explanation,
    feature="worst concave points",
    color="worst radius",
    # Kept short: the colour key occupies the top-right, so a title much longer
    # than this would run into it.
    title="How “worst concave points” drives malignancy",
    source="Data: sklearn breast cancer dataset · Model: Random Forest (200 trees)",
    direction_labels=("↑ toward malignant", "↓ toward benign"),
)

out_dir = Path(__file__).resolve().parent / "images" / "scatter"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "hero.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved {out_path}")
