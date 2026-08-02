"""End-to-end example: train a real model, compute real SHAP values,
render an editorial beeswarm plot."""

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

# TreeExplainer on a binary classifier returns shape (n, features, 2)
# classes here; take the "malignant present" class (index 1).
if explanation.values.ndim == 3:
    explanation = explanation[..., 1]

fig, ax = se.beeswarm(
    explanation,
    max_display=10,
    title="What drives the malignancy prediction",
    source="Data: sklearn breast cancer dataset · Model: Random Forest (200 trees)",
)

out_path = Path(__file__).resolve().parent / "beeswarm_output.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved {out_path}")
