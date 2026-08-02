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

# TreeExplainer on a binary classifier returns shape (n, features, 2).
# In sklearn's breast cancer dataset the target is coded 0 = malignant,
# 1 = benign, so we take class 0 to explain the *malignant* prediction —
# matching the title. (Slicing class 1 would explain P(benign) instead,
# which flips the direction of every effect and contradicts the title.)
if explanation.values.ndim == 3:
    explanation = explanation[..., 0]

fig, ax = se.beeswarm(
    explanation,
    max_display=10,
    title="What drives the malignancy prediction",
    source="Data: sklearn breast cancer dataset · Model: Random Forest (200 trees)",
    direction_labels=("← toward benign", "toward malignant →"),
)

out_dir = Path(__file__).resolve().parent / "images" / "beeswarm"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "hero.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved {out_path}")
