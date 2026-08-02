"""Quickstart for `bar`: global feature importance end-to-end.

Trains a RandomForest on the breast-cancer dataset, computes real SHAP values,
and renders a feature-importance bar chart into
`examples/images/bar/hero.png`.

Run it (needs the optional `example` deps: shap + scikit-learn):

    uv run --extra example python examples/bar_quickstart.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

import shap_editorial as se

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target  # target: 0 = malignant, 1 = benign

model = RandomForestClassifier(n_estimators=200, random_state=0)
model.fit(X, y)

explanation = shap.TreeExplainer(model)(X)[..., 0]  # class 0 = malignant

fig, ax = se.bar(
    explanation,
    title="Which features matter most for the malignancy model",
    source="Data: sklearn breast cancer · Model: Random Forest (200 trees)",
)

out_dir = Path(__file__).resolve().parent / "images" / "bar"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "hero.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved {out_path}")
