"""Quickstart for `waterfall`: explain a single prediction end-to-end.

Trains a RandomForest on the breast-cancer dataset, computes real SHAP values,
and renders a waterfall for one malignant case into
`examples/images/waterfall/hero.png`.

Run it (needs the optional `example` deps: shap + scikit-learn):

    uv run --extra example python examples/waterfall_quickstart.py
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

explanation = explainer = shap.TreeExplainer(model)(X)

# Explain one malignant case (class 0) — pick the first instance labelled 0.
idx = int((y == 0).to_numpy().argmax())
single = explanation[..., 0][idx]  # class 0, one instance -> 1-D values + base

fig, ax = se.waterfall(
    single,
    title="Why this case was predicted malignant",
    source="Data: sklearn breast cancer · Model: Random Forest (200 trees)",
)

out_dir = Path(__file__).resolve().parent / "images" / "waterfall"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "hero.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved {out_path}")
