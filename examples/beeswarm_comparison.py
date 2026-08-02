"""Article asset: SHAP's stock beeswarm beside the shap-editorial one.

Both panels are rendered from the *same* explainer output on the same model,
so the only thing that differs between them is presentation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

import shap_editorial as se

MAX_DISPLAY = 10

out_dir = Path(__file__).resolve().parent / "images" / "beeswarm"
out_dir.mkdir(parents=True, exist_ok=True)

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

model = RandomForestClassifier(n_estimators=200, random_state=0)
model.fit(X, y)

explainer = shap.TreeExplainer(model)
explanation = explainer(X)

# Class 0 = malignant in sklearn's coding, matching the "after" panel's title.
if explanation.values.ndim == 3:
    explanation = explanation[..., 0]

# Stock SHAP on stock matplotlib, so the "before" is genuinely what a reader
# gets out of the box rather than our theme leaking into it.
plt.rcdefaults()
shap.summary_plot(explanation.values, X, max_display=MAX_DISPLAY, show=False)
before_path = out_dir / "comparison_before.png"
plt.gcf().savefig(before_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close("all")

fig, _ = se.beeswarm(
    explanation,
    max_display=MAX_DISPLAY,
    title="What drives the malignancy prediction",
    source="Data: sklearn breast cancer dataset · Model: Random Forest (200 trees)",
    direction_labels=("← toward benign", "toward malignant →"),
)
after_path = out_dir / "comparison_after.png"
fig.savefig(after_path, dpi=200, bbox_inches="tight")
plt.close(fig)


CAPTIONS = ["Before: shap.summary_plot()", "After: shap_editorial.beeswarm()"]


def compose_side_by_side(panels, dest, height=6.0):
    """Panels scaled to a common height, so widths carry the aspect ratios."""
    plt.rcdefaults()
    imgs = [mpimg.imread(p) for p in panels]
    aspects = [im.shape[1] / im.shape[0] for im in imgs]

    fig, axes = plt.subplots(
        1,
        len(imgs),
        figsize=(height * sum(aspects), height + 0.4),
        gridspec_kw={"width_ratios": aspects},
    )
    for ax, im, caption in zip(axes, imgs, CAPTIONS, strict=True):
        ax.imshow(im)
        ax.set_title(caption, fontsize=13, fontweight="bold", pad=10)
        ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.04)
    fig.savefig(dest, dpi=200, facecolor="white")
    plt.close(fig)


def compose_stacked(panels, dest, width=8.0):
    """Panels scaled to a common width. Keeps type sizes honest between the
    two, and the portrait shape suits a social feed."""
    plt.rcdefaults()
    imgs = [mpimg.imread(p) for p in panels]
    heights = [width * im.shape[0] / im.shape[1] for im in imgs]

    fig, axes = plt.subplots(
        len(imgs),
        1,
        figsize=(width, sum(heights) + 0.7),
        gridspec_kw={"height_ratios": heights},
    )
    for ax, im, caption in zip(axes, imgs, CAPTIONS, strict=True):
        ax.imshow(im)
        ax.set_title(caption, fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01, hspace=0.08)
    fig.savefig(dest, dpi=200, facecolor="white")
    plt.close(fig)


panels = [before_path, after_path]
side_path = out_dir / "comparison.png"
stacked_path = out_dir / "comparison_stacked.png"
compose_side_by_side(panels, side_path)
compose_stacked(panels, stacked_path)

for p in (before_path, after_path, side_path, stacked_path):
    print(f"Saved {p}")
