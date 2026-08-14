# shap-editorial — Feedback & Next Iteration Notes
Source: LinkedIn post comments, "SHAP will tell you why your model made a prediction. It won't tell your audience."

## Actionable enhancements for v2

1. **Preserve density/distribution signal (Shashank Garewal)**
   - Original beeswarm shows density via point clustering — how common a given SHAP value is across the dataset.
   - Reformatted version optimizes for reading a single prediction and loses this.
   - Possible fix: offer both views — a "single prediction" mode (current simplified version) and a "distribution" mode that keeps density/jitter for audiences who need to see the full dataset pattern, not just one decision.
