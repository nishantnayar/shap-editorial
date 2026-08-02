class FakeExplanation:
    """Duck-typed stand-in for shap.Explanation, so tests don't need
    the real (heavy) shap dependency."""

    def __init__(self, values, data=None, feature_names=None, base_values=None):
        self.values = values
        self.data = data
        self.feature_names = feature_names
        self.base_values = base_values
