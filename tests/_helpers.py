import numpy as np


class FakeExplanation:
    """Duck-typed stand-in for shap.Explanation, so tests don't need
    the real (heavy) shap dependency."""

    def __init__(self, values, data=None, feature_names=None, base_values=None):
        self.values = values
        self.data = data
        self.feature_names = feature_names
        self.base_values = base_values


def make_explanation(n_samples=40, n_features=6, seed=0, with_data=True):
    """Multi-sample explanation, as beeswarm and bar expect."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_samples, n_features))
    data = rng.uniform(size=(n_samples, n_features)) if with_data else None
    names = [f"feat_{i}" for i in range(n_features)]
    return FakeExplanation(values, data, names)


def make_single_explanation(n_features=6, seed=0, base=0.5):
    """Single-instance explanation, as waterfall expects."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=n_features)
    data = rng.uniform(size=n_features)
    names = [f"feat_{i}" for i in range(n_features)]
    return FakeExplanation(values, data, names, base_values=base)
