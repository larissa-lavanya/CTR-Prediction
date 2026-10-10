"""Post-hoc calibration: fix the probabilities of a model without changing how it ranks (much)."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ctr_pipeline.utils import logit, sigmoid


class PlattScaler:
    """p_new = sigmoid(a * logit(p) + b): two parameters, keeps the ranking exactly."""

    def fit(self, p, y):
        # A huge C turns off regularization: this is plain maximum-likelihood logistic regression
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self.model.fit(logit(p).reshape(-1, 1), y.astype(int))
        self.a = float(self.model.coef_[0, 0])
        self.b = float(self.model.intercept_[0])
        return self

    def transform(self, p):
        return sigmoid(self.a * logit(p) + self.b)


class IsotonicCalibrator:
    """Monotone step function fitted to the data: more flexible than Platt, but can create ties."""

    def fit(self, p, y):
        self.model = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
        self.model.fit(p, y)
        return self

    def transform(self, p):
        return np.asarray(self.model.predict(p))


CALIBRATORS = {"platt": PlattScaler, "isotonic": IsotonicCalibrator}
