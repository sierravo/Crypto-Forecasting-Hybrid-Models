"""
Simple forecasting baselines for decision-grade model comparison.

These baselines are intentionally lightweight. They provide reference points
that help answer whether the neural models are adding value beyond simple,
low-maintenance alternatives.
"""

from collections import deque
from typing import Deque, Optional

import torch


class BaseBaseline:
    """
    Base class for non-trainable forecasting baselines.

    Baselines expose the same predict/update pattern so evaluation can avoid
    target leakage: predict first using only prior information, then update
    the baseline state with the current target after the error is measured.
    """

    name = "base"

    def reset(self):
        """Reset any state carried across chronological samples."""
        return None

    def predict(self, features, target, adj=None):
        """Return a prediction with the same shape as target."""
        raise NotImplementedError

    def update(self, target):
        """Update baseline state after the current target has been evaluated."""
        return None


class ZeroBaseline(BaseBaseline):
    """
    Predict zero for every asset.

    This is a strong basic baseline for noisy financial targets, especially
    when the target is centered around zero.
    """

    name = "zero"

    def predict(self, features, target, adj=None):
        return torch.zeros_like(target)


class PreviousTargetBaseline(BaseBaseline):
    """
    Predict the previous observed target vector.

    For the first chronological sample, this baseline predicts zeros because no
    prior target has been observed yet.
    """

    name = "previous_target"

    def __init__(self, initial_value: float = 0.0):
        self.initial_value = initial_value
        self.previous_target: Optional[torch.Tensor] = None

    def reset(self):
        self.previous_target = None

    def predict(self, features, target, adj=None):
        if self.previous_target is None:
            return torch.full_like(target, fill_value=self.initial_value)
        return self.previous_target.to(device=target.device, dtype=target.dtype)

    def update(self, target):
        self.previous_target = target.detach().clone()


class RollingMeanTargetBaseline(BaseBaseline):
    """
    Predict the rolling mean of the last N observed target vectors.

    For the first chronological sample, this baseline predicts zeros because no
    prior target history has been observed yet.
    """

    name = "rolling_mean_target"

    def __init__(self, window: int = 5, initial_value: float = 0.0):
        if window <= 0:
            raise ValueError("window must be greater than 0")
        self.window = window
        self.initial_value = initial_value
        self.history: Deque[torch.Tensor] = deque(maxlen=window)

    def reset(self):
        self.history.clear()

    def predict(self, features, target, adj=None):
        if not self.history:
            return torch.full_like(target, fill_value=self.initial_value)
        stacked = torch.stack([
            item.to(device=target.device, dtype=target.dtype) for item in self.history
        ], dim=0)
        return stacked.mean(dim=0)

    def update(self, target):
        self.history.append(target.detach().clone())


def create_baseline(name: str, rolling_window: int = 5) -> BaseBaseline:
    """
    Factory for supported baseline predictors.

    Args:
        name: one of zero, previous_target, or rolling_mean.
        rolling_window: target-history window used by rolling_mean.

    Returns:
        A baseline predictor instance.
    """
    normalized_name = name.lower()

    if normalized_name in {"zero", "baseline_zero"}:
        return ZeroBaseline()
    if normalized_name in {"previous_target", "baseline_previous_target", "persistence"}:
        return PreviousTargetBaseline()
    if normalized_name in {"rolling_mean", "rolling_mean_target", "baseline_rolling_mean"}:
        return RollingMeanTargetBaseline(window=rolling_window)

    raise ValueError(
        "Unsupported baseline '{}'. Choose from zero, previous_target, or rolling_mean.".format(name)
    )


BASELINE_NAMES = {"zero", "previous_target", "rolling_mean"}
