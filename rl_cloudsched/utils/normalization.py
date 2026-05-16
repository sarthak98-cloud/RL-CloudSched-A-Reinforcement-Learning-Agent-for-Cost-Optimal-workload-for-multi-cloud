"""
normalization.py
----------------
Online min-max and z-score feature normalizers used to
keep all state observations in a consistent numerical range.
"""

import numpy as np
from typing import Optional


class FeatureNormalizer:
    """
    Running min-max normalizer that updates bounds across episodes.
    Falls back to unit output when min == max (constant feature).
    """

    def __init__(self, n_features: int, clip_range: tuple = (0.0, 1.0)):
        self.n = n_features
        self.clip_lo, self.clip_hi = clip_range
        self.mins = np.full(n_features,  1e9)
        self.maxs = np.full(n_features, -1e9)
        self._fitted = False

    def update(self, x: np.ndarray):
        """Update running min/max from a new observation vector."""
        self.mins = np.minimum(self.mins, x)
        self.maxs = np.maximum(self.maxs, x)
        self._fitted = True

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize x to [clip_lo, clip_hi]."""
        if not self._fitted:
            return np.zeros_like(x, dtype=np.float32)
        denom = self.maxs - self.mins
        denom = np.where(denom < 1e-8, 1.0, denom)
        out = (x - self.mins) / denom
        return np.clip(out, self.clip_lo, self.clip_hi).astype(np.float32)

    def reset(self):
        self.mins = np.full(self.n,  1e9)
        self.maxs = np.full(self.n, -1e9)
        self._fitted = False


def cyclical_encode(value: float, period: float) -> tuple:
    """
    Encode a periodic value as (sin, cos) pair to preserve continuity.
    E.g. hour-of-day: value=23 and value=0 should be close together.
    """
    angle = 2.0 * np.pi * value / period
    return float(np.sin(angle)), float(np.cos(angle))
