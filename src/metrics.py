"""Held-out likelihood metrics for point-process models."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def poisson_log_likelihood(y: np.ndarray, rate: np.ndarray) -> float:
    """Full Poisson log likelihood, including the count factorial term."""
    rate = np.clip(np.asarray(rate, dtype=float), 1e-12, None)
    y = np.asarray(y, dtype=float)
    return float(np.sum(y * np.log(rate) - rate - gammaln(y + 1)))


def bits_per_spike(y: np.ndarray, rate: np.ndarray, baseline_rate: float) -> float:
    """Held-out information gain relative to a train-derived constant-rate model."""
    n_spikes = float(np.sum(y))
    if n_spikes == 0:
        return float("nan")
    ll_model = poisson_log_likelihood(y, rate)
    ll_baseline = poisson_log_likelihood(y, np.full_like(y, baseline_rate, dtype=float))
    return (ll_model - ll_baseline) / (n_spikes * np.log(2.0))
