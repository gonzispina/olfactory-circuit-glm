"""Nested regularized Poisson GLMs and trial-level held-out evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import BinnedDataset
from .metrics import bits_per_spike, poisson_log_likelihood


@dataclass(frozen=True)
class EvaluationResult:
    model: str
    n_train_trials: int
    n_test_trials: int
    n_test_bins: int
    n_test_spikes: int
    held_out_log_likelihood: float
    held_out_bits_per_spike: float
    selected_alpha: float | None


def trial_split(trial_ids: np.ndarray, trial_types: np.ndarray, *, test_size: float, seed: int):
    """Split complete trials, stratified by condition when feasible."""
    unique_ids, first_rows = np.unique(trial_ids, return_index=True)
    labels = trial_types[first_rows]
    if len(unique_ids) < 4:
        raise ValueError("Need at least four complete trials for held-out evaluation.")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    try:
        train_index, test_index = next(splitter.split(unique_ids, labels))
    except ValueError as error:
        raise ValueError(
            "Could not stratify trial split; each included condition needs enough trials."
        ) from error
    return unique_ids[train_index], unique_ids[test_index]


def _designs(data: BinnedDataset) -> dict[str, np.ndarray]:
    stimulus = data.X_stimulus
    self_history = data.X_history
    stimulus_history = np.hstack([stimulus, self_history])
    designs = {
        "stimulus": stimulus,
        "self_history": self_history,
        "stimulus_history": stimulus_history,
    }
    if data.X_coupling.shape[1] > 0:
        designs["self_history_coupling"] = np.hstack([self_history, data.X_coupling])
        designs["stimulus_history_coupling"] = np.hstack([stimulus_history, data.X_coupling])
    return designs


def _fit_ridge(design: np.ndarray, y: np.ndarray, alpha: float):
    estimator = make_pipeline(
        StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=1_000)
    )
    return estimator.fit(design, y)


def _select_ridge_alpha(
    design: np.ndarray,
    y: np.ndarray,
    trial_ids: np.ndarray,
    trial_types: np.ndarray,
    *,
    alpha_grid: tuple[float, ...],
) -> float:
    """Choose ridge strength using only training trials.

    The source paper selects its ridge prior by marginal likelihood. This
    scikit-learn implementation instead uses spike-normalized held-out
    log-likelihood across inner whole-trial folds, which preserves the same
    separation between tuning and the outer evaluation.
    """
    unique_ids, first_rows = np.unique(trial_ids, return_index=True)
    labels = trial_types[first_rows]
    _, counts = np.unique(labels, return_counts=True)
    n_splits = min(10, int(counts.min()))
    if n_splits < 2:
        return alpha_grid[0]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores: dict[float, float] = {}
    for alpha in alpha_grid:
        total_ll, total_spikes = 0.0, 0
        for fit_index, validation_index in splitter.split(unique_ids, labels):
            fit_ids, validation_ids = unique_ids[fit_index], unique_ids[validation_index]
            fit_mask = np.isin(trial_ids, fit_ids)
            validation_mask = np.isin(trial_ids, validation_ids)
            model = _fit_ridge(design[fit_mask], y[fit_mask], alpha)
            prediction = model.predict(design[validation_mask])
            total_ll += poisson_log_likelihood(y[validation_mask], prediction)
            total_spikes += int(y[validation_mask].sum())
        scores[alpha] = total_ll / max(total_spikes, 1)
    return max(scores, key=scores.get)


def evaluate_nested_models(
    data: BinnedDataset,
    *,
    test_size: float = 0.2,
    seed: int = 0,
    alpha_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
) -> list[EvaluationResult]:
    """Evaluate nested models on entire held-out trials.

    The baseline rate is estimated only from training bins. Every non-baseline
    model selects its ridge strength using inner whole-trial folds of the same
    training trials, so the outer held-out likelihood remains comparable.
    """
    train_trials, test_trials = trial_split(
        data.trial_ids, data.trial_types, test_size=test_size, seed=seed
    )
    train_mask = np.isin(data.trial_ids, train_trials)
    test_mask = np.isin(data.trial_ids, test_trials)
    y_train, y_test = data.y[train_mask], data.y[test_mask]
    baseline_rate = max(float(y_train.mean()), 1e-12)

    results = [
        EvaluationResult(
            model="constant_rate",
            n_train_trials=len(train_trials),
            n_test_trials=len(test_trials),
            n_test_bins=len(y_test),
            n_test_spikes=int(y_test.sum()),
            held_out_log_likelihood=poisson_log_likelihood(y_test, np.full_like(y_test, baseline_rate)),
            held_out_bits_per_spike=0.0,
            selected_alpha=None,
        )
    ]
    for name, design in _designs(data).items():
        if design.shape[1] == 0:
            continue
        selected_alpha = _select_ridge_alpha(
            design[train_mask],
            y_train,
            data.trial_ids[train_mask],
            data.trial_types[train_mask],
            alpha_grid=alpha_grid,
        )
        estimator = _fit_ridge(design[train_mask], y_train, selected_alpha)
        prediction = estimator.predict(design[test_mask])
        results.append(
            EvaluationResult(
                model=name,
                n_train_trials=len(train_trials),
                n_test_trials=len(test_trials),
                n_test_bins=len(y_test),
                n_test_spikes=int(y_test.sum()),
                held_out_log_likelihood=poisson_log_likelihood(y_test, prediction),
                held_out_bits_per_spike=bits_per_spike(y_test, prediction, baseline_rate),
                selected_alpha=selected_alpha,
            )
        )
    return results
