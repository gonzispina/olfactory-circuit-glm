#!/usr/bin/env python3
"""Make one held-out OR+CR prediction figure for the course-project note."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_spike_events
from src.features import build_binned_dataset
from src.model import _designs, trial_split


def main() -> None:
    events = load_spike_events(ROOT.parent / "neuroscience" / "MatrizFinal_8col.csv")
    events = events[events.trial_type.eq(1)].copy()  # OR + CR
    data = build_binned_dataset(events, target_neuron=45, coupling_neurons=[52])
    train_trials, test_trials = trial_split(data.trial_ids, data.trial_types, test_size=.2, seed=0)
    train = np.isin(data.trial_ids, train_trials)
    test = np.isin(data.trial_ids, test_trials)
    design = _designs(data)["stimulus_history"]
    model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=.1, max_iter=1_000))
    model.fit(design[train], data.y[train])
    predicted = model.predict(design[test])

    n_trials = len(test_trials)
    n_bins = len(predicted) // n_trials
    observed_trials = data.y[test].reshape(n_trials, n_bins) / .05
    predicted_trials = predicted.reshape(n_trials, n_bins) / .05
    time = np.linspace(-2.0, 3.0, n_bins, endpoint=False) + .025

    rng = np.random.default_rng(0)
    bootstrap = np.stack([
        observed_trials[rng.integers(0, n_trials, n_trials)].mean(axis=0)
        for _ in range(2_000)
    ])
    lower, upper = np.percentile(bootstrap, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=(9, 4.3), constrained_layout=True)
    ax.fill_between(time, lower, upper, color="#b8c2cc", alpha=.55, label="Observed 95% bootstrap interval")
    ax.plot(time, observed_trials.mean(axis=0), color="#222", lw=1.8, label="Observed mean rate")
    ax.plot(time, predicted_trials.mean(axis=0), color="#2a78d6", lw=2.2, label="Poisson GLM prediction")
    ax.axvline(0, color="#111", lw=1, linestyle="--")
    ax.set(xlabel="Time from first odor inhalation (s)", ylabel="Firing rate (Hz)",
           title="Neuron 45 · held-out OR + CR trials (13 trials)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    output = ROOT / "results" / "figures" / "neuron_45_orcr_heldout_glm_prediction.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
