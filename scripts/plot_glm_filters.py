#!/usr/bin/env python3
"""Plot regularized causal GLM filters used in the portfolio note.

The figures are descriptive fits to all selected trials. Held-out evidence is
reported separately by ``run_evaluation.py``; coefficient shapes do not imply a
causal connection.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_spike_events
from src.features import STIMULUS_NAMES, build_binned_dataset
from src.model import _fit_ridge, _select_ridge_alpha


DATA = ROOT.parent / "neuroscience" / "MatrizFinal_8col.csv"
OUTPUT = ROOT / "results" / "figures"
BIN_SIZE = 0.05
N_STIM_LAGS = 25
N_HISTORY_LAGS = 30
ALPHA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def fit_raw_coefficients(dataset, design: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a selected ridge GLM and undo StandardScaler's coefficient scale."""
    alpha = _select_ridge_alpha(
        design,
        dataset.y,
        dataset.trial_ids,
        dataset.trial_types,
        alpha_grid=ALPHA_GRID,
    )
    pipeline = _fit_ridge(design, dataset.y, alpha)
    scaler = pipeline.named_steps["standardscaler"]
    regressor = pipeline.named_steps["poissonregressor"]
    return regressor.coef_ / scaler.scale_, alpha


def plot_neuron52_event_filters(events) -> None:
    data = build_binned_dataset(
        events[events["trial_type"].isin([1, 2, 3, 4])],
        target_neuron=52,
        coupling_neurons=(),
        stimulus_encoding="identity",
    )
    weights, alpha = fit_raw_coefficients(data, data.X_stimulus)
    filters = weights.reshape(len(STIMULUS_NAMES), N_STIM_LAGS)
    time = np.arange(N_STIM_LAGS) * BIN_SIZE
    # Match the condition palette used in the raster/PSTH figures: rewarded
    # responses use the dark red/orange family; unrewarded responses use
    # black/gray. The same semantic colors are reused for entry and exit.
    colors = {"OR": "#b64b3f", "OU": "#333333", "CR": "#b64b3f", "CU": "#e18420", "CER": "#b64b3f", "CEU": "#8b8b8b"}
    labels = {
        "OR": "OR: Odor Rewarded",
        "OU": "OU: Odor Unrewarded",
        "CR": "CR: Context Entry Rewarded",
        "CU": "CU: Context Entry Unrewarded",
        "CER": "CER: Context Exit Rewarded",
        "CEU": "CEU: Context Exit Unrewarded",
    }

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.7), sharex=True, sharey=True)
    for axis, name, curve in zip(axes.flat, STIMULUS_NAMES, filters):
        axis.axhline(0, color="#898781", linewidth=0.8, linestyle="--")
        axis.plot(time, curve, color=colors[name], linewidth=2, marker="o", markersize=3)
        axis.set_title(labels[name], fontsize=10.5, fontweight="semibold")
        axis.set_xlim(0, 1.2)
        axis.set_xlabel("seconds after event")
    axes[0, 0].set_ylabel("GLM weight")
    axes[1, 0].set_ylabel("GLM weight")
    fig.suptitle(f"Neuron 52: causal event filters (ridge α = {alpha:g})", fontsize=13, fontweight="semibold")
    fig.text(0.5, 0.01, "OR/OU are aligned to odor inhalation; CR/CU and CER/CEU to context entry/exit.", ha="center", fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "neuron52-identity-event-filters.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_52_to_45_coupling_filters(events) -> None:
    conditions = [(1, "OR + CR"), (2, "OR + CU"), (3, "OU + CR"), (4, "OU + CU")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), sharex=True, sharey=True)
    time = np.arange(1, N_HISTORY_LAGS + 1) * BIN_SIZE
    summaries = []
    for axis, (trial_type, label) in zip(axes.flat, conditions):
        data = build_binned_dataset(
            events[events["trial_type"] == trial_type],
            target_neuron=45,
            coupling_neurons=[52],
            stimulus_encoding="identity",
        )
        design = np.hstack([data.X_stimulus, data.X_history, data.X_coupling])
        weights, alpha = fit_raw_coefficients(data, design)
        coupling = weights[-N_HISTORY_LAGS:]
        summaries.append((label, alpha, coupling.min(), coupling.max()))
        axis.axhline(0, color="#898781", linewidth=0.8, linestyle="--")
        axis.plot(time, coupling, color="#d95926", linewidth=2, marker="o", markersize=3)
        axis.set_title(f"{label}  ·  ridge α = {alpha:g}", fontsize=11, fontweight="semibold")
        axis.set_xlabel("seconds after a spike in neuron 52")
        axis.set_xlim(0.05, 1.5)
    axes[0, 0].set_ylabel("52 → 45 GLM weight")
    axes[1, 0].set_ylabel("52 → 45 GLM weight")
    fig.suptitle("Neuron 52 history when predicting neuron 45", fontsize=13, fontweight="semibold")
    fig.text(0.5, 0.01, "Negative weight means a recent spike in 52 lowers the model's expected spike count for 45, conditional on the included task events and 45 history.", ha="center", fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "neuron45-from52-coupling-filters-by-condition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    for label, alpha, minimum, maximum in summaries:
        print(f"{label}: alpha={alpha:g}, min={minimum:.4f}, max={maximum:.4f}")


def main() -> None:
    events = load_spike_events(DATA)
    plot_neuron52_event_filters(events)
    plot_52_to_45_coupling_filters(events)


if __name__ == "__main__":
    main()
