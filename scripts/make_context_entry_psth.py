#!/usr/bin/env python3
"""Render the OR+CR neuron-45/52 PSTH aligned to first odor inhalation."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_spike_events, trial_metadata


DATA = ROOT.parent / "neuroscience" / "MatrizFinal_8col.csv"
OUTPUT = ROOT / "results" / "figures" / "neuron45-orcr-first-odor-psth.png"


def main() -> None:
    events = load_spike_events(DATA)
    metadata = trial_metadata(events)
    trials = metadata[metadata.trial_type == 1]  # OR + CR
    edges = np.arange(-2.5, 2.0001, 0.05)
    curves = {45: [], 52: []}
    for trial in trials.itertuples():
        for neuron in curves:
            spikes = events[
                (events.neuron_id == neuron) & (events.trial_number == trial.trial_number)
            ].spike_time.to_numpy()
            relative = spikes - trial.first_odor_inhale
            curves[neuron].append(np.histogram(relative, bins=edges)[0] / 0.05)
    mean_rates = {neuron: np.asarray(values).mean(axis=0) for neuron, values in curves.items()}
    centers = (edges[:-1] + edges[1:]) / 2

    fig, axis = plt.subplots(figsize=(10.2, 4.3))
    axis.plot(centers, mean_rates[45], color="#b4473b", linewidth=2.2, label="Neuron 45")
    axis.plot(centers, mean_rates[52], color="#222222", linewidth=2.2, label="Neuron 52")
    axis.axvline(0, color="#111111", linewidth=1, linestyle="--")
    axis.axvspan(-2.0, -1.5, color="#d9d8d2", alpha=.72, label="Baseline\n−2.0 to −1.5 s")
    axis.axvspan(-.5, 0, color="#cde2fb", alpha=.68, label="Pre-odor\n−0.5 to 0 s")
    axis.axvspan(0, 1.0, color="#f8d9cb", alpha=.68, label="Post-odor\n0 to +1.0 s")
    axis.set_title("Neurons 45 and 52 · OR + CR · aligned to first odor inhalation", fontsize=13, fontweight="semibold")
    axis.set_xlabel("seconds relative to first odor inhalation")
    axis.set_ylabel("mean firing rate (Hz)")
    axis.set_xlim(-2.5, 2.0)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles, labels, loc="upper left", frameon=False, ncol=2, fontsize=9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.text(.5, .01, "Shaded windows are the intervals used for the statistical comparisons.", ha="center", fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, .05, 1, 1))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
