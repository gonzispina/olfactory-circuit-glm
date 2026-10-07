#!/usr/bin/env python3
"""Render event-aligned rasters and PSTHs for selected neurons."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_spike_events, trial_metadata


DEFAULT_DATA = ROOT.parent / "neuroscience" / "MatrizFinal_8col.csv"
LABELS = {1: "OR + CR", 2: "OR + CU", 3: "OU + CR", 4: "OU + CU"}
COLORS = {1: "#b4473b", 2: "#d9822b", 3: "#343434", 4: "#7a7a7a"}
ALIGNMENTS = {
    "odor": ("first_odor_inhale", "first odor inhalation"),
    "context-entry": ("ctx_entry_time", "context entry"),
    "context-exit": ("ctx_exit_time", "context exit"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--neurons", type=int, nargs="+", default=[13, 33])
    parser.add_argument("--window", type=float, nargs=2, default=[-2.0, 3.0])
    parser.add_argument("--bin-size", type=float, default=0.05)
    parser.add_argument("--smooth-sigma-bins", type=float, default=1.0)
    parser.add_argument("--align", choices=ALIGNMENTS, default="odor")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "figures")
    return parser.parse_args()


def make_figure(events, metadata, neuron: int, args: argparse.Namespace) -> None:
    alignment_column, alignment_label = ALIGNMENTS[args.align]
    lo, hi = args.window
    edges = np.arange(lo, hi + args.bin_size * .5, args.bin_size)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, (ax_raster, ax_psth) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]}, constrained_layout=True,
    )
    row = 0
    for condition in LABELS:
        trials = metadata[metadata.trial_type.eq(condition)]
        condition_counts = []
        for trial in trials.itertuples(index=False):
            spike_times = events.loc[
                (events.neuron_id.eq(neuron)) & (events.trial_number.eq(trial.trial_number)),
                "spike_time",
            ].to_numpy() - getattr(trial, alignment_column)
            visible = spike_times[(spike_times >= lo) & (spike_times <= hi)]
            ax_raster.vlines(visible, row + .1, row + .9, color=COLORS[condition], linewidth=.7)
            condition_counts.append(np.histogram(visible, bins=edges)[0])
            row += 1
        if condition_counts:
            rate = np.mean(condition_counts, axis=0) / args.bin_size
            rate = gaussian_filter1d(rate, args.smooth_sigma_bins)
            ax_psth.plot(centers, rate, color=COLORS[condition], lw=2, label=LABELS[condition])
        ax_raster.axhline(row - .5, color="#d9d8d2", lw=.6)

    for axis in (ax_raster, ax_psth):
        axis.axvline(0, color="#111", linestyle="--", lw=1)
        axis.spines[["top", "right"]].set_visible(False)
    ax_raster.set_ylabel("Trials, grouped by condition")
    ax_raster.set_title(f"Neuron {neuron} · {args.align}-aligned spiking")
    ax_psth.set_xlabel(f"Time from {alignment_label} (s)")
    ax_psth.set_ylabel("Rate (Hz)")
    ax_psth.legend(frameon=False, ncol=2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"neuron_{neuron}_{args.align}_aligned_raster_psth.png"
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    args = parse_args()
    events = load_spike_events(args.data)
    metadata = trial_metadata(events)
    metadata = metadata[metadata.trial_type.isin(LABELS)].copy()
    for neuron in args.neurons:
        make_figure(events, metadata, neuron, args)


if __name__ == "__main__":
    main()
