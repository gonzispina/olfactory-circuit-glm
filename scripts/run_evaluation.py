#!/usr/bin/env python3
"""Run a first held-out Poisson-GLM comparison for one target neuron."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_spike_events
from src.features import build_binned_dataset
from src.model import evaluate_nested_models


DEFAULT_DATA = ROOT.parent / "neuroscience" / "MatrizFinal_8col.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--target-neuron", type=int, default=45)
    parser.add_argument("--coupling-neurons", type=int, nargs="*", default=[52])
    parser.add_argument("--trial-types", type=int, nargs="*", default=[1])
    parser.add_argument(
        "--alignment-event",
        choices=["ctx_entry_time", "first_odor_inhale", "ctx_exit_time"],
        default="first_odor_inhale",
    )
    parser.add_argument("--stimulus-encoding", choices=["identity", "generic"], default="identity")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(20)),
        help="Independent trial-level outer splits used to quantify variability.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "held_out_metrics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_spike_events(args.data)
    events = events[events["trial_type"].isin(args.trial_types)].copy()
    if events.empty:
        raise ValueError(f"No events found for trial types {args.trial_types}.")

    dataset = build_binned_dataset(
        events,
        target_neuron=args.target_neuron,
        coupling_neurons=args.coupling_neurons,
        alignment_event=args.alignment_event,
        stimulus_encoding=args.stimulus_encoding,
    )
    rows = []
    for seed in args.seeds:
        for result in evaluate_nested_models(dataset, seed=seed):
            rows.append({**result.__dict__, "seed": seed})
    output = pd.DataFrame(rows)
    output["target_neuron"] = args.target_neuron
    output["coupling_neurons"] = ",".join(map(str, args.coupling_neurons))
    output["trial_types"] = ",".join(map(str, args.trial_types))
    output["alignment_event"] = args.alignment_event
    output["stimulus_encoding"] = args.stimulus_encoding
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = (output.groupby("model", as_index=False)
               .agg(
                   splits=("seed", "nunique"),
                   mean_bits_per_spike=("held_out_bits_per_spike", "mean"),
                   median_bits_per_spike=("held_out_bits_per_spike", "median"),
                   q05_bits_per_spike=("held_out_bits_per_spike", lambda x: x.quantile(.05)),
                   q95_bits_per_spike=("held_out_bits_per_spike", lambda x: x.quantile(.95)),
                   selected_alpha_modes=("selected_alpha", lambda x: x.dropna().mode().tolist()),
               ))
    summary_path = args.output.with_name(f"{args.output.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.output}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
