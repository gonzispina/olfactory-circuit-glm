#!/usr/bin/env python3
"""Build a held-out stimulus-plus-history reference for every neuron and condition."""

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
CONDITIONS = {
    1: ("OR + CR", "first_odor_inhale"),
    2: ("OR + CU", "first_odor_inhale"),
    3: ("OU + CR", "first_odor_inhale"),
    4: ("OU + CU", "first_odor_inhale"),
    5: ("CR · no odor", "ctx_entry_time"),
    6: ("CU · no odor", "ctx_entry_time"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "population_reference.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_spike_events(args.data)
    neurons = sorted(events["neuron_id"].unique())
    rows = []
    failures = []

    for condition, (label, alignment_event) in CONDITIONS.items():
        condition_events = events[events["trial_type"] == condition].copy()
        n_trials = condition_events["trial_number"].nunique()
        print(f"Condition {condition} ({label}) · {n_trials} trials", flush=True)
        for index, neuron in enumerate(neurons, start=1):
            try:
                data = build_binned_dataset(
                    condition_events,
                    target_neuron=neuron,
                    alignment_event=alignment_event,
                )
                for seed in args.seeds:
                    for result in evaluate_nested_models(data, seed=seed):
                        rows.append({
                            **result.__dict__,
                            "seed": seed,
                            "target_neuron": neuron,
                            "trial_type": condition,
                            "condition": label,
                            "alignment_event": alignment_event,
                        })
            except Exception as error:  # persist incomplete targets instead of hiding them
                failures.append({
                    "target_neuron": neuron,
                    "trial_type": condition,
                    "condition": label,
                    "error": str(error),
                })
            if index % 16 == 0 or index == len(neurons):
                print(f"  {index}/{len(neurons)} neurons complete", flush=True)

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = (output[output["model"] == "stimulus_history"]
               .groupby(["target_neuron", "trial_type", "condition", "alignment_event"], as_index=False)
               .agg(
                   splits=("seed", "nunique"),
                   valid_splits=("held_out_bits_per_spike", lambda x: x.notna().sum()),
                   mean_test_spikes=("n_test_spikes", "mean"),
                   mean_bits_per_spike=("held_out_bits_per_spike", "mean"),
                   median_bits_per_spike=("held_out_bits_per_spike", "median"),
                   q05_bits_per_spike=("held_out_bits_per_spike", lambda x: x.quantile(.05)),
                   q95_bits_per_spike=("held_out_bits_per_spike", lambda x: x.quantile(.95)),
               ))
    summary_path = args.output.with_name(f"{args.output.stem}_stimulus_history_summary.csv")
    summary.to_csv(summary_path, index=False)
    failures_path = args.output.with_name(f"{args.output.stem}_failures.csv")
    pd.DataFrame(
        failures, columns=["target_neuron", "trial_type", "condition", "error"]
    ).to_csv(failures_path, index=False)
    print(f"Saved: {args.output}")
    print(f"Saved: {summary_path}")
    print(f"Failures: {len(failures)} · {failures_path}")


if __name__ == "__main__":
    main()
