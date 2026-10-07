#!/usr/bin/env python3
"""Measure held-out contribution of each event-identity block for neuron 52."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_spike_events
from src.features import STIMULUS_NAMES, build_binned_dataset
from src.model import evaluate_nested_models


DATA = ROOT.parent / "neuroscience" / "MatrizFinal_8col.csv"
OUTPUT = ROOT / "results" / "neuron52_event_identity_ablations.csv"
N_LAGS = 25


def remove_block(dataset, removed: str):
    keep = np.ones(dataset.X_stimulus.shape[1], dtype=bool)
    block = STIMULUS_NAMES.index(removed)
    keep[block * N_LAGS:(block + 1) * N_LAGS] = False
    return replace(
        dataset,
        X_stimulus=dataset.X_stimulus[:, keep],
        feature_names={
            **dataset.feature_names,
            "stimulus": [name for name, include in zip(dataset.feature_names["stimulus"], keep) if include],
        },
    )


def main() -> None:
    events = load_spike_events(DATA)
    dataset = build_binned_dataset(
        events[events["trial_type"].isin([1, 2, 3, 4])],
        target_neuron=52,
        coupling_neurons=(),
        stimulus_encoding="identity",
    )
    variants = {"full": dataset}
    variants.update({f"without_{name}": remove_block(dataset, name) for name in STIMULUS_NAMES})

    rows = []
    for variant, variant_data in variants.items():
        for seed in range(20):
            result = next(item for item in evaluate_nested_models(variant_data, seed=seed) if item.model == "stimulus")
            rows.append({"variant": variant, "seed": seed, **result.__dict__})
    output = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)

    full = output.loc[output.variant == "full", ["seed", "held_out_bits_per_spike"]].set_index("seed")
    summary = []
    for name in STIMULUS_NAMES:
        ablated = output.loc[output.variant == f"without_{name}", ["seed", "held_out_bits_per_spike"]].set_index("seed")
        delta = full.held_out_bits_per_spike - ablated.held_out_bits_per_spike
        summary.append({
            "event_block": name,
            "full_minus_ablated_bits_per_spike": delta.mean(),
            "q05": delta.quantile(.05),
            "q95": delta.quantile(.95),
            "positive_splits": int((delta > 0).sum()),
        })
    report = pd.DataFrame(summary)
    report.to_csv(OUTPUT.with_name(f"{OUTPUT.stem}_summary.csv"), index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
