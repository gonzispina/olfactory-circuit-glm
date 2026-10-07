#!/usr/bin/env python3
"""Select a CR visual-context lead using nested complete-trial splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_spike_events
from src.features import build_binned_dataset
from src.metrics import bits_per_spike
from src.model import trial_split

DEFAULT_DATA = ROOT.parent / "neuroscience" / "MatrizFinal_8col.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--target-neuron", type=int, default=45)
    parser.add_argument("--trial-types", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--leads-ms", type=int, nargs="+", default=list(range(0, 1001, 50)))
    parser.add_argument("--outer-seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--inner-seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "cr_context_lead_selection.csv")
    return parser.parse_args()


def masks_for_trial_ids(data, train_ids, test_ids):
    return np.isin(data.trial_ids, train_ids), np.isin(data.trial_ids, test_ids)


def fit_score(data, train_ids, test_ids, alpha: float) -> float:
    train_mask, test_mask = masks_for_trial_ids(data, train_ids, test_ids)
    x = np.hstack([data.X_stimulus, data.X_history])
    y_train, y_test = data.y[train_mask], data.y[test_mask]
    baseline = max(float(y_train.mean()), 1e-12)
    model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=1_000))
    model.fit(x[train_mask], y_train)
    return bits_per_spike(y_test, model.predict(x[test_mask]), baseline)


def inner_split(trial_ids: np.ndarray, trial_types: np.ndarray, seed: int):
    ids, first = np.unique(trial_ids, return_index=True)
    labels = trial_types[first]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=.25, random_state=seed)
    train, validation = next(splitter.split(ids, labels))
    return ids[train], ids[validation]


def main() -> None:
    args = parse_args()
    events = load_spike_events(args.data)
    events = events[events.trial_type.isin(args.trial_types)].copy()
    if events.empty:
        raise ValueError("No events matched the requested trial types.")

    leads = [lead / 1000 for lead in args.leads_ms]
    datasets = {
        lead: build_binned_dataset(
            events, target_neuron=args.target_neuron, context_visual_leads={"CR": lead}
        )
        for lead in leads
    }
    reference = datasets[leads[0]]
    rows = []
    for outer_seed in args.outer_seeds:
        outer_train, outer_test = trial_split(
            reference.trial_ids, reference.trial_types, test_size=.2, seed=outer_seed
        )
        lead_scores = {}
        for lead, data in datasets.items():
            scores = []
            outer_train_rows = np.isin(data.trial_ids, outer_train)
            for inner_seed in args.inner_seeds:
                inner_train, inner_validation = inner_split(
                    data.trial_ids[outer_train_rows], data.trial_types[outer_train_rows], inner_seed
                )
                scores.append(fit_score(data, inner_train, inner_validation, args.alpha))
            lead_scores[lead] = float(np.nanmean(scores))
        selected_lead = max(lead_scores, key=lead_scores.get)
        outer_score = fit_score(datasets[selected_lead], outer_train, outer_test, args.alpha)
        for lead, score in lead_scores.items():
            rows.append({
                "outer_seed": outer_seed,
                "cr_lead_ms": int(round(lead * 1000)),
                "inner_mean_bits_per_spike": score,
                "selected": lead == selected_lead,
                "outer_bits_per_spike_if_selected": outer_score if lead == selected_lead else np.nan,
            })

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    selected = output[output.selected]
    print("Selected CR lead per outer split:")
    print(selected[["outer_seed", "cr_lead_ms", "inner_mean_bits_per_spike", "outer_bits_per_spike_if_selected"]].to_string(index=False))
    print("\nSelection frequency:")
    print(selected.cr_lead_ms.value_counts().sort_index().to_string())
    print("\nNested outer performance:")
    print(selected.outer_bits_per_spike_if_selected.describe(percentiles=[.05, .5, .95]).to_string())
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
