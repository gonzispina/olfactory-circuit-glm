"""Loading and validation for the headerless spike-event matrix."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


COLUMNS = [
    "neuron_id",
    "spike_time",
    "trial_number",
    "ctx_entry_time",
    "ctx_exit_time",
    "first_odor_inhale",
    "trial_type",
    "trial_start_time",
]


def load_spike_events(path: str | Path) -> pd.DataFrame:
    """Load the eight-column matrix without losing its first spike record.

    The original CSV has no header. Passing ``header=None`` is therefore part
    of the analysis contract, rather than a cosmetic choice.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Spike matrix not found: {path}")

    events = pd.read_csv(path, header=None, names=COLUMNS)
    if events.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Expected {len(COLUMNS)} columns in {path.name}; got {events.shape[1]}."
        )
    # Trial types 5 and 6 are no-odor controls in the original notebook, so
    # ``first_odor_inhale`` is structurally missing for them. It is preserved
    # as NaN and handled as an absent event in the design matrix. All remaining
    # fields are required to identify and align a trial.
    required = [column for column in COLUMNS if column != "first_odor_inhale"]
    if events[required].isna().any().any():
        missing = events[required].columns[events[required].isna().any()].tolist()
        raise ValueError(f"Missing values in required columns: {missing}")

    numeric = COLUMNS
    events[numeric] = events[numeric].apply(pd.to_numeric, errors="raise")
    events["trial_number"] = events["trial_number"].astype(int)
    events["neuron_id"] = events["neuron_id"].astype(int)
    events["trial_type"] = events["trial_type"].astype(int)

    # All downstream timing is relative to the trial start, so the analysis
    # never silently mixes absolute session clock and within-trial time.
    for column in ("spike_time", "ctx_entry_time", "ctx_exit_time", "first_odor_inhale"):
        events[column] = events[column] - events["trial_start_time"]

    validate_trial_metadata(events)
    return events


def validate_trial_metadata(events: pd.DataFrame) -> None:
    """Check that each trial has one coherent set of event labels/times."""
    metadata = ["ctx_entry_time", "ctx_exit_time", "first_odor_inhale", "trial_type"]
    inconsistent = events.groupby("trial_number")[metadata].nunique(dropna=False).gt(1).any(axis=1)
    if inconsistent.any():
        examples = inconsistent[inconsistent].index[:5].tolist()
        raise ValueError(f"Inconsistent metadata inside trials: {examples}")


def trial_metadata(events: pd.DataFrame) -> pd.DataFrame:
    """Return exactly one row per trial."""
    columns = ["trial_number", "ctx_entry_time", "ctx_exit_time", "first_odor_inhale", "trial_type"]
    return events[columns].drop_duplicates("trial_number").sort_values("trial_number").reset_index(drop=True)
