"""Causal, trial-level design matrices for binned spike-count GLMs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .data import trial_metadata


# OR/OU identify the odor used in a trial; CR/CU identify the visual context.
# Types 5 and 6 are the no-odor controls recorded by the original experiment.
TRIAL_STIMULI = {
    1: ("OR", "CR"),
    2: ("OR", "CU"),
    3: ("OU", "CR"),
    4: ("OU", "CU"),
    5: (None, "CR"),
    6: (None, "CU"),
}
STIMULUS_NAMES = ("OR", "OU", "CR", "CU", "CER", "CEU")


@dataclass(frozen=True)
class BinnedDataset:
    X_stimulus: np.ndarray
    X_history: np.ndarray
    X_coupling: np.ndarray
    y: np.ndarray
    trial_ids: np.ndarray
    trial_types: np.ndarray
    feature_names: dict[str, list[str]]


def _lagged_counts(counts: np.ndarray, n_lags: int, prefix: str) -> tuple[np.ndarray, list[str]]:
    """Return strictly past bins only; no current/future-bin information leaks in."""
    n = len(counts)
    matrix = np.zeros((n, n_lags), dtype=float)
    for lag in range(1, n_lags + 1):
        matrix[lag:, lag - 1] = counts[:-lag]
    return matrix, [f"{prefix}_lag_{lag}" for lag in range(1, n_lags + 1)]


def _square_stimulus(start: float, end: float, edges: np.ndarray) -> np.ndarray:
    """Return one binary time-series column active from ``start`` to ``end``.

    A bin represents its center point. This avoids treating an event impulse as
    if it occupied a full bin on either side of its timestamp.
    """
    centers = (edges[:-1] + edges[1:]) / 2
    return ((centers >= start) & (centers < end)).astype(float)[:, None]


def _event_basis(event_time: float, edges: np.ndarray, n_lags: int, name: str) -> tuple[np.ndarray, list[str]]:
    """A one-bin event impulse followed by a causal lag basis."""
    n_bins = len(edges) - 1
    matrix = np.zeros((n_bins, n_lags), dtype=float)
    event_bin = np.searchsorted(edges, event_time, side="right") - 1
    if 0 <= event_bin < n_bins:
        for lag in range(n_lags):
            row = event_bin + lag
            if row < n_bins:
                matrix[row, lag] = 1.0
    return matrix, [f"{name}_lag_{lag}" for lag in range(n_lags)]


def _trial_counts(
    events: pd.DataFrame, trial_id: int, neuron_id: int, edges: np.ndarray, *, origin: float
) -> np.ndarray:
    spikes = events.loc[
        (events["trial_number"] == trial_id) & (events["neuron_id"] == neuron_id), "spike_time"
    ].to_numpy()
    return np.histogram(spikes - origin, bins=edges)[0].astype(float)


def build_binned_dataset(
    events: pd.DataFrame,
    *,
    target_neuron: int,
    coupling_neurons: Iterable[int] = (),
    window: tuple[float, float] = (-2.0, 3.0),
    bin_size: float = 0.05,
    n_history_lags: int = 30,
    n_stimulus_lags: int = 25,
    alignment_event: str = "first_odor_inhale",
    stimulus_encoding: str = "identity",
) -> BinnedDataset:
    """Bin all trials and build causal stimulus, history and coupling blocks.

    In the default ``identity`` encoding, each stimulus identity is represented by a one-bin event impulse and 25
    causal 50-ms lags (1.25 s). OR/OU occur at first odor inhalation, CR/CU at
    context entry and CER/CEU at context exit. ``generic`` instead uses odor,
    context-entry and context-exit kernels shared across identities. The time
    origin affects only bin coordinates; it does not alter physical event time.
    """
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")
    edges = np.arange(window[0], window[1] + bin_size * 0.5, bin_size)
    metadata = trial_metadata(events)
    coupling_neurons = tuple(coupling_neurons)
    if stimulus_encoding not in {"identity", "generic"}:
        raise ValueError("stimulus_encoding must be 'identity' or 'generic'.")
    if alignment_event not in {"ctx_entry_time", "first_odor_inhale", "ctx_exit_time"}:
        raise ValueError(f"Unknown alignment event: {alignment_event}")
    if metadata[alignment_event].isna().any():
        absent = metadata.loc[metadata[alignment_event].isna(), "trial_number"].head(5).tolist()
        raise ValueError(
            f"{alignment_event} is absent in selected trials (examples: {absent}). "
            "Choose an observed alignment event or exclude those trials."
        )

    stimulus_rows: list[np.ndarray] = []
    history_rows: list[np.ndarray] = []
    coupling_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    trial_id_rows: list[np.ndarray] = []
    trial_type_rows: list[np.ndarray] = []
    stimulus_names: list[str] | None = None
    history_names: list[str] | None = None
    coupling_names: list[str] = []

    for trial in metadata.itertuples(index=False):
        origin = getattr(trial, alignment_event)
        y_trial = _trial_counts(events, trial.trial_number, target_neuron, edges, origin=origin)
        odor, context = TRIAL_STIMULI[trial.trial_type]
        exit_name = "CER" if context == "CR" else "CEU"
        if stimulus_encoding == "identity":
            event_specs = [
                (name, trial.first_odor_inhale - origin) if name == odor and odor is not None
                else (name, trial.ctx_entry_time - origin) if name == context
                else (name, trial.ctx_exit_time - origin) if name == exit_name
                else (name, None)
                for name in STIMULUS_NAMES
            ]
        else:
            event_specs = [
                ("odor", trial.first_odor_inhale - origin if odor is not None else None),
                ("context_entry", trial.ctx_entry_time - origin),
                ("context_exit", trial.ctx_exit_time - origin),
            ]
        stim_blocks, names = [], []
        for name, event_time in event_specs:
            if event_time is not None:
                block, block_names = _event_basis(event_time, edges, n_stimulus_lags, name)
            else:
                block = np.zeros((len(y_trial), n_stimulus_lags), dtype=float)
                block_names = [f"{name}_lag_{lag}" for lag in range(n_stimulus_lags)]
            stim_blocks.append(block)
            names.extend(block_names)
        x_stim = np.hstack(stim_blocks)
        x_history, h_names = _lagged_counts(y_trial, n_history_lags, "self")

        coupling_blocks = []
        trial_coupling_names = []
        for neuron in coupling_neurons:
            source = _trial_counts(events, trial.trial_number, neuron, edges, origin=origin)
            block, block_names = _lagged_counts(source, n_history_lags, f"neuron_{neuron}")
            coupling_blocks.append(block)
            trial_coupling_names.extend(block_names)
        x_coupling = (
            np.hstack(coupling_blocks)
            if coupling_blocks
            else np.empty((len(y_trial), 0), dtype=float)
        )

        stimulus_rows.append(x_stim)
        history_rows.append(x_history)
        coupling_rows.append(x_coupling)
        target_rows.append(y_trial)
        trial_id_rows.append(np.full(len(y_trial), trial.trial_number, dtype=int))
        trial_type_rows.append(np.full(len(y_trial), trial.trial_type, dtype=int))
        stimulus_names = names
        history_names = h_names
        coupling_names = trial_coupling_names

    return BinnedDataset(
        X_stimulus=np.vstack(stimulus_rows),
        X_history=np.vstack(history_rows),
        X_coupling=np.vstack(coupling_rows),
        y=np.concatenate(target_rows),
        trial_ids=np.concatenate(trial_id_rows),
        trial_types=np.concatenate(trial_type_rows),
        feature_names={
            "stimulus": stimulus_names or [],
            "history": history_names or [],
            "coupling": coupling_names,
        },
    )
