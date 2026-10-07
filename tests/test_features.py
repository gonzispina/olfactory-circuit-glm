import numpy as np
import pandas as pd

from src.features import build_binned_dataset


def test_design_uses_trial_relative_event_alignment_and_past_history_only():
    events = pd.DataFrame(
        {
            "neuron_id": [1, 1, 2],
            "spike_time": [10.0, 10.21, 10.11],
            "trial_number": [7, 7, 7],
            "ctx_entry_time": [10.0, 10.0, 10.0],
            "ctx_exit_time": [10.4, 10.4, 10.4],
            "first_odor_inhale": [10.0, 10.0, 10.0],
            "trial_type": [1, 1, 1],
            "trial_start_time": [0.0, 0.0, 0.0],
        }
    )
    data = build_binned_dataset(
        events,
        target_neuron=1,
        coupling_neurons=[2],
        window=(0.0, 0.3),
        bin_size=0.1,
        n_history_lags=1,
        n_stimulus_lags=1,
    )

    # Target spikes at t=0 and t=0.21 after odor alignment.
    assert data.y.tolist() == [1.0, 0.0, 1.0]
    # The first history bin cannot include the target's current-bin spike.
    assert data.X_history[:, 0].tolist() == [0.0, 1.0, 0.0]
    # Coupling history is likewise strictly past-only.
    assert data.X_coupling[:, 0].tolist() == [0.0, 0.0, 1.0]
    # Type 1 is OR + CR: both event impulses occur at t=0; all other
    # identity-specific event bases remain zero.
    names = data.feature_names["stimulus"]
    assert data.X_stimulus[:, names.index("OR_lag_0")].tolist() == [1.0, 0.0, 0.0]
    assert data.X_stimulus[:, names.index("CR_lag_0")].tolist() == [1.0, 0.0, 0.0]
    assert data.X_stimulus[:, names.index("OU_lag_0")].tolist() == [0.0, 0.0, 0.0]
    assert data.X_stimulus[:, names.index("CU_lag_0")].tolist() == [0.0, 0.0, 0.0]
    assert data.X_stimulus[:, names.index("CER_lag_0")].tolist() == [0.0, 0.0, 0.0]
    assert data.X_stimulus[:, names.index("CEU_lag_0")].tolist() == [0.0, 0.0, 0.0]
