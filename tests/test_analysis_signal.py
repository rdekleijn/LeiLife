import numpy as np
import pandas as pd
import pytest

from analysis.signal_analysis import (
    signal_food_correlation,
    signal_usage_entropy,
    compare_group_dialects,
)


def make_group_run(signal_tracks_distance=True, n_ticks=50, agent_id=1):
    rng = np.random.default_rng(0)
    dist = rng.uniform(1, 40, n_ticks)
    bearing = rng.uniform(0, 360, n_ticks)
    if signal_tracks_distance:
        # closer food -> stronger signal, plus a little noise
        signal = np.clip(2.0 - dist / 40 * 2.0 + rng.normal(0, 0.05, n_ticks), 0, 2)
    else:
        signal = rng.uniform(0, 2, n_ticks)  # unrelated to food state
    return pd.DataFrame({
        'tick': range(n_ticks),
        'agent_idx': 0,
        'agent_id': agent_id,
        'agent_x': rng.uniform(0, 50, n_ticks),
        'agent_y': rng.uniform(0, 50, n_ticks),
        'orientation': rng.uniform(0, 360, n_ticks),
        'eaten_tokens': 0,
        'signal': signal,
        'nearest_food_dist': dist,
        'nearest_food_bearing': bearing,
    })


def test_signal_food_correlation_detects_distance_tracking_signal():
    df = make_group_run(signal_tracks_distance=True)
    result = signal_food_correlation(df)
    assert result.loc[0, 'corr_with_food_dist'] < -0.8  # strongly negative: closer -> louder


def test_signal_food_correlation_near_zero_for_unrelated_signal():
    # Larger sample than the other tests so the spurious correlation from
    # random noise reliably settles close to zero (law of large numbers).
    df = make_group_run(signal_tracks_distance=False, n_ticks=2000)
    result = signal_food_correlation(df)
    assert abs(result.loc[0, 'corr_with_food_dist']) < 0.1


def test_signal_food_correlation_handles_constant_signal():
    df = make_group_run()
    df['signal'] = 1.0  # zero variance
    result = signal_food_correlation(df)
    assert pd.isna(result.loc[0, 'corr_with_food_dist'])


def test_signal_usage_entropy_low_for_constant_signal():
    df = make_group_run()
    df['signal'] = 1.0
    result = signal_usage_entropy(df)
    assert result.loc[0, 'signal_entropy_bits'] == pytest.approx(0.0)


def test_signal_usage_entropy_higher_for_varied_signal():
    constant_df = make_group_run()
    constant_df['signal'] = 1.0
    varied_df = make_group_run(signal_tracks_distance=False)

    constant_entropy = signal_usage_entropy(constant_df).loc[0, 'signal_entropy_bits']
    varied_entropy = signal_usage_entropy(varied_df).loc[0, 'signal_entropy_bits']

    assert varied_entropy > constant_entropy


def test_compare_group_dialects(tmp_path):
    path_a = tmp_path / "group_a.csv"
    path_b = tmp_path / "group_b.csv"
    make_group_run(signal_tracks_distance=True).to_csv(path_a, index=False)
    make_group_run(signal_tracks_distance=False, agent_id=2).to_csv(path_b, index=False)

    summary = compare_group_dialects([str(path_a), str(path_b)])

    assert len(summary) == 2
    assert set(summary['path']) == {str(path_a), str(path_b)}
