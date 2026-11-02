import pandas as pd
import pytest

from analysis.plots import (
    fitness_summary_by_gen,
    build_lineage_table,
    ancestors_of,
    plot_fitness_over_generations,
    plot_signal_usage_over_generations,
)


def make_log():
    return pd.DataFrame([
        {'gen': 0, 'agent_id': 1, 'parent_id': None, 'lineage_id': 0, 'fitness': 1.0, 'signal_emitted': None},
        {'gen': 0, 'agent_id': 2, 'parent_id': None, 'lineage_id': 0, 'fitness': 3.0, 'signal_emitted': None},
        {'gen': 1, 'agent_id': 3, 'parent_id': 2, 'lineage_id': 0, 'fitness': 5.0, 'signal_emitted': None},
        {'gen': 1, 'agent_id': 4, 'parent_id': 2, 'lineage_id': 0, 'fitness': 2.0, 'signal_emitted': None},
        {'gen': 2, 'agent_id': 5, 'parent_id': 3, 'lineage_id': 0, 'fitness': 7.0, 'signal_emitted': None},
    ])


def test_fitness_summary_by_gen():
    summary = fitness_summary_by_gen(make_log())
    row0 = summary[summary['gen'] == 0].iloc[0]
    assert row0['min'] == 1.0
    assert row0['max'] == 3.0
    assert row0['mean'] == pytest.approx(2.0)


def test_build_lineage_table_and_ancestors_of():
    lineage = build_lineage_table(make_log())
    assert set(lineage['agent_id']) == {1, 2, 3, 4, 5}

    chain = ancestors_of(lineage, 5)
    assert chain == [5, 3, 2]  # 5's parent is 3, 3's parent is 2, 2 has no parent


def test_plot_fitness_over_generations_saves_file(tmp_path):
    out_path = tmp_path / "fitness.png"
    result = plot_fitness_over_generations(make_log(), out_path=str(out_path))
    assert result is None  # returns None when saving to a file
    assert out_path.exists()


def test_plot_signal_usage_over_generations_handles_nonsignaling_data(tmp_path):
    # signal_emitted is all-None (non-signaling run) -- should not raise.
    out_path = tmp_path / "signal.png"
    plot_signal_usage_over_generations(make_log(), out_path=str(out_path))
    assert out_path.exists()
