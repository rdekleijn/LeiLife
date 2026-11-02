"""Reusable analysis/plotting functions over a run's generation_log.csv
(one row per agent per generation -- see alife.logbook.GenerationLog).

Typical usage:

    from analysis.plots import load_generation_log, plot_fitness_over_generations
    df = load_generation_log('output/my_run/generation_log.csv')
    plot_fitness_over_generations(df, out_path='output/my_run/fitness.png')
"""

import pandas as pd


def load_generation_log(path):
    return pd.read_csv(path)


def fitness_summary_by_gen(df):
    """Per-generation min/mean/max fitness."""
    return df.groupby('gen')['fitness'].agg(['min', 'mean', 'max']).reset_index()


def plot_fitness_over_generations(df, out_path=None, title='Fitness over generations'):
    import matplotlib.pyplot as plt

    summary = fitness_summary_by_gen(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary['gen'], summary['mean'], color='tab:blue', label='mean')
    ax.fill_between(summary['gen'], summary['min'], summary['max'], alpha=0.2, color='tab:blue', label='min-max')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
        return None
    return fig


def plot_signal_usage_over_generations(df, out_path=None, title='Signal emission over generations'):
    """Only meaningful for signaling=True runs -- signal_emitted is blank
    (NaN) for non-signaling agents, and .mean() skips those automatically."""
    import matplotlib.pyplot as plt

    summary = df.groupby('gen')['signal_emitted'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary['gen'], summary['signal_emitted'], color='tab:orange')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean cumulative signal emitted per agent (per lifetime)')
    ax.set_title(title)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
        return None
    return fig


def build_lineage_table(df):
    """One row per agent (first generation it appears in), with its
    parent_id -- a queryable ancestry table. Use `ancestors_of` to walk a
    specific agent's lineage back to the founding generation."""
    return (
        df.sort_values('gen')
        .drop_duplicates('agent_id', keep='first')[['agent_id', 'parent_id', 'gen', 'lineage_id', 'fitness']]
        .reset_index(drop=True)
    )


def ancestors_of(lineage_table, agent_id):
    """Walks parent_id back from `agent_id` to the founding generation,
    returning the chain from `agent_id` to its earliest known ancestor."""
    by_id = lineage_table.set_index('agent_id')
    chain = []
    current = agent_id
    while current is not None and current in by_id.index:
        chain.append(current)
        parent = by_id.loc[current, 'parent_id']
        current = None if pd.isna(parent) else int(parent)
    return chain
