"""Signal-content analysis over a saved verbose group-run CSV (see
alife.viz.GroupRunRecorder.to_csv -- produced by experiment.run_group's
verbose=True path, e.g. the files run_group_evolution writes under
save_best_every / save_all_groups_every).

Typical usage:

    from analysis.signal_analysis import load_group_run, signal_food_correlation
    df = load_group_run('output/my_run/best_group_gen100_group_run.csv')
    print(signal_food_correlation(df))
"""

import glob

import numpy as np
import pandas as pd


def load_group_run(path):
    return pd.read_csv(path)


def signal_food_correlation(df):
    """Per-agent Pearson correlation between emitted signal strength and
    (a) distance to the nearest food token and (b) how far off-center that
    food's bearing is (0 = directly ahead, 180 = directly behind -- see
    geometry.calc_dir_and_dist's convention). A meaningfully evolved
    "I'm near food" or "food is this way" signal should show up as a
    nonzero correlation here; near-zero suggests the signal isn't tracking
    food state (for this agent, in this run) -- it's a cheap first pass,
    not a substitute for a real mutual-information estimate."""
    rows = []
    for agent_id, g in df.groupby('agent_id'):
        g = g.dropna(subset=['nearest_food_dist', 'nearest_food_bearing'])
        if len(g) < 2 or g['signal'].std() == 0:
            rows.append({'agent_id': agent_id, 'corr_with_food_dist': np.nan, 'corr_with_food_bearing': np.nan})
            continue
        # circular distance from "directly ahead" (bearing 0), not linear
        # distance from the bearing value itself -- 350 degrees is nearly
        # dead ahead, not nearly "off-center".
        offcenter = np.minimum(g['nearest_food_bearing'], 360 - g['nearest_food_bearing'])
        rows.append({
            'agent_id': agent_id,
            'corr_with_food_dist': g['signal'].corr(g['nearest_food_dist']),
            'corr_with_food_bearing': g['signal'].corr(offcenter),
        })
    return pd.DataFrame(rows)


def signal_usage_entropy(df, n_bins=10):
    """Per-agent Shannon entropy (in bits) of the emitted signal's
    histogram over its observed range -- a simple, discretized measure of
    "how varied is this agent's signaling," not a rigorous continuous
    mutual-information estimate. Near 0 means the agent emits an
    (almost) constant signal; higher means more varied usage. Combine with
    signal_food_correlation to say something about *meaning*, not just
    variety."""
    rows = []
    for agent_id, g in df.groupby('agent_id'):
        counts, _ = np.histogram(g['signal'], bins=n_bins)
        probs = counts[counts > 0] / counts.sum()
        entropy = float(-(probs * np.log2(probs)).sum())
        rows.append({'agent_id': agent_id, 'signal_entropy_bits': entropy})
    return pd.DataFrame(rows)


def compare_group_dialects(paths):
    """Summarizes signal usage across multiple saved group-run CSVs (e.g.
    every group from one generation, via save_all_groups_every), to check
    whether different groups converge on measurably different signaling
    conventions ("dialects") despite facing the same task -- a concrete,
    checkable signature of cultural (group-specific, not purely genetic)
    variation."""
    rows = []
    for path in paths:
        df = load_group_run(path)
        corr = signal_food_correlation(df)
        entropy = signal_usage_entropy(df)
        rows.append({
            'path': path,
            'mean_signal': df['signal'].mean(),
            'mean_corr_with_food_bearing': corr['corr_with_food_bearing'].mean(),
            'mean_signal_entropy_bits': entropy['signal_entropy_bits'].mean(),
        })
    return pd.DataFrame(rows)


def find_group_runs(output_dir, gen):
    """Convenience glob for the files save_all_groups_every writes for one
    generation: output_dir/gen{gen}_group*_group_run.csv."""
    return sorted(glob.glob(f'{output_dir}/gen{gen}_group*_group_run.csv'))
