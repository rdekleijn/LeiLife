"""Compare two run_experiment.py runs that differ only in kin_based_groups,
to see whether group structure produces different evolved + culturally
learned signaling conventions -- the central question this facelift set
out to let us investigate (see docs/facelift-design.md).

    python run_experiment.py configs/kin_grouping.yaml
    python run_experiment.py configs/mixed_grouping.yaml
    python examples/compare_kin_vs_mixed.py output/kin_grouping output/mixed_grouping

Run the two experiments first (see examples/README.md for expected
runtime); this script only reads their output, it doesn't simulate
anything itself.
"""

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.plots import load_generation_log, fitness_summary_by_gen  # noqa: E402
from analysis.signal_analysis import compare_group_dialects, find_group_runs  # noqa: E402

GROUP_RUN_RE = re.compile(r'gen(\d+)_group\d+_group_run\.csv$')


def summarize_fitness(output_dir, label):
    df = load_generation_log(os.path.join(output_dir, 'generation_log.csv'))
    summary = fitness_summary_by_gen(df)
    last = summary.iloc[-1]
    print(f"\n=== {label} ({output_dir}) ===")
    print(f"Generations logged: {summary['gen'].min()}-{summary['gen'].max()}")
    print(f"Final generation ({int(last['gen'])}) fitness -- "
          f"min: {last['min']:.4f}  mean: {last['mean']:.4f}  max: {last['max']:.4f}")
    return summary


def saved_group_run_generations(output_dir):
    """Which generations have a full set of per-group replays saved (see
    save_all_groups_every), sorted ascending."""
    gens = set()
    for path in glob.glob(os.path.join(output_dir, 'gen*_group*_group_run.csv')):
        m = GROUP_RUN_RE.search(os.path.basename(path))
        if m:
            gens.add(int(m.group(1)))
    return sorted(gens)


def dialect_report(output_dir, label):
    """Compares this condition's own groups against each other at its
    latest saved generation, to see how much groups within one condition
    already diverge -- the baseline to compare the kin-vs-mixed
    *between-condition* difference against."""
    gens = saved_group_run_generations(output_dir)
    if not gens:
        print(f"\n{label}: no save_all_groups_every replays found in {output_dir} "
              "(re-run with that option set to compare dialects)")
        return None
    latest_gen = gens[-1]
    paths = find_group_runs(output_dir, latest_gen)
    dialects = compare_group_dialects(paths)
    print(f"\n{label}: {len(paths)} groups' dialects at generation {latest_gen}")
    print(dialects.drop(columns=['path']).to_string(index=False))
    return dialects


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('kin_output_dir', help="e.g. output/kin_grouping")
    parser.add_argument('mixed_output_dir', help="e.g. output/mixed_grouping")
    parser.add_argument('--fitness-plot', default=None,
                         help="Optional path to save a kin-vs-mixed fitness comparison plot")
    args = parser.parse_args()

    kin_summary = summarize_fitness(args.kin_output_dir, "Kin-based grouping")
    mixed_summary = summarize_fitness(args.mixed_output_dir, "Mixed grouping")

    dialect_report(args.kin_output_dir, "Kin-based grouping")
    dialect_report(args.mixed_output_dir, "Mixed grouping")

    if args.fitness_plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(kin_summary['gen'], kin_summary['mean'], color='tab:blue', label='kin-based')
        ax.plot(mixed_summary['gen'], mixed_summary['mean'], color='tab:orange', label='mixed')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Mean fitness')
        ax.set_title('Kin-based vs. mixed grouping')
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.fitness_plot)
        plt.close(fig)
        print(f"\nSaved fitness comparison plot to {args.fitness_plot}")


if __name__ == '__main__':
    main()
