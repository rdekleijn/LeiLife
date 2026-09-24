"""Rerun the kin_grouping/mixed_grouping experiment (configs/kin_grouping.yaml,
configs/mixed_grouping.yaml) across additional seeds, to check whether a
kin-vs-mixed pattern seen in one seed is a robust property of this setup
or specific to that one evolutionary trajectory.

Writes to output/{kin,mixed}_grouping_seed<N>/ -- does not touch the
original output/kin_grouping or output/mixed_grouping directories.

    python examples/replicate_seeds.py 1 2 3
    python examples/replicate_seeds.py 1 2 3 --save-mp4   # also render replay videos

mp4 rendering is off by default here: it's ~75x slower than the
simulation itself (see run_group's save_mp4 docs), and a seed-replication
batch is about statistics, not something you sit and watch -- the CSVs
(everything analysis/ reads) are written either way. Turn it on with
--save-mp4 if you specifically want to eyeball a replay from one of
these seeds.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alife.config import ExperimentConfig  # noqa: E402
from alife.experiment import run_group_evolution  # noqa: E402

CONFIGS = [
    ('configs/kin_grouping.yaml', 'kin_grouping'),
    ('configs/mixed_grouping.yaml', 'mixed_grouping'),
]


def run_one(config_path, condition_name, seed, save_mp4):
    config = ExperimentConfig.from_yaml(config_path)
    config.seed = seed
    config.name = f'{condition_name}_seed{seed}'
    config.save_mp4 = save_mp4
    output_dir = os.path.join('output', config.name)
    config.save_provenance(output_dir)

    print(f'--- Running {config.name} (save_mp4={save_mp4}) ---', flush=True)
    run_group_evolution(
        num_agents=config.num_agents,
        num_gens=config.num_gens,
        lifetime=config.lifetime,
        env_size=config.env_size,
        group_size=config.group_size,
        kin_based_groups=config.kin_based_groups,
        num_cores=config.num_cores,
        save_best_every=config.save_best_every,
        save_all_groups_every=config.save_all_groups_every,
        n_hidden=config.n_hidden,
        top_n=config.top_n,
        offspring_per=config.offspring_per,
        mutation_std=config.mutation_std,
        signaling=config.signaling,
        social_learning=config.social_learning,
        seed=config.seed,
        checkpoint_every=config.checkpoint_every,
        resume_from=None,
        output_dir=output_dir,
        save_mp4=config.save_mp4,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('seeds', nargs='*', type=int, default=[1, 2],
                         help="Seeds to run (default: 1 2)")
    parser.add_argument('--save-mp4', action='store_true',
                         help="Also render replay mp4s (slow -- see module docstring)")
    args = parser.parse_args()

    for seed in args.seeds:
        for config_path, condition_name in CONFIGS:
            run_one(config_path, condition_name, seed, save_mp4=args.save_mp4)


if __name__ == '__main__':
    main()
