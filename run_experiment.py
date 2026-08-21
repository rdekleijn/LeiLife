"""Run a config-driven experiment.

    python run_experiment.py configs/example.yaml

Writes to output/<config.name>/: the resolved config + git commit
(config.yaml, for provenance), a per-agent-per-generation log
(generation_log.csv), the summary text logs, and any verbose replays
(best-group / all-groups, depending on the config).
"""

import argparse
import os

from alife.config import ExperimentConfig
from alife.experiment import run_group_evolution


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', help='Path to a YAML experiment config')
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    output_dir = os.path.join('output', config.name)
    config.save_provenance(output_dir)

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
        resume_from=config.resume_from,
        output_dir=output_dir,
    )


if __name__ == '__main__':
    main()
