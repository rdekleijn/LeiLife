"""YAML-driven experiment configuration for run_group_evolution.

Replaces hand-editing hyperparameters into a script (see the old
`run % 3` condition branch this superseded) with a config file that gets
saved alongside a run's output for provenance.
"""

import dataclasses
import os
import subprocess
from typing import Optional

import yaml


@dataclasses.dataclass
class ExperimentConfig:
    name: str = "experiment"
    num_agents: int = 100
    num_gens: int = 250
    lifetime: int = 600
    env_size: int = 50
    group_size: int = 5
    kin_based_groups: bool = False
    num_cores: Optional[int] = None
    save_best_every: int = 20
    save_all_groups_every: Optional[int] = None
    n_hidden: int = 8
    top_n: int = 20
    offspring_per: int = 5
    mutation_std: float = 0.3
    signaling: bool = False
    social_learning: bool = False
    seed: Optional[int] = None
    checkpoint_every: Optional[int] = None
    resume_from: Optional[str] = None

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config key(s) in {path}: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self):
        return dataclasses.asdict(self)

    def save_provenance(self, output_dir):
        """Writes the resolved config plus the current git commit (if this
        is a git checkout) into `output_dir`, so a run's output directory
        is self-describing after the fact."""
        os.makedirs(output_dir, exist_ok=True)
        record = self.to_dict()
        record['git_commit'] = _git_commit()
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.safe_dump(record, f, sort_keys=False)


def _git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).decode().strip()
    except Exception:
        return None
