"""Structured per-agent-per-generation logging.

Complements Experiment.write_log's population-summary text file (mean/min/
max fitness) with one row per agent per generation, so a run's full
evolutionary and social history can be reloaded and analyzed afterwards
without re-running anything -- see analysis/plots.py.
"""

import os

import numpy as np
import pandas as pd

AGENT_LOG_COLUMNS = [
    'gen', 'agent_id', 'parent_id', 'lineage_id', 'group_id',
    'fitness', 'eaten_tokens', 'wheel_distance', 'signal_emitted',
    'plastic_weight_norm',
]


def agent_log_rows(gen, agents):
    rows = []
    for a in agents:
        rows.append({
            'gen': gen,
            'agent_id': a.agent_id,
            'parent_id': a.parent_id,
            'lineage_id': a.lineage_id,
            'group_id': a.group_id,
            'fitness': a.fitness,
            'eaten_tokens': a.eatenTokens,
            'wheel_distance': a.wheelDistanceTraveled,
            'signal_emitted': a.signalEmitted if a.signaling else None,
            'plastic_weight_norm': float(np.linalg.norm(a.W_plastic)) if a.social_learning else None,
        })
    return rows


class GenerationLog:
    """Appends one row per agent per generation to a single CSV."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self._wrote_header = os.path.exists(path)

    def append(self, gen, agents):
        df = pd.DataFrame(agent_log_rows(gen, agents), columns=AGENT_LOG_COLUMNS)
        df.to_csv(self.path, mode='a', header=not self._wrote_header, index=False)
        self._wrote_header = True

    def load(self):
        return pd.read_csv(self.path)
