"""Population checkpointing, so a long run can be interrupted and resumed
without starting evolution over from generation 0.

A checkpoint is the population about to be evaluated in the *next*
generation (i.e. saved as `offspring` at the end of a generation in
run_group_evolution), plus that point's RNG state for best-effort
reproducibility across a resume (see experiment.run_group_evolution's
`seed` docs for the same caveat: full determinism across a resume also
requires num_cores=1, since parallel workers reseed themselves
independently of the parent process's RNG state).
"""

import os
import pickle
import random

import numpy as np


def save_checkpoint(path, gen, agents):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({
            'gen': gen,
            'agents': agents,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(),
        }, f)


def load_checkpoint(path):
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    random.setstate(checkpoint['random_state'])
    np.random.set_state(checkpoint['np_random_state'])
    return checkpoint
