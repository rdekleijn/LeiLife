import numpy as np

from .agents import Agent
from .nn import FeedForwardNet


def select_and_reproduce(evaluated_agents, n_hidden, top_n=20, offspring_per=5, mutation_std=0.3):
    """Truncation selection: take the `top_n` fittest agents (by `.fitness`)
    and produce `offspring_per` Gaussian-mutated offspring from each."""
    sortedagents = sorted(evaluated_agents, key=lambda a: a.fitness, reverse=True)
    n_params = FeedForwardNet(8, n_hidden, 2).n_params
    offspring = []
    for parent in sortedagents[:top_n]:
        for _ in range(offspring_per):
            weights = parent.nnet.params + np.random.normal(0, mutation_std, n_params)
            offspring.append(Agent(nnet=FeedForwardNet(8, n_hidden, 2), weights=weights))
    return offspring, sortedagents
