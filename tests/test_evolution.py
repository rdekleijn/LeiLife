import numpy as np

from alife.agents import Agent
from alife.evolution import select_and_reproduce
from alife.nn import FeedForwardNet


def make_agent(fitness, n_hidden=4):
    net = FeedForwardNet(8, n_hidden, 2)
    agent = Agent(nnet=net)
    agent.fitness = fitness
    return agent


def test_select_and_reproduce_counts_and_ordering():
    np.random.seed(0)
    agents = [make_agent(fitness=f) for f in [3, 1, 5, 2, 4]]

    offspring, sortedagents = select_and_reproduce(
        agents, n_hidden=4, top_n=2, offspring_per=3, mutation_std=0.1,
    )

    assert [a.fitness for a in sortedagents] == [5, 4, 3, 2, 1]
    assert len(offspring) == 2 * 3
    for child in offspring:
        assert child.nnet.n_params == FeedForwardNet(8, 4, 2).n_params


def test_offspring_weights_are_mutated_from_parent():
    np.random.seed(0)
    agents = [make_agent(fitness=1, n_hidden=4)]
    offspring, _ = select_and_reproduce(agents, n_hidden=4, top_n=1, offspring_per=1, mutation_std=1.0)
    assert not np.array_equal(offspring[0].nnet.params, agents[0].nnet.params)
