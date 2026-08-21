import numpy as np

from alife.agents import Agent
from alife.evolution import select_and_reproduce, form_groups
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


def test_offspring_lineage_id_matches_parent_rank():
    np.random.seed(0)
    agents = [make_agent(fitness=f) for f in [3, 1, 5, 2, 4]]
    offspring, _ = select_and_reproduce(agents, n_hidden=4, top_n=2, offspring_per=3, mutation_std=0.1)
    # sortedagents ranks are [fitness 5 -> family 0, fitness 4 -> family 1]
    assert [c.lineage_id for c in offspring] == [0, 0, 0, 1, 1, 1]


def test_form_groups_kin_based_keeps_siblings_together():
    agents = []
    for family in range(2):
        for _ in range(3):
            a = Agent()
            a.lineage_id = family
            agents.append(a)
    np.random.shuffle(agents)  # order shouldn't matter for kin-based grouping

    groups = form_groups(agents, group_size=3, kin_based=True)

    assert len(groups) == 2
    for group in groups:
        assert len({a.lineage_id for a in group}) == 1


def test_form_groups_random_uses_all_agents_once():
    agents = [Agent() for _ in range(11)]
    groups = form_groups(agents, group_size=4, kin_based=False)
    assert [len(g) for g in groups] == [4, 4, 3]
    flattened = [a for g in groups for a in g]
    assert len(flattened) == len(agents)
    assert set(id(a) for a in flattened) == set(id(a) for a in agents)


def test_form_groups_kin_based_handles_missing_lineage_id():
    agents = [Agent() for _ in range(4)]  # no lineage_id set (founding generation)
    groups = form_groups(agents, group_size=2, kin_based=True)
    assert [len(g) for g in groups] == [2, 2]
