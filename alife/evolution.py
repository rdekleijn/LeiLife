import random

import numpy as np

from .agents import Agent, net_dims
from .nn import FeedForwardNet


def select_and_reproduce(evaluated_agents, n_hidden, top_n=20, offspring_per=5, mutation_std=0.3,
                          signaling=False, social_learning=False):
    """Truncation selection: take the `top_n` fittest agents (by `.fitness`)
    and produce `offspring_per` Gaussian-mutated offspring from each.

    Offspring inherit a `lineage_id` equal to their parent's rank
    (0..top_n-1) among this generation's selected parents, so all
    `offspring_per` children of one parent share a lineage_id -- see
    `form_groups(kin_based=True)`. `signaling` must match what the
    evaluated agents were built with (it determines the offspring net's
    input/output size -- see `agents.net_dims`).

    `social_learning` only controls whether offspring get the *capacity*
    for social learning (evolved/heritable); their W_plastic itself always
    starts blank (see Agent.__init__) -- what's actually learned is never
    inherited, only the ability to learn is.
    """
    sortedagents = sorted(evaluated_agents, key=lambda a: a.fitness, reverse=True)
    n_in, n_out = net_dims(signaling)
    n_params = FeedForwardNet(n_in, n_hidden, n_out).n_params
    offspring = []
    for family_id, parent in enumerate(sortedagents[:top_n]):
        for _ in range(offspring_per):
            weights = parent.nnet.params + np.random.normal(0, mutation_std, n_params)
            offspring.append(Agent(nnet=FeedForwardNet(n_in, n_hidden, n_out), weights=weights,
                                    lineage_id=family_id, signaling=signaling,
                                    social_learning=social_learning, parent_id=parent.agent_id))
    return offspring, sortedagents


def form_groups(agents, group_size, kin_based=False):
    """Partition `agents` into groups of up to `group_size` for a
    shared-environment trial (see experiment.run_group).

    kin_based=True sorts by `lineage_id` first so siblings (offspring of
    the same parent -- see `select_and_reproduce`) land in the same group
    wherever possible; this is exact when `group_size` divides
    `offspring_per`, and approximate (groups may straddle a family
    boundary) otherwise. Agents with no lineage_id (e.g. a freshly
    initialized founding generation, which has no shared parentage yet)
    keep their input order, so a founding-generation "kin" grouping is
    really just an index-order grouping.

    kin_based=False shuffles agents randomly before chunking.

    The last group may be smaller than `group_size` if `len(agents)` isn't
    a multiple of it.
    """
    agents = list(agents)
    if kin_based:
        agents.sort(key=lambda a: a.lineage_id if a.lineage_id is not None else -1)
    else:
        random.shuffle(agents)
    return [agents[i:i + group_size] for i in range(0, len(agents), group_size)]
