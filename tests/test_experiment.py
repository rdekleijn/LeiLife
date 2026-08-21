from alife.agents import Agent
from alife.experiment import run_group, run_group_evolution
from alife.nn import FeedForwardNet
from .helpers import ConstantNet


def test_run_group_shares_one_environment():
    agents = [Agent(nnet=ConstantNet([0.0, 0.0]), lineage_id=i) for i in range(3)]

    result = run_group(size=40, agents=agents, lifetime=5)

    assert len(result) == 3
    envs = {id(a.env) for a in result}
    assert len(envs) == 1  # all three agents ran in the same shared Environment
    assert len(result[0].env.agents) == 3
    assert len(result[0].env.foodtokens) == 3  # defaults to one token per agent


def test_run_group_respects_num_foodtokens():
    agents = [Agent(nnet=ConstantNet([0.0, 0.0])) for _ in range(3)]
    result = run_group(size=40, agents=agents, lifetime=5, num_foodtokens=7)
    assert len(result[0].env.foodtokens) == 7


def test_run_group_evolution_smoke():
    # Small/fast end-to-end check that groups form, run, and produce the
    # next generation without error. lifetime must exceed 601 for
    # Agent.update() to ever set .fitness (see Agent.update).
    final_agents = run_group_evolution(
        num_agents=4, num_gens=1, lifetime=605, env_size=40,
        group_size=2, num_cores=1, save_best_every=999, n_hidden=4,
        top_n=2, offspring_per=2,
    )
    assert len(final_agents) == 2 * 2  # top_n * offspring_per
    for agent in final_agents:
        assert agent.lineage_id in (0, 1)


def test_run_group_evolution_kin_based_smoke():
    final_agents = run_group_evolution(
        num_agents=4, num_gens=1, lifetime=605, env_size=40,
        group_size=2, kin_based_groups=True, num_cores=1, save_best_every=999,
        n_hidden=4, top_n=2, offspring_per=2,
    )
    assert len(final_agents) == 4


def test_run_group_evolution_with_signaling_smoke():
    # gen 0 always triggers the save-best replay (0 % save_best_every == 0),
    # so this also exercises run_group's verbose/animate path with
    # signaling-shaped agents.
    final_agents = run_group_evolution(
        num_agents=4, num_gens=1, lifetime=605, env_size=40,
        group_size=2, num_cores=1, save_best_every=999, n_hidden=4,
        top_n=2, offspring_per=2, signaling=True,
    )
    assert len(final_agents) == 4
    for agent in final_agents:
        assert agent.signaling is True
        assert agent.nnet.n_in == 16
        assert agent.nnet.n_out == 3
