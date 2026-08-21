import glob
import os

import numpy as np

from alife.agents import Agent
from alife.experiment import run_group, run_group_evolution
from alife.logbook import GenerationLog
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


def test_run_group_evolution_with_social_learning_smoke():
    final_agents = run_group_evolution(
        num_agents=4, num_gens=1, lifetime=605, env_size=40,
        group_size=2, num_cores=1, save_best_every=999, n_hidden=4,
        top_n=2, offspring_per=2, signaling=True, social_learning=True,
    )
    assert len(final_agents) == 4
    for agent in final_agents:
        assert agent.social_learning is True
        assert agent.W_plastic.shape == (8, 2)


def test_run_group_evolution_writes_generation_log(tmp_path):
    output_dir = str(tmp_path / "run")
    run_group_evolution(
        num_agents=4, num_gens=2, lifetime=605, env_size=40, group_size=2,
        num_cores=1, save_best_every=999, n_hidden=4, top_n=2, offspring_per=2,
        output_dir=output_dir,
    )
    df = GenerationLog(os.path.join(output_dir, 'generation_log.csv')).load()
    assert set(df['gen']) == {0, 1}
    assert len(df) == 8  # 4 agents x 2 generations


def test_run_group_evolution_seeded_run_is_reproducible(tmp_path):
    kwargs = dict(
        num_agents=4, num_gens=2, lifetime=605, env_size=40, group_size=2,
        num_cores=1, save_best_every=999, n_hidden=4, top_n=2, offspring_per=2,
        seed=42,
    )
    agents_a = run_group_evolution(output_dir=str(tmp_path / "run_a"), **kwargs)
    agents_b = run_group_evolution(output_dir=str(tmp_path / "run_b"), **kwargs)

    for a, b in zip(agents_a, agents_b):
        np.testing.assert_allclose(a.nnet.params, b.nnet.params)


def test_run_group_evolution_checkpoint_and_resume(tmp_path):
    output_dir = str(tmp_path / "run")
    run_group_evolution(
        num_agents=4, num_gens=2, lifetime=605, env_size=40, group_size=2,
        num_cores=1, save_best_every=999, checkpoint_every=1, seed=1,
        n_hidden=4, top_n=2, offspring_per=2, output_dir=output_dir,
    )
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pkl')
    assert os.path.exists(checkpoint_path)

    resumed_agents = run_group_evolution(
        num_agents=4, num_gens=3, lifetime=605, env_size=40, group_size=2,
        num_cores=1, save_best_every=999, resume_from=checkpoint_path,
        n_hidden=4, top_n=2, offspring_per=2, output_dir=output_dir,
    )
    assert len(resumed_agents) == 4
    # the resumed run only had to execute generation 2 (0 and 1 already ran
    # before the checkpoint), so the log should show exactly 3 generations
    # once combined -- but since resuming reopens the same log path, all
    # three generations should be present in one file.
    df = GenerationLog(os.path.join(output_dir, 'generation_log.csv')).load()
    assert set(df['gen']) == {0, 1, 2}


def test_run_group_evolution_save_all_groups_every(tmp_path):
    output_dir = str(tmp_path / "run")
    run_group_evolution(
        num_agents=4, num_gens=1, lifetime=605, env_size=40, group_size=2,
        num_cores=1, save_best_every=999, save_all_groups_every=1,
        n_hidden=4, top_n=2, offspring_per=2, output_dir=output_dir,
    )
    csv_files = sorted(glob.glob(os.path.join(output_dir, 'gen0_group*_group_run.csv')))
    assert len(csv_files) == 2  # 4 agents / group_size 2 = 2 groups
