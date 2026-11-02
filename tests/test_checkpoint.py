import random

import numpy as np

from alife.agents import Agent, bump_agent_id_counter
from alife.checkpoint import save_checkpoint, load_checkpoint


def test_save_and_load_checkpoint_round_trip(tmp_path):
    path = str(tmp_path / "ckpt.pkl")
    agents = [Agent(lineage_id=i) for i in range(3)]
    for a in agents:
        a.fitness = 1.0

    save_checkpoint(path, gen=7, agents=agents)
    loaded = load_checkpoint(path)

    assert loaded['gen'] == 7
    assert len(loaded['agents']) == 3
    assert [a.agent_id for a in loaded['agents']] == [a.agent_id for a in agents]


def test_load_checkpoint_restores_rng_state(tmp_path):
    path = str(tmp_path / "ckpt.pkl")
    random.seed(42)
    np.random.seed(42)
    save_checkpoint(path, gen=0, agents=[])
    saved_random_state = random.getstate()
    saved_np_state = np.random.get_state()

    random.random()
    np.random.random()
    assert random.getstate() != saved_random_state

    load_checkpoint(path)

    assert random.getstate() == saved_random_state
    restored_np_state = np.random.get_state()
    assert restored_np_state[0] == saved_np_state[0]
    assert np.array_equal(restored_np_state[1], saved_np_state[1])


def test_bump_agent_id_counter_avoids_collisions():
    a = Agent()
    high_id = a.agent_id + 1000
    bump_agent_id_counter(high_id)

    b = Agent()

    assert b.agent_id > high_id
