import numpy as np
import pandas as pd
import pytest

from alife.agents import Agent
from alife.logbook import GenerationLog, agent_log_rows


def make_agent(fitness=1.0, signaling=False, social_learning=False):
    a = Agent(signaling=signaling, social_learning=social_learning)
    a.fitness = fitness
    return a


def test_agent_log_rows_basic_fields():
    a = make_agent(fitness=2.5)
    rows = agent_log_rows(gen=3, agents=[a])
    assert rows[0]['gen'] == 3
    assert rows[0]['agent_id'] == a.agent_id
    assert rows[0]['fitness'] == 2.5
    assert rows[0]['signal_emitted'] is None  # not signaling -> not applicable
    assert rows[0]['plastic_weight_norm'] is None  # not social-learning -> not applicable


def test_agent_log_rows_signaling_and_social_learning_fields():
    a = make_agent(signaling=True, social_learning=True)
    a.signalEmitted = 5.0
    a.W_plastic = np.ones((8, 2))

    rows = agent_log_rows(gen=0, agents=[a])

    assert rows[0]['signal_emitted'] == 5.0
    assert rows[0]['plastic_weight_norm'] == pytest.approx(np.linalg.norm(np.ones((8, 2))))


def test_generation_log_append_and_load(tmp_path):
    path = str(tmp_path / "gen_log.csv")
    log = GenerationLog(path)
    log.append(0, [make_agent(fitness=1.0)])
    log.append(1, [make_agent(fitness=2.0), make_agent(fitness=3.0)])

    df = log.load()

    assert len(df) == 3
    assert list(df['gen']) == [0, 1, 1]


def test_generation_log_reopen_does_not_duplicate_header(tmp_path):
    path = str(tmp_path / "gen_log.csv")
    GenerationLog(path).append(0, [make_agent()])
    GenerationLog(path).append(1, [make_agent()])  # simulates reopening after a resume

    df = pd.read_csv(path)

    assert len(df) == 2
    assert list(df['gen']) == [0, 1]
