import numpy as np
import pytest

from alife.agents import Agent, net_dims, SIGNAL_FITNESS_FACTOR, DISTANCE_FITNESS_FACTOR
from alife.environment import Environment
from alife.evolution import select_and_reproduce
from alife.nn import FeedForwardNet
from .helpers import ConstantNet


def test_net_dims():
    assert net_dims(False) == (8, 2)
    assert net_dims(True) == (16, 3)


def test_non_signaling_agent_ignores_signal_channel():
    # Default (signaling=False) behavior must be untouched by Phase 2:
    # cycle_nnet only ever sees the 8-dim food channel.
    agent = Agent(location=[0.0, 0.0], orientation=0, nnet=ConstantNet([1.0, 1.0]))
    agent.move()
    assert agent.emitted_signal == 0.0
    assert agent.signalEmitted == 0.0
    assert list(agent.motor_output) == [1.0, 1.0]


def test_signaling_agent_emits_and_pays_cost():
    agent = Agent(location=[0.0, 0.0], orientation=0, nnet=ConstantNet([1.0, 1.0, 1.5]), signaling=True)
    agent.move()
    assert list(agent.motor_output) == [1.0, 1.0]
    assert agent.emitted_signal == pytest.approx(1.5)
    assert agent.signalEmitted == pytest.approx(1.5)

    agent.move()
    assert agent.signalEmitted == pytest.approx(3.0)  # accumulates across ticks


def test_signal_output_is_clipped_to_nonnegative_range():
    agent = Agent(location=[0.0, 0.0], orientation=0, nnet=ConstantNet([0.0, 0.0, -5.0]), signaling=True)
    agent.move()
    assert agent.emitted_signal == 0.0

    agent = Agent(location=[0.0, 0.0], orientation=0, nnet=ConstantNet([0.0, 0.0, 5.0]), signaling=True)
    agent.move()
    assert agent.emitted_signal == 2.0


def test_update_signal_field_senses_emitting_neighbor():
    env = Environment(size=100)
    listener = Agent(env=env, location=[10.0, 10.0], orientation=0, nnet=ConstantNet([0, 0, 0]), signaling=True)
    speaker = Agent(env=env, location=[10.0, 20.0], orientation=0, nnet=ConstantNet([0, 0, 0]), signaling=True)
    speaker.emitted_signal = 2.0
    env.add_agent(listener)
    env.add_agent(speaker)

    listener.update_signal_field()

    assert np.sum(listener.signal_input) > 0.5  # picks up the speaker's broadcast


def test_update_signal_field_ignores_silent_neighbor():
    env = Environment(size=100)
    listener = Agent(env=env, location=[10.0, 10.0], orientation=0, nnet=ConstantNet([0, 0, 0]), signaling=True)
    silent = Agent(env=env, location=[10.0, 20.0], orientation=0, nnet=ConstantNet([0, 0, 0]), signaling=True)
    # silent.emitted_signal stays at its default of 0.0
    env.add_agent(listener)
    env.add_agent(silent)

    listener.update_signal_field()

    # only the np.random.normal(0, .1, 8) noise term should be present
    assert np.max(np.abs(listener.signal_input)) < 1.0


def test_update_includes_signal_cost_in_fitness_when_signaling():
    env = Environment(size=100)
    agent = Agent(env=env, location=[10.0, 10.0], orientation=0, nnet=ConstantNet([0.0, 0.0, 0.0]),
                  signaling=True)
    agent.eatenTokens = 5
    agent.wheelDistanceTraveled = 100.0
    agent.signalEmitted = 40.0
    agent.lifecycle = 601
    agent.update()
    assert agent.fitness == pytest.approx(
        5 - DISTANCE_FITNESS_FACTOR * 100.0 - SIGNAL_FITNESS_FACTOR * 40.0
    )


def test_select_and_reproduce_with_signaling_builds_correct_net_shape():
    np.random.seed(0)
    n_in, n_out = net_dims(True)
    agents = []
    for f in [3, 1]:
        net = FeedForwardNet(n_in, 4, n_out)
        a = Agent(nnet=net, signaling=True)
        a.fitness = f
        agents.append(a)

    offspring, _ = select_and_reproduce(agents, n_hidden=4, top_n=2, offspring_per=1,
                                         mutation_std=0.1, signaling=True)

    assert len(offspring) == 2
    for child in offspring:
        assert child.signaling is True
        assert child.nnet.n_in == 16
        assert child.nnet.n_out == 3
