import numpy as np
import pytest

from alife.agents import Agent, PLASTICITY_DECAY, SOCIAL_LEARNING_RATE, net_dims
from alife.environment import Environment
from alife.nn import FeedForwardNet
from .helpers import ConstantNet


def test_social_learning_requires_signaling():
    with pytest.raises(ValueError):
        Agent(signaling=False, social_learning=True)


def test_social_learning_starts_blank():
    agent = Agent(signaling=True, social_learning=True)
    assert np.all(agent.W_plastic == 0)
    assert np.all(agent.signal_trace == 0)


def test_observe_and_learn_is_noop_without_social_learning():
    agent = Agent(signaling=True, social_learning=False)
    teacher = Agent(signaling=True)
    teacher.motor_output = np.array([1.0, -1.0])
    agent.observe_and_learn([teacher])
    assert np.all(agent.W_plastic == 0)


def test_observe_and_learn_associates_trace_with_rewarded_neighbors_action():
    observer = Agent(signaling=True, social_learning=True)
    observer.signal_trace = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    teacher = Agent(signaling=True)
    teacher.motor_output = np.array([1.5, -0.3])

    observer.observe_and_learn([teacher])

    expected = SOCIAL_LEARNING_RATE * np.outer(observer.signal_trace, teacher.motor_output)
    np.testing.assert_allclose(observer.W_plastic, expected)


def test_observe_and_learn_ignores_self_in_rewarded_list():
    observer = Agent(signaling=True, social_learning=True)
    observer.signal_trace = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    observer.motor_output = np.array([9.0, 9.0])  # would blow up the assertion if self weren't excluded

    observer.observe_and_learn([observer])

    assert np.all(observer.W_plastic == 0)


def test_observe_and_learn_decays_existing_weights_without_reward():
    observer = Agent(signaling=True, social_learning=True)
    observer.W_plastic = np.ones((8, 2))

    observer.observe_and_learn([])

    np.testing.assert_allclose(observer.W_plastic, np.ones((8, 2)) * (1 - PLASTICITY_DECAY))


def test_plastic_bias_shifts_motor_output():
    agent = Agent(location=[0.0, 0.0], orientation=0, nnet=ConstantNet([0.0, 0.0, 0.0]),
                   signaling=True, social_learning=True)
    agent.signal_input = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    agent.W_plastic = np.zeros((8, 2))
    agent.W_plastic[0] = [0.5, -0.5]

    agent.move()

    assert list(agent.motor_output) == pytest.approx([0.5, -0.5])


def test_environment_update_calls_observe_and_learn_end_to_end():
    # Scripted scenario: a teacher starts on top of a food token (so it
    # eats on tick 1 regardless of motion) and signals; an observer starts
    # 10 units away, with nothing to do but listen and learn. Because
    # env.agents processes the teacher first, the observer senses the
    # teacher's *this-tick* signal already on tick 1 (see the ordering
    # note in agents.py / docs/facelift-design.md Phase 2).
    np.random.seed(0)
    env = Environment(size=40)
    env.add_foodtoken(location=[10.0, 10.0])
    teacher = Agent(env=env, location=[10.0, 10.0], orientation=0,
                     nnet=ConstantNet([1.0, -1.0, 1.5]), signaling=True)
    observer = Agent(env=env, location=[10.0, 20.0], orientation=0,
                      nnet=ConstantNet([0.0, 0.0, 0.0]), signaling=True, social_learning=True)
    env.add_agent(teacher)
    env.add_agent(observer)

    env.update()

    assert teacher.eatenTokens == 1
    assert np.any(observer.W_plastic != 0)


def test_social_learning_diverges_by_social_experience_not_genotype():
    # The core Phase 3 claim: two agents built from the *same* evolved
    # weights end up with different learned associations purely because
    # of what they experienced socially -- nothing genetic differs.
    np.random.seed(0)
    n_in, n_out = net_dims(True)
    shared_weights = FeedForwardNet(n_in, 4, n_out).params.copy()

    def make_clone():
        net = FeedForwardNet(n_in, 4, n_out, params=shared_weights.copy())
        return Agent(nnet=net, signaling=True, social_learning=True)

    def run_scripted(teacher_output, num_ticks=10):
        env = Environment(size=40)
        env.add_foodtoken(location=[10.0, 10.0])
        clone = make_clone()
        clone.env, clone.location, clone.orientation = env, [10.0, 20.0], 0
        teacher = Agent(env=env, location=[10.0, 10.0], orientation=0,
                         nnet=ConstantNet(teacher_output), signaling=True)
        env.add_agent(teacher)  # processed first, see ordering note above
        env.add_agent(clone)
        for _ in range(num_ticks):
            env.update()
        return clone

    # Both teachers signal equally and both succeed (start on the food);
    # they differ only in *which* motor action is rewarded.
    clone_a = run_scripted([1.0, -1.0, 1.5])
    clone_b = run_scripted([-1.0, 1.0, 1.5])

    assert not np.allclose(clone_a.W_plastic, clone_b.W_plastic, atol=1e-9)
