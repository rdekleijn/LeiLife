import pytest

from alife.agents import Agent, DISTANCE_FITNESS_FACTOR
from alife.environment import Environment
from .helpers import ConstantNet


def test_move_straight_line():
    agent = Agent(location=[10.0, 10.0], orientation=0, nnet=ConstantNet([1.0, 1.0]))
    agent.move()
    # orientation 0 == "north": x unchanged, y increases by the wheel speed
    assert agent.location[0] == pytest.approx(10.0)
    assert agent.location[1] == pytest.approx(11.0)
    assert agent.orientation == 0
    assert agent.wheelDistanceTraveled == pytest.approx(2.0)


def test_move_pure_rotation_in_place():
    # left + right == 0 => turning radius R collapses to 0: agent spins
    # without translating.
    agent = Agent(location=[10.0, 10.0], orientation=0, nnet=ConstantNet([1.0, -1.0]))
    agent.move()
    assert agent.location[0] == pytest.approx(10.0)
    assert agent.location[1] == pytest.approx(10.0)
    assert agent.orientation == pytest.approx(350.0)  # (0 + (-1-1)/0.2) % 360


def test_motor_output_is_clipped():
    agent = Agent(location=[0.0, 0.0], orientation=0, nnet=ConstantNet([10.0, 10.0]))
    agent.move()
    assert list(agent.motor_output) == [2.0, 2.0]


def test_update_sets_fitness_after_lifecycle_threshold():
    env = Environment(size=100)  # no food tokens
    agent = Agent(env=env, location=[10.0, 10.0], orientation=0, nnet=ConstantNet([0.0, 0.0]))
    agent.eatenTokens = 5
    agent.wheelDistanceTraveled = 100.0
    agent.lifecycle = 601
    agent.update()
    assert agent.lifeOver is True
    assert agent.fitness == pytest.approx(5 - DISTANCE_FITNESS_FACTOR * 100.0)


def test_update_before_threshold_keeps_life_going():
    env = Environment(size=100)
    agent = Agent(env=env, location=[10.0, 10.0], orientation=0, nnet=ConstantNet([0.0, 0.0]))
    agent.lifecycle = 5
    agent.update()
    assert agent.lifeOver is False
    assert agent.lifecycle == 6
    assert agent.fitness is None
