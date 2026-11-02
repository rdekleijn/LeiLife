from alife.agents import Agent
from alife.environment import Environment
from .helpers import ConstantNet


def test_check_boundary_clamps_location():
    env = Environment(size=50)
    agent = Agent(env=env, location=[-5.0, 60.0], orientation=0, nnet=ConstantNet([0.0, 0.0]))
    env.add_agent(agent)
    env.check_boundary()
    assert agent.location == [0.0, 50.0]


def test_eaten_foodtoken_is_replaced_and_credited_once():
    env = Environment(size=50)
    env.add_foodtoken(location=[10.0, 10.0])
    agent = Agent(env=env, location=[10.0, 10.0], orientation=0, nnet=ConstantNet([0.0, 0.0]))
    env.add_agent(agent)

    env.check_eaten_foodtokens()

    assert agent.eatenTokens == 1
    assert len(env.foodtokens) == 1  # eaten token replaced with a new one


def test_two_agents_on_same_token_only_one_gets_credit_per_tick():
    # Regression test: the original implementation removed items from
    # self.foodtokens while iterating over it, which could silently skip a
    # token two agents reached in the same tick. This checks the token is
    # eaten exactly once (by the first agent in iteration order) rather
    # than being double-counted or skipped.
    env = Environment(size=50)
    env.add_foodtoken(location=[10.0, 10.0])
    agent_a = Agent(env=env, location=[10.0, 10.0], orientation=0, nnet=ConstantNet([0.0, 0.0]))
    agent_b = Agent(env=env, location=[10.5, 10.5], orientation=0, nnet=ConstantNet([0.0, 0.0]))
    env.add_agent(agent_a)
    env.add_agent(agent_b)

    env.check_eaten_foodtokens()

    assert agent_a.eatenTokens + agent_b.eatenTokens == 1
    assert len(env.foodtokens) == 1


def test_uneaten_foodtoken_is_untouched():
    env = Environment(size=50)
    env.add_foodtoken(location=[40.0, 40.0])
    agent = Agent(env=env, location=[0.0, 0.0], orientation=0, nnet=ConstantNet([0.0, 0.0]))
    env.add_agent(agent)

    env.check_eaten_foodtokens()

    assert agent.eatenTokens == 0
    assert env.foodtokens[0].location == [40.0, 40.0]
