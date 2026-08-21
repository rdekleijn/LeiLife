from .geometry import gen_rand_location


class FoodToken:
    def __init__(self, env=None, location=None, value=1):
        self.location = location
        self.env = env
        self.isEaten = False
        self.value = value


class Predator:
    def __init__(self, env=None, location=None):
        self.location = location
        self.env = env


class Environment:
    def __init__(self, experiment=None, iter=0, size=100):
        self.experiment = experiment
        self.iter = iter
        self.size = size
        self.agents = []
        self.deadagents = []
        self.foodtokens = []

    def check_boundary(self):
        for agent in self.agents:
            if agent.location[0] < 0:
                agent.location[0] = 0
            if agent.location[1] < 0:
                agent.location[1] = 0
            if agent.location[0] > self.size:
                agent.location[0] = self.size
            if agent.location[1] > self.size:
                agent.location[1] = self.size

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_foodtoken(self, location):
        self.foodtokens.append(FoodToken(env=self, location=location))

    def check_eaten_foodtokens(self):
        # Collect eaten tokens first, then mutate self.foodtokens afterwards.
        # (Removing items from the list mid-iteration could skip a token two
        # agents reach in the same tick.)
        eaten = []
        for agent in self.agents:
            for foodtoken in self.foodtokens:
                if foodtoken in eaten:
                    continue
                if (abs(agent.location[0] - foodtoken.location[0]) < 3
                        and abs(agent.location[1] - foodtoken.location[1]) < 3):
                    eaten.append(foodtoken)
                    agent.eatenTokens += 1
        for foodtoken in eaten:
            self.foodtokens.remove(foodtoken)
            self.add_foodtoken(location=gen_rand_location(self.size))

    def update(self):
        self.iter += 1
        for agent in self.agents:
            agent.update()
        self.check_eaten_foodtokens()
        self.check_boundary()
