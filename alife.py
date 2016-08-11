import random
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import one_to_n
from math import atan2, degrees, pi
from datetime import datetime


class Experiment:
    def __init__(self, num_cores = 1):
        self.num_cores = num_cores
        self.starttime = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.logfile = 'output/' + self.starttime

    def init_logfiles(self, filename=None):
        if filename is None:
            filename = self.logfile
        filename = filename + '.txt'
        f = open(filename, 'w')
        output = 'condition,gen,min,max,avg\n'
        f.write(output)
        f.close()

    def write_log(self, condition, current_generation, fitness, time_elapsed, filename=None):
        if filename is None:
            filename = self.logfile
        filename = filename + '.txt'
        print "Generation", current_generation, \
            "-- Mean fitness:", '{0:.4f}'.format(np.mean(fitness)), \
            '-- Min fitness:', min(fitness), \
            '-- Max fitness:', max(fitness), \
            '-- Elapsed time:', str(int(time_elapsed)), 's'
        output = "Generation " + str(current_generation) + " Mean fitness: " + str(
            '{0:.4f}'.format(np.mean(fitness))) + "\n"
        f = open('output/textlog.txt', 'a')
        f.write(output)
        f.close
        output = str(condition) + ',' + str(current_generation) + ',' + str(min(fitness)) + ',' + str(
            max(fitness)) + ',' + str(
            '{0:.4f}'.format(np.mean(fitness))) + '\n'
        f = open(filename, 'a')
        f.write(output)
        f.close()


class Environment:
    def __init__(self, experiment, iter=0, size=100):
        self.experiment = experiment
        self.iter = iter
        self.size = size
        self.agents = []
        self.deadagents = []
        self.foodtokens = []

    def check_boundary(self):
        for agent in self.agents:
            if agent.location[0] < 0: agent.location[0] = 0
            if agent.location[1] < 0: agent.location[1] = 0
            if agent.location[0] > self.size: agent.location[0] = self.size
            if agent.location[1] > self.size: agent.location[1] = self.size

    def add_agent(self, location, nnet=None, weights=None):
        self.agents.append(Agent(env=self, location=location, nnet=nnet, weights=weights))

    def add_foodtoken(self, location):
        self.foodtokens.append(FoodToken(location=location))

    def check_eaten_foodtokens(self):
        for agent in self.agents:
            for foodtoken in self.foodtokens:
                if abs(agent.location[0] - foodtoken.location[0]) < 3 and abs(agent.location[1] - foodtoken.location[1]) < 3:
                    self.foodtokens.remove(foodtoken)
                    self.add_foodtoken(location=gen_rand_loc(self.size))
                    agent.eatenTokens += 1

    def update(self):
        self.iter += 1
        for agent in self.agents:
            agent.update()
        self.check_eaten_foodtokens()
        self.check_boundary()


class FoodToken:
    def __init__(self, env=None, location=None):
        self.location = location
        self.env = env
        self.isEaten = False


class Predator:
    def __init__(self, env=None, location=None):
        self.location = location
        self.env = env


class Agent:
    def __init__(self, env=None, location=None, nnet=None, weights=None):
        self.location = location
        self.lifecycle = 0
        self.lifeOver = False
        self.eatenTokens = 0
        self.env = env
        self.visual_input = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.motor_output = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if nnet is None:
            self.nnet = buildNetwork(8, 8, 8)
            if weights is not None:
                self.nnet._setParameters(weights)
        else:
            self.nnet = nnet
            if weights is not None:
                self.nnet._setParameters(weights)

    def give_status(self):
        print('X location = ', self.location[0], '\n', 'Y location =', self.location[1])

    def update_visual_field(self):
        for foodtoken in self.env.foodtokens:
            dx = foodtoken.location[0] - self.location[0]
            dy = foodtoken.location[1] - self.location[1]
            rads = atan2(dy, dx)
            rads %= 2 * pi
            degs = degrees(rads)
            self.visual_input = one_to_n(degs/360*8, 8) + np.random.normal(0, .2, 8)

    def cycle_nnet(self):
        motor_output = self.nnet.activate(self.visual_input)
        return(np.argmax(motor_output))

    def move(self):
        if self.cycle_nnet() == 1:
            self.location = [self.location[0] - 1, self.location[1] + 1]
        elif self.cycle_nnet() == 2:
            self.location = [self.location[0], self.location[1] + 1]
        elif self.cycle_nnet() == 3:
            self.location = [self.location[0] + 1, self.location[1] + 1]
        elif self.cycle_nnet() == 4:
            self.location = [self.location[0] - 1, self.location[1]]
        elif self.cycle_nnet() == 5:
            self.location = [self.location[0] + 1, self.location[1]]
        elif self.cycle_nnet() == 6:
            self.location = [self.location[0] - 1, self.location[1] - 1]
        elif self.cycle_nnet() == 7:
            self.location = [self.location[0], self.location[1] + 1]
        elif self.cycle_nnet() == 8:
            self.location = [self.location[0] + 1, self.location[1] + 1]

    def update(self):
        self.update_visual_field()
        self.cycle_nnet()
        self.move()
        if self.lifecycle > 600:
            self.lifeOver = True
        else:
            self.lifecycle += 1


def gen_rand_loc(size):
    return([random.randint(0, size), random.randint(0, size)])


def run_agent(exp, nnet=None, lifetime=600, weights=None):
    env = Environment(experiment=exp, size=50)
    env.add_foodtoken(location=gen_rand_loc(env.size))
    env.add_agent(location=gen_rand_loc(env.size), nnet=nnet, weights=weights)
    for iter in range(lifetime):
        env.update()
    return (env.agents[0])