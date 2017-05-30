import random
import csv
import numpy as np
import pandas as pd
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import one_to_n
from math import atan2, degrees, radians, sin, cos, pi, floor, ceil, sqrt
from datetime import datetime
import animate
# from pybrain.structure import LSTMLayer, SigmoidLayer, LinearLayer
# net = buildNetwork(5, 2, 10, hiddenclass=LSTMLayer, outclass=LinearLayer, recurrent=True, bias=True)

# fitness function parameters (must tweak in different tasks/environments)
distance_fitness_factor = .0001
connection_fitness_factor = 0 #.00001

class Experiment:
    def __init__(self, num_cores = 1, lifetime = 600):
        self.num_cores = num_cores
        self.starttime = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.logfile = 'output/' + self.starttime
        self.lifetime = lifetime

    def init_logfiles(self, filename=None):
        if filename is None:
            filename = self.logfile
        filename = filename + '.txt'
        f = open(filename, 'w')
        output = 'condition,gen,min,max,avg\n'
        f.write(output)
        f.close()

    def write_log(self, condition, current_generation, fitness, eatentokens, disttrav, time_elapsed, timecenter, filename=None):
        if filename is None:
            filename = self.logfile
        filename = filename + '.txt'
        print "Generation", current_generation, \
            "-- Mean fitness:", '{0:.4f}'.format(np.mean(fitness)), \
            "-- Mean tokens eaten:", '{0:.4f}'.format(np.mean(eatentokens)), \
            '-- Mean distance traveled:', '{0:.4f}'.format(np.mean(disttrav)), \
            '-- Prop in center:', '{0:.4f}'.format(np.mean(timecenter)/600), \
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
    def __init__(self, experiment, iter=0, ITI=20, size=100, LFE=0, sequence=[], verbose=False):
        self.experiment = experiment
        self.verbose = verbose
        self.sequence = sequence # if [], generate random stimlocs
        self.iter = iter
        self.ITI = ITI # time betwen food tokens?
        self.size = size
        self.last_stim_loc = 0
        self.agents = []
        self.deadagents = []
        self.foodtokens = []
        self.last_food_eaten = LFE
        self.eatenTokens = 0 # use this to track sequence position
        self.stimlocs = {1:[size*.2, size*.8], # upper left
                         2:[size*.8, size*.8], # upper right
                         3:[size*.2, size*.2], # lower left
                         4:[size*.8, size*.2]} # lower right

    def check_boundary(self):
        for agent in self.agents:
            if agent.location[0] < 0: agent.location[0] = 0
            if agent.location[1] < 0: agent.location[1] = 0
            if agent.location[0] > self.size: agent.location[0] = self.size
            if agent.location[1] > self.size: agent.location[1] = self.size

    def add_agent(self, location, orientation, nnet=None, weights=None):
        self.agents.append(Agent(env=self, location=location, orientation=orientation, nnet=nnet, weights=weights))

    def add_foodtoken(self, location):
        # print 'food token added at ' + str(location)
        self.foodtokens.append(FoodToken(location=location, iter_created=self.iter))

    def check_eaten_foodtokens(self):
        for agent in self.agents:
            for foodtoken in self.foodtokens:
                # print 'food token is at ' + str(foodtoken.location)
                if abs(agent.location[0] - foodtoken.location[0]) < 3 and abs(agent.location[1] - foodtoken.location[1]) < 3:
                    self.foodtokens.remove(foodtoken)
                    self.last_food_eaten = self.iter
                    self.eatenTokens += 1 # track in environment
                    agent.eatenTokens += 1 # track in agent (meh)
                    agent.totalReward += max(0, 100 - (self.iter - foodtoken.iter_created))
        if self.iter - self.last_food_eaten == self.ITI:
            if self.sequence:
                # print "eaten: " + str(self.eatenTokens) + " "
                # print self.sequence
                self.add_foodtoken(location=self.stimlocs[self.sequence[(self.eatenTokens%len(self.sequence))]])
            else:
                self.add_foodtoken(location=self.gen_next_food_location(self.size, stimulus=True, lastloc=self.last_stim_loc))

    def update(self):
        self.iter += 1
        for agent in self.agents:
            agent.update()
        self.check_eaten_foodtokens()
        self.check_boundary()

    def gen_next_food_location(self, size, stimulus=False, lastloc=0):
        if stimulus is False:
            return([random.uniform(0, size), random.uniform(0, size)])
        else:
            if lastloc == 1:
                stimpos = int(random.sample([2,3,4], 1)[0])
            elif lastloc == 2:
                stimpos = int(random.sample([1,3,4], 1)[0])
            elif lastloc == 3:
                stimpos = int(random.sample([1,2,4], 1)[0])
            elif lastloc == 4:
                stimpos = int(random.sample([1,2,3], 1)[0])
            elif lastloc == 0:
                stimpos = int(random.sample([1,2,3,4], 1)[0])

            return(self.stimlocs[stimpos])


class FoodToken:
    def __init__(self, env=None, location=None, value=1, iter_created=None):
        self.location = location
        self.env = env
        self.isEaten = False
        self.value = value
        self.iter_created = iter_created


class Predator:
    def __init__(self, env=None, location=None):
        self.location = location
        self.env = env


class Agent:
    def __init__(self, env=None, location=None, orientation=None, nnet=None, weights=None):
        self.location = location
        self.orientation = orientation
        self.lifecycle = 0
        self.lifeOver = False
        self.eatenTokens = 0
        self.totalReward = 0
        self.wheelDistanceTraveled = 0
        self.fitness = None
        self.env = env
        self.steps_in_center = 0
        self.visual_input = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.motor_output = [0.0, 0.0]
        if nnet is None:
            self.nnet = buildNetwork(8, 8, 2)
            if weights is not None:
                self.nnet._setParameters(weights)
        else:
            self.nnet = nnet
            if weights is not None:
                self.nnet._setParameters(weights)

    def give_status(self):
        print('X location = ', self.location[0], '\n', 'Y location =', self.location[1])

    def update_visual_field(self):
        self.visual_input = [float((self.location[0] - (self.env.size / 2)) / (self.env.size / 2)),
                             float((self.location[1] - (self.env.size / 2)) / (self.env.size / 2)),
                             0,0] + np.random.normal(0,.1, 4)
        for foodtoken in self.env.foodtokens:
            self.visual_input = [float((self.location[0] - (self.env.size / 2)) / (self.env.size / 2)),
                                 float((self.location[1] - (self.env.size / 2)) / (self.env.size / 2)),
                                 (foodtoken.location[0] - (self.env.size / 2)) / (self.env.size / 2),
                                 (foodtoken.location[1] - (self.env.size / 2)) / (self.env.size / 2)] + np.random.normal(0, .1, 4)
            #print self.visual_input

    def cycle_nnet(self):
        motor_output = self.nnet.activate(self.visual_input)
        return(motor_output)

    def move(self):
        self.motor_output = self.cycle_nnet()
        #print self.location
        self.motor_output = np.clip(self.cycle_nnet(), -2, 2)
        self.wheelDistanceTraveled += sqrt(np.square(self.motor_output[0]) + np.square(self.motor_output[1]))
        self.location = [self.location[0] + self.motor_output[0], self.location[1] + self.motor_output[1]]
        if abs(self.env.size / 2 - self.location[0]) < 7 and abs(self.env.size / 2 - self.location[1]) < 7:
            self.steps_in_center += 1

    def update(self):
        self.update_visual_field()
        self.move()
        if self.lifecycle == self.env.experiment.lifetime-1:
            self.lifeOver = True
            self.fitness = self.eatenTokens + self.totalReward
        else:
            self.lifecycle += 1
        # We need to punish time spent between stim appearance and touching it! We need ISI


def gen_rand_orientation():
    return(random.uniform(0,360))

def gen_rand_position(max):
    return([random.uniform(0, max), random.uniform(0, max)])

def calc_dir_and_dist(dx, dy, orientation):
    rads = atan2(dy, dx)
    degs = ((degrees(rads) - 90) * -1) % 360
    degs = (degs - orientation) % 360
    dist = sqrt(dx ** 2 + dy ** 2)
    return(degs, dist)


def run_agent(exp, size=None, nnet=None, lifetime=600, weights=None, verbose=False, fname="", sequence=[]):
    env = Environment(experiment=exp, size=size, sequence=sequence, verbose=verbose)
    # env.add_foodtoken(location=env.gen_next_food_location(size, stimulus=True))
    env.add_foodtoken(location=env.stimlocs[1])
    env.add_agent(location=[size/2.0, size/2.0], orientation=gen_rand_orientation(), nnet=nnet, weights=weights)
    if verbose:
        ddat = {'agent_xpos':[],
                'agent_ypos':[],
                'food_xpos':[],
                'food_ypos':[],
                'tokenCount':[]}
    for iter in range(lifetime):
        env.update()
        if verbose:
            ddat['agent_xpos'].append(env.agents[0].location[0])
            ddat['agent_ypos'].append(env.agents[0].location[1])
            if not env.foodtokens: # ITI - no food now
                ddat['food_xpos'].append(-1)
                ddat['food_ypos'].append(-1)
            else:
                ddat['food_xpos'].append(env.foodtokens[0].location[0])
                ddat['food_ypos'].append(env.foodtokens[0].location[1])
                # print env.foodtokens
                # print env.foodtokens[0].location
            ddat['tokenCount'].append(env.agents[0].eatenTokens)
    if verbose:
        df = pd.DataFrame(ddat)
        df.to_csv('output/'+fname+'_agent_run.csv')
        animate.save_movie(fname, df, lifetime, size)

    return (env.agents[0]) # full agent
