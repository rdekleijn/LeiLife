import random
import csv
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import one_to_n
from math import atan2, degrees, radians, sin, cos, pi, floor, ceil, sqrt
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# fitness function parameters (must tweak in different tasks/environments)
distance_fitness_factor = .0001
connection_fitness_factor = 0 #.00001

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(0, 50), ylim=(0, 50))
ax.grid()
x = np.arange(0, 50, 0.1)
line, = ax.plot([], [], 'o-', lw=2)
food, = ax.plot([], [], '*r', lw=3)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

def init_animation():
    line.set_data([], [])
    time_text.set_text('')
    energy_text.set_text('')
    return line, time_text, energy_text

def animate(i):
    global agent_xpos, agent_ypos, food_xpos, food_ypox, tokenCount
    line.set_data(agent_xpos[i], agent_ypos[i])
    #print agent_xpos[i], agent_ypos[i]
    #plt.plot(food_xpos[i], food_ypos[i], marker='*', linestyle='')
    food.set_data(food_xpos[i], food_ypos[i])
    time_text.set_text('Tokens Eaten: %.0f' % tokenCount[i]) #
    #energy_text.set_text('food_y = %.1f' % food_ypos[i]) # orientation? dist traveled?
    return line, time_text, energy_text

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
    def __init__(self, experiment, iter=0, ITI=20, size=100, LFE=0):
        self.experiment = experiment
        self.iter = iter
        self.ITI = ITI
        self.size = size
        self.last_stim_loc = 0
        self.agents = []
        self.deadagents = []
        self.foodtokens = []
        self.last_food_eaten = LFE

    def check_boundary(self):
        for agent in self.agents:
            if agent.location[0] < 0: agent.location[0] = 0
            if agent.location[1] < 0: agent.location[1] = 0
            if agent.location[0] > self.size: agent.location[0] = self.size
            if agent.location[1] > self.size: agent.location[1] = self.size

    def add_agent(self, location, orientation, nnet=None, weights=None):
        self.agents.append(Agent(env=self, location=location, orientation=orientation, nnet=nnet, weights=weights))

    def add_foodtoken(self, location):
        self.foodtokens.append(FoodToken(location=location, iter_created=self.iter))

    def check_eaten_foodtokens(self):
        for agent in self.agents:
            for foodtoken in self.foodtokens:
                if abs(agent.location[0] - foodtoken.location[0]) < 3 and abs(agent.location[1] - foodtoken.location[1]) < 3:
                    self.foodtokens.remove(foodtoken)
                    self.last_food_eaten = self.iter
                    agent.eatenTokens += 1
                    agent.totalReward += max(0, 100 - (self.iter - foodtoken.iter_created))
        if self.iter - self.last_food_eaten == self.ITI:
            self.add_foodtoken(location=gen_rand_location(self.size, stimulus=True, lastloc=self.last_stim_loc))

    def update(self):
        self.iter += 1
        for agent in self.agents:
            agent.update()
        self.check_eaten_foodtokens()
        self.check_boundary()


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
        if self.lifecycle > 600:
            self.lifeOver = True
            self.fitness = self.eatenTokens + self.totalReward
        else:
            self.lifecycle += 1
        # We need to punish time spent between stim appearance and touching it! We need ISI

def gen_rand_location(size, stimulus=False, lastloc=0):
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

        if stimpos == 1:
            return([size*.25, size*.75])
        elif stimpos == 2:
            return([size*.75, size*.75])
        elif stimpos == 3:
            return([size*.25, size*.25])
        elif stimpos == 4:
            return([size*.75, size*.25])



def gen_rand_orientation():
    return(random.uniform(0,360))


def calc_dir_and_dist(dx, dy, orientation):
    rads = atan2(dy, dx)
    degs = ((degrees(rads) - 90) * -1) % 360
    degs = (degs - orientation) % 360
    dist = sqrt(dx ** 2 + dy ** 2)
    return(degs, dist)


def run_agent(exp, size=None, nnet=None, lifetime=600, weights=None, verbose=False, fname=""):
    env = Environment(experiment=exp, size=size)
    env.add_foodtoken(location=gen_rand_location(size))
    env.add_agent(location=gen_rand_location(size), orientation=gen_rand_orientation(), nnet=nnet, weights=weights)
    if verbose:
        global agent_xpos, agent_ypos, food_xpos, food_ypos, tokenCount
        agent_xpos = []
        agent_ypos = []
        food_xpos = []
        food_ypos = [] #env.foodtokens[0].location
        tokenCount = []
        lines_to_write = []
    for iter in range(lifetime):
        env.update()
        if verbose:
            agent_xpos.append(env.agents[0].location[0])
            agent_ypos.append(env.agents[0].location[1])
            food_xpos.append(env.foodtokens[0].location[0])
            food_ypos.append(env.foodtokens[0].location[1])
            tokenCount.append(env.agents[0].eatenTokens)
            lines_to_write.append( env.agents[0].location + env.foodtokens[0].location + [env.agents[0].orientation] ) # env.agents[0].orientation too?
    if verbose:
        dt = 1./60
        interval = 1000 * dt #- (t1 - t0)
        ani = animation.FuncAnimation(fig, animate, frames=600, interval=interval, blit=False, init_func=init_animation)
        # To save as an mp4 requires ffmpeg or mencoder to be installed.
        # The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this:
        # more info: http://matplotlib.sourceforge.net/api/animation_api.html
        ani.save('output/'+fname+'_agent_run.mp4', fps=30, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        #plt.show()
        with open('output/'+fname+'_agent_run.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile) # delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL
            writer.writerow(['agent_x','agent_y','food_x','food_y','orientation'])
            writer.writerows(lines_to_write)

    return (env.agents[0]) # full agent
