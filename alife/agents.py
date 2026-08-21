from math import radians, sin, cos, floor, ceil, sqrt

import numpy as np

from .geometry import calc_dir_and_dist
from .nn import FeedForwardNet

# fitness function parameters (must tweak in different tasks/environments)
DISTANCE_FITNESS_FACTOR = .0001
CONNECTION_FITNESS_FACTOR = 0  # .00001


class Agent:
    def __init__(self, env=None, location=None, orientation=None, nnet=None, weights=None):
        self.location = location
        self.orientation = orientation
        self.lifecycle = 0
        self.lifeOver = False
        self.eatenTokens = 0
        self.wheelDistanceTraveled = 0
        self.fitness = None
        self.env = env
        self.visual_input = np.zeros(8)
        self.motor_output = [0.0, 0.0]
        self.nnet = nnet if nnet is not None else FeedForwardNet(8, 8, 2)
        if weights is not None:
            self.nnet.set_params(weights)

    def give_status(self):
        print('X location =', self.location[0], '\n', 'Y location =', self.location[1])

    def update_visual_field(self):
        visual = np.zeros(8)
        for foodtoken in self.env.foodtokens:
            direction, dist = calc_dir_and_dist(
                foodtoken.location[0] - self.location[0],
                foodtoken.location[1] - self.location[1],
                self.orientation,
            )
            prop_dist = dist / sqrt(2 * (self.env.size ** 2))
            dir_reduced = direction / 45
            sector = np.zeros(8)
            sector[int(floor(dir_reduced)) % 8] = 1 - (dir_reduced - floor(dir_reduced))
            if dir_reduced != int(dir_reduced):
                sector[int(ceil(dir_reduced)) % 8] = dir_reduced - floor(dir_reduced)
            visual += sector * (1 - prop_dist)
        self.visual_input = visual + np.random.normal(0, .1, 8)

    def cycle_nnet(self):
        return self.nnet.activate(self.visual_input)

    def move(self):
        unitsAxisWidth = 0.2
        self.motor_output = np.clip(self.cycle_nnet(), -2, 2)
        left_act = self.motor_output[0]
        right_act = self.motor_output[1]
        self.wheelDistanceTraveled += abs(left_act) + abs(right_act)

        if abs(left_act - right_act) < .0001:
            v = (left_act + right_act) / 2
            self.location = [
                self.location[0] + v * sin(radians(self.orientation)),
                self.location[1] + v * cos(radians(self.orientation)),
            ]
        else:
            R = unitsAxisWidth * (left_act + right_act) / (2 * (right_act - left_act))
            wd = (right_act - left_act) / unitsAxisWidth
            self.location = [
                self.location[0] + R * cos(radians(wd + self.orientation)) - R * cos(radians(self.orientation)),
                self.location[1] - R * sin(radians(wd + self.orientation)) + R * sin(radians(self.orientation)),
            ]
            self.orientation = (self.orientation + wd) % 360

    def update(self):
        self.update_visual_field()
        self.move()
        if self.lifecycle > 600:
            self.lifeOver = True
            self.fitness = (self.eatenTokens
                             - DISTANCE_FITNESS_FACTOR * self.wheelDistanceTraveled
                             - CONNECTION_FITNESS_FACTOR * np.sum(np.abs(self.nnet.params)))
        else:
            self.lifecycle += 1
