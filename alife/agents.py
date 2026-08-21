from math import radians, sin, cos, floor, ceil, sqrt

import numpy as np

from .geometry import calc_dir_and_dist
from .nn import FeedForwardNet

# fitness function parameters (must tweak in different tasks/environments)
DISTANCE_FITNESS_FACTOR = .0001
CONNECTION_FITNESS_FACTOR = 0  # .00001
SIGNAL_FITNESS_FACTOR = .0001  # metabolic cost per unit of signal emitted

N_FOOD_SECTORS = 8
N_SIGNAL_SECTORS = 8
N_MOTOR_OUTPUTS = 2  # left wheel, right wheel


def net_dims(signaling):
    """(n_in, n_out) for an Agent's controller net. With signaling, the net
    gains an 8-sector "hear other agents' signals" input channel (mirroring
    the food-direction sectors) and a 3rd motor output used as a broadcast
    signal strength."""
    n_in = N_FOOD_SECTORS + (N_SIGNAL_SECTORS if signaling else 0)
    n_out = N_MOTOR_OUTPUTS + (1 if signaling else 0)
    return n_in, n_out


def _bearing_sectors(direction, prop_dist, weight=1.0):
    """8-sector one-hot(-ish) encoding of a bearing, scaled by (1 - prop_dist)
    and `weight`. Shared by food- and signal-sensing, which differ only in
    what `weight` means (implicitly 1 for a food token; another agent's
    emitted_signal strength for signal-sensing)."""
    sector = np.zeros(8)
    dir_reduced = direction / 45
    sector[int(floor(dir_reduced)) % 8] = 1 - (dir_reduced - floor(dir_reduced))
    if dir_reduced != int(dir_reduced):
        sector[int(ceil(dir_reduced)) % 8] = dir_reduced - floor(dir_reduced)
    return sector * (1 - prop_dist) * weight


class Agent:
    def __init__(self, env=None, location=None, orientation=None, nnet=None, weights=None,
                 lineage_id=None, group_id=None, signaling=False):
        self.location = location
        self.orientation = orientation
        self.lifecycle = 0
        self.lifeOver = False
        self.eatenTokens = 0
        self.wheelDistanceTraveled = 0
        self.fitness = None
        self.env = env
        self.visual_input = np.zeros(N_FOOD_SECTORS)
        self.signal_input = np.zeros(N_SIGNAL_SECTORS)
        self.motor_output = [0.0, 0.0]
        self.signaling = signaling
        # emitted_signal is this tick's broadcast strength, sensed by other
        # agents starting next tick (see update_signal_field: within one
        # Environment.update() tick, agents earlier in env.agents will have
        # already recomputed emitted_signal by the time a later agent senses
        # them, so "this tick's" vs. "last tick's" value depends on list
        # position -- a deliberate simplification for this first signaling
        # pass; see docs/facelift-design.md Phase 2 notes).
        self.emitted_signal = 0.0
        self.signalEmitted = 0.0  # cumulative |signal| over the agent's lifetime, for fitness cost
        if nnet is not None:
            self.nnet = nnet
        else:
            n_in, n_out = net_dims(signaling)
            self.nnet = FeedForwardNet(n_in, 8, n_out)
        if weights is not None:
            self.nnet.set_params(weights)
        # lineage_id identifies siblings from the same reproduction event
        # (see evolution.select_and_reproduce); group_id identifies which
        # shared-environment trial an agent was run in (see
        # evolution.form_groups / experiment.run_group). Both are metadata
        # only -- they don't affect simulation dynamics.
        self.lineage_id = lineage_id
        self.group_id = group_id

    def give_status(self):
        print('X location =', self.location[0], '\n', 'Y location =', self.location[1])

    def update_visual_field(self):
        visual = np.zeros(N_FOOD_SECTORS)
        for foodtoken in self.env.foodtokens:
            direction, dist = calc_dir_and_dist(
                foodtoken.location[0] - self.location[0],
                foodtoken.location[1] - self.location[1],
                self.orientation,
            )
            prop_dist = dist / sqrt(2 * (self.env.size ** 2))
            visual += _bearing_sectors(direction, prop_dist)
        self.visual_input = visual + np.random.normal(0, .1, N_FOOD_SECTORS)

    def update_signal_field(self):
        signal = np.zeros(N_SIGNAL_SECTORS)
        for other in self.env.agents:
            if other is self or other.emitted_signal <= 0:
                continue
            direction, dist = calc_dir_and_dist(
                other.location[0] - self.location[0],
                other.location[1] - self.location[1],
                self.orientation,
            )
            prop_dist = dist / sqrt(2 * (self.env.size ** 2))
            signal += _bearing_sectors(direction, prop_dist, weight=other.emitted_signal)
        self.signal_input = signal + np.random.normal(0, .1, N_SIGNAL_SECTORS)

    def cycle_nnet(self):
        if self.signaling:
            sensory_input = np.concatenate([self.visual_input, self.signal_input])
        else:
            sensory_input = self.visual_input
        return self.nnet.activate(sensory_input)

    def move(self):
        unitsAxisWidth = 0.2
        raw_output = self.cycle_nnet()
        self.motor_output = np.clip(raw_output[:N_MOTOR_OUTPUTS], -2, 2)
        if self.signaling:
            self.emitted_signal = float(np.clip(raw_output[N_MOTOR_OUTPUTS], 0, 2))
            self.signalEmitted += self.emitted_signal
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
        if self.signaling:
            self.update_signal_field()
        self.move()
        if self.lifecycle > 600:
            self.lifeOver = True
            self.fitness = (self.eatenTokens
                             - DISTANCE_FITNESS_FACTOR * self.wheelDistanceTraveled
                             - CONNECTION_FITNESS_FACTOR * np.sum(np.abs(self.nnet.params))
                             - SIGNAL_FITNESS_FACTOR * self.signalEmitted)
        else:
            self.lifecycle += 1
