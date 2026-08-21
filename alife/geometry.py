import random
from math import atan2, degrees, sqrt


def gen_rand_location(size):
    return [random.uniform(0, size), random.uniform(0, size)]


def gen_rand_orientation():
    return random.uniform(0, 360)


def calc_dir_and_dist(dx, dy, orientation):
    """Bearing (degrees, relative to `orientation`) and distance to a point
    offset by (dx, dy)."""
    rads = atan2(dy, dx)
    degs = ((degrees(rads) - 90) * -1) % 360
    degs = (degs - orientation) % 360
    dist = sqrt(dx ** 2 + dy ** 2)
    return degs, dist
