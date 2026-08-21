import os
import time
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed

from .agents import Agent
from .environment import Environment
from .evolution import select_and_reproduce
from .geometry import gen_rand_location, gen_rand_orientation
from .nn import FeedForwardNet
from .viz import RunRecorder, animate_run

# Preserves the original `run % 3` condition mapping from experiments.py
# (an accident of the original loop order: run%3==1 -> condition 1, etc.)
# Superseded by config-driven experiments in a later phase.
CONDITIONS = {
    0: dict(condition=2, hid_size=8),
    1: dict(condition=1, hid_size=4),
    2: dict(condition=3, hid_size=16),
}


class Experiment:
    def __init__(self, num_cores=1):
        self.num_cores = num_cores
        self.starttime = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.logfile = 'output/' + self.starttime

    def init_logfiles(self, filename=None):
        filename = (filename or self.logfile) + '.txt'
        with open(filename, 'w') as f:
            f.write('condition,gen,min,max,avg\n')

    def write_log(self, condition, current_generation, fitness, eatentokens, disttrav, time_elapsed, filename=None):
        filename = (filename or self.logfile) + '.txt'
        print(
            "Generation", current_generation,
            "-- Mean fitness:", '{0:.4f}'.format(np.mean(fitness)),
            "-- Mean tokens eaten:", '{0:.4f}'.format(np.mean(eatentokens)),
            '-- Mean distance traveled:', '{0:.4f}'.format(np.mean(disttrav)),
            '-- Elapsed time:', str(int(time_elapsed)), 's',
        )
        with open('output/textlog.txt', 'a') as f:
            f.write("Generation " + str(current_generation) + " Mean fitness: "
                     + '{0:.4f}'.format(np.mean(fitness)) + "\n")
        with open(filename, 'a') as f:
            f.write(','.join([
                str(condition), str(current_generation),
                str(min(fitness)), str(max(fitness)),
                '{0:.4f}'.format(np.mean(fitness)),
            ]) + '\n')


def run_agent(exp=None, size=100, nnet=None, n_hidden=8, lifetime=600, weights=None, verbose=False, fname=""):
    if nnet is None:
        nnet = FeedForwardNet(8, n_hidden, 2)
    env = Environment(experiment=exp, size=size)
    env.add_foodtoken(location=gen_rand_location(size))
    agent = Agent(env=env, location=gen_rand_location(size), orientation=gen_rand_orientation(),
                  nnet=nnet, weights=weights)
    env.add_agent(agent)

    recorder = RunRecorder() if verbose else None
    for _ in range(lifetime):
        env.update()
        if recorder is not None:
            recorder.record(env)

    if recorder is not None:
        os.makedirs('output', exist_ok=True)
        animate_run(recorder, env_size=size, out_path='output/' + fname + '_agent_run.mp4')
        recorder.to_csv('output/' + fname + '_agent_run.csv')

    return env.agents[0]


def run_evolution(num_agents=100, num_gens=250, lifetime=600, env_size=50, num_cores=None,
                   save_best_every=20, num_runs=30):
    """Truncation-selection evolution across `num_runs` runs, cycling through
    the three hidden-layer-size conditions in CONDITIONS."""
    exp = Experiment(num_cores=4)
    exp.init_logfiles()
    for run in range(num_runs):
        cond = CONDITIONS[run % 3]
        condition, hid_size = cond['condition'], cond['hid_size']
        newagents = None
        for gen in range(num_gens):
            start = time.time()
            if gen == 0:
                deadagents = Parallel(n_jobs=num_cores)(
                    delayed(run_agent)(exp=exp, size=env_size, n_hidden=hid_size, lifetime=lifetime)
                    for _ in range(num_agents)
                )
            else:
                deadagents = Parallel(n_jobs=num_cores)(
                    delayed(run_agent)(exp=exp, size=env_size, n_hidden=hid_size, lifetime=lifetime,
                                        weights=newagents[i].nnet.params)
                    for i in range(num_agents)
                )

            newagents, sortedagents = select_and_reproduce(deadagents, n_hidden=hid_size)

            if gen % save_best_every == 0:
                best_agent = sortedagents[0]
                run_agent(exp, size=env_size, n_hidden=hid_size, lifetime=lifetime,
                          weights=best_agent.nnet.params, verbose=True, fname="best_gen" + str(gen))

            fitness = [o.fitness for o in deadagents]
            eatentokens = [o.eatenTokens for o in deadagents]
            disttrav = [o.wheelDistanceTraveled for o in deadagents]
            time_elapsed = time.time() - start
            exp.write_log(condition, gen, fitness, eatentokens, disttrav, time_elapsed)
