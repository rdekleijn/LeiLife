import os
import time
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed

from .agents import Agent
from .environment import Environment
from .evolution import select_and_reproduce, form_groups
from .geometry import gen_rand_location, gen_rand_orientation
from .nn import FeedForwardNet
from .viz import RunRecorder, animate_run, GroupRunRecorder, animate_group_run

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
        os.makedirs('output', exist_ok=True)
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


def run_group(exp=None, size=100, agents=None, lifetime=600, num_foodtokens=None, verbose=False, fname=""):
    """Run a group of pre-built Agents together in one shared Environment
    for `lifetime` ticks (Phase 1: agents coexist and compete for the same
    food tokens, but don't yet sense each other -- Phase 2 adds an
    inter-agent signal channel on top of this scaffolding).

    Each agent's env/location/orientation is (re)initialized here, so the
    same Agent objects can be reused across trials by callers. Returns the
    same list of agents, each with fitness/eatenTokens/etc. filled in.
    """
    if num_foodtokens is None:
        num_foodtokens = len(agents)
    env = Environment(experiment=exp, size=size)
    for _ in range(num_foodtokens):
        env.add_foodtoken(location=gen_rand_location(size))
    for agent in agents:
        agent.env = env
        agent.location = gen_rand_location(size)
        agent.orientation = gen_rand_orientation()
        env.add_agent(agent)

    recorder = GroupRunRecorder() if verbose else None
    for _ in range(lifetime):
        env.update()
        if recorder is not None:
            recorder.record(env)

    if recorder is not None:
        os.makedirs('output', exist_ok=True)
        animate_group_run(recorder, env_size=size, out_path='output/' + fname + '_group_run.mp4')
        recorder.to_csv('output/' + fname + '_group_run.csv')

    return env.agents


def run_group_evolution(num_agents=100, num_gens=250, lifetime=600, env_size=50, group_size=5,
                         kin_based_groups=False, num_cores=None, save_best_every=20,
                         n_hidden=8, top_n=20, offspring_per=5, mutation_std=0.3):
    """Same truncation-selection evolution as `run_evolution`, but each
    trial runs a shared-environment group of `group_size` agents rather
    than one agent alone, so there's someone for a future communication
    channel (Phase 2) to talk to.

    `group_size=offspring_per` (both default to 5) by default, so that
    kin-based grouping (`kin_based_groups=True`) places exact sibling sets
    -- all offspring of one parent -- together each generation.
    """
    exp = Experiment(num_cores=4)
    exp.init_logfiles()

    agents = [Agent(nnet=FeedForwardNet(8, n_hidden, 2), lineage_id=i) for i in range(num_agents)]

    for gen in range(num_gens):
        start = time.time()
        groups = form_groups(agents, group_size, kin_based=kin_based_groups)
        for group_id, group in enumerate(groups):
            for agent in group:
                agent.group_id = group_id

        results = Parallel(n_jobs=num_cores)(
            delayed(run_group)(exp=exp, size=env_size, agents=group, lifetime=lifetime)
            for group in groups
        )
        deadagents = [agent for group_result in results for agent in group_result]

        offspring, sortedagents = select_and_reproduce(
            deadagents, n_hidden=n_hidden, top_n=top_n, offspring_per=offspring_per, mutation_std=mutation_std,
        )

        if gen % save_best_every == 0:
            replay_group = [
                Agent(nnet=FeedForwardNet(8, n_hidden, 2), weights=a.nnet.params, lineage_id=a.lineage_id)
                for a in sortedagents[:group_size]
            ]
            run_group(exp, size=env_size, agents=replay_group, lifetime=lifetime,
                      verbose=True, fname="best_group_gen" + str(gen))

        fitness = [a.fitness for a in deadagents]
        eatentokens = [a.eatenTokens for a in deadagents]
        disttrav = [a.wheelDistanceTraveled for a in deadagents]
        time_elapsed = time.time() - start
        exp.write_log("group", gen, fitness, eatentokens, disttrav, time_elapsed)

        agents = offspring

    return agents
