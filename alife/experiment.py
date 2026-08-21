import os
import random
import time
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed

from .agents import Agent, net_dims, bump_agent_id_counter
from .checkpoint import save_checkpoint, load_checkpoint
from .environment import Environment
from .evolution import select_and_reproduce, form_groups
from .geometry import gen_rand_location, gen_rand_orientation
from .logbook import GenerationLog
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
    def __init__(self, num_cores=1, output_dir='output'):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.num_cores = num_cores
        self.starttime = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.logfile = os.path.join(output_dir, self.starttime)

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
        with open(os.path.join(self.output_dir, 'textlog.txt'), 'a') as f:
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


def run_group(exp=None, size=100, agents=None, lifetime=600, num_foodtokens=None, verbose=False, fname="",
              seed=None, output_dir='output'):
    """Run a group of pre-built Agents together in one shared Environment
    for `lifetime` ticks. Agents coexist and compete for the same food
    tokens; whether they also sense/emit signals is entirely determined by
    each Agent's own `.signaling` flag (set at construction -- see
    `agents.Agent` and `evolution.select_and_reproduce`), so this function
    doesn't need to know about signaling itself.

    Each agent's env/location/orientation is (re)initialized here, so the
    same Agent objects can be reused across trials by callers. Returns the
    same list of agents, each with fitness/eatenTokens/etc. filled in.

    `seed`, if given, reseeds the global `random`/`np.random` state at the
    top of this call -- needed for reproducibility under joblib's default
    process-based parallelism, where a worker's RNG state isn't a
    deterministic function of the parent process's (see
    run_group_evolution's `seed` docs).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
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
        os.makedirs(output_dir, exist_ok=True)
        animate_group_run(recorder, env_size=size, out_path=os.path.join(output_dir, fname + '_group_run.mp4'))
        recorder.to_csv(os.path.join(output_dir, fname + '_group_run.csv'))

    return env.agents


def run_group_evolution(num_agents=100, num_gens=250, lifetime=600, env_size=50, group_size=5,
                         kin_based_groups=False, num_cores=None, save_best_every=20,
                         n_hidden=8, top_n=20, offspring_per=5, mutation_std=0.3, signaling=False,
                         social_learning=False, seed=None, output_dir='output',
                         checkpoint_every=None, resume_from=None, save_all_groups_every=None):
    """Same truncation-selection evolution as `run_evolution`, but each
    trial runs a shared-environment group of `group_size` agents rather
    than one agent alone, so there's someone for a communication channel
    to talk to.

    `group_size=offspring_per` (both default to 5) by default, so that
    kin-based grouping (`kin_based_groups=True`) places exact sibling sets
    -- all offspring of one parent -- together each generation.

    `signaling=True` gives agents an extra sensory channel (hearing nearby
    agents' signals) and motor output (a broadcast signal strength) on top
    of the base food-foraging net -- see `agents.net_dims`.

    `social_learning=True` (requires `signaling=True`) additionally gives
    agents a within-lifetime, non-evolved learning mechanism: hearing a
    neighbor's signal shortly before that neighbor eats nudges this
    agent's own response toward imitating the neighbor's action (see
    `Agent.observe_and_learn`). Unlike the evolved net, this learned state
    is never inherited -- only the capacity to learn is.

    `seed`, if given, makes group formation and mutation (which happen in
    this process) reproducible, and is also used to derive a distinct,
    deterministic per-group-trial seed for each parallel `run_group` call
    -- so the whole run is reproducible regardless of joblib's backend,
    *provided* `num_cores=1`. With real parallelism, worker scheduling can
    still interleave float summation order etc. differently run to run;
    this makes each individual trial reproducible, not necessarily the
    aggregate down to the last bit.

    `output_dir` is where the run's logs, checkpoints, and any verbose
    replays are written. A `generation_log.csv` (one row per agent per
    generation -- see `logbook.GenerationLog`) is always written there.

    `checkpoint_every`, if given, pickles the population there every N
    generations (see `checkpoint.save_checkpoint`); `resume_from`, if
    given a checkpoint path, resumes from it instead of starting fresh at
    generation 0 (also restores RNG state -- see `checkpoint.load_checkpoint`).

    `save_all_groups_every`, if given, additionally saves a verbose replay
    (CSV + mp4) of *every* group (not just the single best-fitness group
    `save_best_every` already saves) every N generations -- fresh trials of
    that generation's evolved genomes, not literal replays of the
    evaluation episode (same caveat as `save_best_every`'s replay). Useful
    for cross-group "dialect" comparison -- see analysis/signal_analysis.py.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    exp = Experiment(num_cores=num_cores, output_dir=output_dir)
    exp.init_logfiles()
    generation_log = GenerationLog(os.path.join(output_dir, 'generation_log.csv'))

    n_in, n_out = net_dims(signaling)
    start_gen = 0
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from)
        agents = checkpoint['agents']
        start_gen = checkpoint['gen'] + 1
        bump_agent_id_counter(max(a.agent_id for a in agents))
    else:
        agents = [Agent(nnet=FeedForwardNet(n_in, n_hidden, n_out), lineage_id=i, signaling=signaling,
                         social_learning=social_learning)
                  for i in range(num_agents)]

    for gen in range(start_gen, num_gens):
        start = time.time()
        groups = form_groups(agents, group_size, kin_based=kin_based_groups)
        for group_id, group in enumerate(groups):
            for agent in group:
                agent.group_id = group_id

        results = Parallel(n_jobs=num_cores)(
            delayed(run_group)(exp=exp, size=env_size, agents=group, lifetime=lifetime,
                                seed=(None if seed is None else seed + gen * 1_000_000 + i))
            for i, group in enumerate(groups)
        )
        deadagents = [agent for group_result in results for agent in group_result]
        generation_log.append(gen, deadagents)

        offspring, sortedagents = select_and_reproduce(
            deadagents, n_hidden=n_hidden, top_n=top_n, offspring_per=offspring_per, mutation_std=mutation_std,
            signaling=signaling, social_learning=social_learning,
        )

        if gen % save_best_every == 0:
            replay_group = [
                Agent(nnet=FeedForwardNet(n_in, n_hidden, n_out), weights=a.nnet.params,
                      lineage_id=a.lineage_id, signaling=signaling, social_learning=social_learning)
                for a in sortedagents[:group_size]
            ]
            run_group(exp, size=env_size, agents=replay_group, lifetime=lifetime,
                      verbose=True, fname="best_group_gen" + str(gen), output_dir=output_dir)

        if save_all_groups_every and gen % save_all_groups_every == 0:
            for gi, group in enumerate(groups):
                run_group(exp, size=env_size, agents=group, lifetime=lifetime,
                          verbose=True, fname=f"gen{gen}_group{gi}", output_dir=output_dir)

        fitness = [a.fitness for a in deadagents]
        eatentokens = [a.eatenTokens for a in deadagents]
        disttrav = [a.wheelDistanceTraveled for a in deadagents]
        time_elapsed = time.time() - start
        exp.write_log("group", gen, fitness, eatentokens, disttrav, time_elapsed)

        agents = offspring

        if checkpoint_every and (gen + 1) % checkpoint_every == 0:
            save_checkpoint(os.path.join(output_dir, 'checkpoint.pkl'), gen, agents)

    return agents
