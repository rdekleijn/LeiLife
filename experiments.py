from alife import *
import time
from joblib import Parallel, delayed


def run_experiment(condition=1, num_agents=100, num_gens=250, lifetime=600, env_size=100, num_cores=None):
    exp = Experiment(num_cores=4)
    exp.init_logfiles()
    for current_generation in range(num_gens):
        start = time.time()
        if current_generation == 0:
            deadagents = Parallel(n_jobs=num_cores)(delayed(run_agent)(lifetime=lifetime) for agent in range(num_agents))
        else:
            deadagents = Parallel(n_jobs=num_cores)(delayed(run_agent)(lifetime=lifetime, weights=newagents[agent].nnet.params) for agent in range(num_agents))
        sortedagents = sorted(deadagents, key=lambda x: x.eatenTokens, reverse=True)
        fitness = [o.eatenTokens for o in deadagents]
        time_elapsed = time.time() - start
        log_and_output(condition, current_generation, fitness, time_elapsed)
        newagents = []
        for agent in sortedagents[0:20]:
            for i in range(5):
                newagents.append(Agent(weights=agent.nnet.params + np.random.normal(0, .3, 144)))


def run_mult_experiments(condition=1, num_agents=100, num_gens=250, lifetime=600, env_size=100, num_cores=None):
    exp = Experiment(num_cores=2)
    exp.init_logfiles()
    for run in range(30):
        if run%3 == 0:
            condition = 1
            hid_size = 4
            num_units = 76
        elif run%3 == 1:
            condition = 2
            hid_size = 8
            num_units = 144
        else:
            condition = 2
            hid_size = 16
            num_units = 320
        for current_generation in range(num_gens):
            start = time.time()
            if current_generation == 0:
                deadagents = Parallel(n_jobs=num_cores)(delayed(run_agent)(exp=exp, nnet=buildNetwork(8, hid_size, 8), lifetime=lifetime) for agent in range(num_agents))
            else:
                deadagents = Parallel(n_jobs=num_cores)(delayed(run_agent)(exp=exp, nnet=buildNetwork(8, hid_size, 8), lifetime=lifetime, weights=newagents[agent].nnet.params) for agent in range(num_agents))
            sortedagents = sorted(deadagents, key=lambda x: x.eatenTokens, reverse=True)
            fitness = [o.eatenTokens for o in deadagents]
            time_elapsed = time.time() - start
            exp.write_log(condition, current_generation, fitness, time_elapsed)
            newagents = []
            for agent in sortedagents[0:20]:
                for i in range(5):
                    newagents.append(Agent(nnet=buildNetwork(8, hid_size, 8),weights=agent.nnet.params + np.random.normal(0, .3, num_units)))