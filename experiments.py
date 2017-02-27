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


def run_mult_experiments(num_agents=100, num_gens=250, lifetime=600,
                         env_size=50, num_cores=None, save_best_Ngens=20, sequence=[]):
    exp = Experiment(num_cores=7)
    exp.init_logfiles()
    for run in range(30): # what's this 30?
        if run%3 == 1:
            condition = 1
            hid_size = 4
            num_units = 30
        elif run%3 == 0:
            condition = 2
            hid_size = 8
            num_units = 58
        else:
            condition = 3
            hid_size = 16
            num_units = 114
        for current_generation in range(num_gens):
            start = time.time()
            if current_generation == 0:
                deadagents = Parallel(n_jobs=num_cores)(delayed(run_agent)(exp=exp, size=env_size, nnet=buildNetwork(4, hid_size, 2), lifetime=lifetime, sequence=sequence) for agent in range(num_agents))
            else:
                deadagents = Parallel(n_jobs=num_cores)(delayed(run_agent)(exp=exp, size=env_size, nnet=buildNetwork(4, hid_size, 2), lifetime=lifetime, weights=newagents[agent].nnet.params, sequence=sequence) for agent in range(num_agents))
            sortedagents = sorted(deadagents, key=lambda x: x.fitness, reverse=True)

            if current_generation%save_best_Ngens==0:
                # now run the best agent again and save/animate the output:
                best_agent = sortedagents[0] # should also save the agent!
                run_agent(exp, size=env_size, nnet=best_agent.nnet, lifetime=lifetime,
                          weights=best_agent.nnet.params, verbose=True, fname="best_gen"+str(current_generation))

            fitness = [o.fitness for o in deadagents]
            eatentokens = [o.eatenTokens for o in deadagents]
            timecenter = [o.steps_in_center for o in deadagents]
            disttrav = [o.wheelDistanceTraveled for o in deadagents]
            time_elapsed = time.time() - start
            exp.write_log(condition, current_generation, fitness, eatentokens, disttrav, time_elapsed, timecenter)
            newagents = []
            for agent in sortedagents[0:20]:
                for i in range(5):
                    newagents.append(Agent(nnet=buildNetwork(8, hid_size, 2),weights=agent.nnet.params + np.random.normal(0, .3, num_units)))
