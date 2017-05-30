from alife import *
import time
import pickle
from joblib import Parallel, delayed
import glob
import os


# penalty for distance traveled
def custom_fitness(agent):
    # afit = agent.eatenTokens - agent.wheelDistanceTraveled*distance_fitness_factor
    afit = agent.eatenTokens + agent.totalReward - agent.wheelDistanceTraveled*distance_fitness_factor
    # afit = agent.eatenTokens

    return afit

def save_agent(agent):
    filename = 'output/agent' + datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filehandler = open(filename, 'w')
    pickle.dump(agent, filehandler)

def load_agent():
    search_dir = "output/"
    files = filter(os.path.isfile, glob.glob(search_dir + "agent*"))
    files.sort(key=lambda x: os.path.getmtime(x))
    print files
    filehandler = open(files[len(files)-1], 'r')
    agent = pickle.load(filehandler)
    return agent

def run_experiment(num_agents=100, num_gens=250, lifetime=600,
                         env_size=50, num_cores=None, save_best_Ngens=20, sequence=[], load_last_agent=False):
    exp = Experiment(num_cores=num_cores, lifetime=lifetime)
    exp.init_logfiles()
    for run in range(1):
        condition = 1
        hid_size = 8
        num_units = 58
        for current_generation in range(num_gens):
            start = time.time()
            if current_generation == 0:
                if load_last_agent is True:
                    deadagents = [load_agent()] * num_agents
                else:
                    deadagents = Parallel(n_jobs=num_cores)(delayed(run_agent)(exp=exp, size=env_size, nnet=buildNetwork(4, hid_size, 2), lifetime=lifetime, sequence=sequence) for agent in range(num_agents))
            else:
                deadagents = Parallel(n_jobs=num_cores)(delayed(run_agent)(exp=exp, size=env_size, nnet=buildNetwork(4, hid_size, 2), lifetime=lifetime, weights=newagents[agent].nnet.params, sequence=sequence) for agent in range(num_agents))


            #fitness = [o.fitness for o in deadagents]
            # fitness = [custom_fitness(o) for o in deadagents]
            for o in deadagents:
                o.fitness = custom_fitness(o)
            fitness = [o.fitness for o in deadagents]
            eatentokens = [o.eatenTokens for o in deadagents]
            timecenter = [o.steps_in_center for o in deadagents]
            disttrav = [o.wheelDistanceTraveled for o in deadagents]
            time_elapsed = time.time() - start
            exp.write_log(condition, current_generation, fitness, eatentokens, disttrav, time_elapsed, timecenter)
            sortedagents = sorted(deadagents, key=lambda x: x.fitness, reverse=True)
            if current_generation%save_best_Ngens==0 and current_generation > 0:
                # now run the best agent again and save/animate the output:
                best_agent = sortedagents[0] # should also save the agent!
                run_agent(exp, size=env_size, nnet=best_agent.nnet, lifetime=lifetime,
                          weights=best_agent.nnet.params, sequence=sequence, verbose=True, fname="best_gen"+str(current_generation))
                save_agent(best_agent)
            newagents = []
            for agent in sortedagents[0:20]:
                for i in range(5):
                    newagents.append(Agent(nnet=buildNetwork(8, hid_size, 2),weights=agent.nnet.params + np.random.normal(0, .3, num_units)))
