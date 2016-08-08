from alife import *


def run_experiment(condition=1, num_agents=100, num_gens=250, lifetime=600, env_size=100, num_cores=None):
    num_cores = 8
    init_logfiles()
    for current_generation in range(num_gens):
        start = time.time()
        if current_generation > 0:
            deadagents = Parallel(n_jobs=num_cores)(delayed(run_agent)(lifetime=lifetime, weights=newagents[agent].nnet.params) for agent in range(num_agents))
        else:
            deadagents = Parallel(n_jobs=num_cores)(delayed(run_agent)(lifetime=lifetime) for agent in range(num_agents))
        sortedagents = sorted(deadagents, key=lambda x: x.eatenTokens, reverse=True)
        fitness = [o.eatenTokens for o in deadagents]
        time_elapsed = time.time() - start
        log_and_output(condition, current_generation, fitness, time_elapsed)
        newagents = []
        for agent in sortedagents[0:20]:
            for i in range(5):
                newagents.append(Agent(weights=agent.nnet.params + np.random.normal(0, .3, 144)))


def run_mult_experiments(condition=1, num_agents=100, num_gens=250, lifetime=600, env_size=100, num_cores=None):
    num_cores = 8
    init_logfiles()

    for exp in range(10):
        if exp < 5:
            condition = 1
            mut_par = 0.2
        else:
            condition = 2
            mut_par = 0.4
        for current_generation in range(num_gens):
            start = time.time()
            if current_generation > 0:
                deadagents = Parallel(n_jobs=num_cores)(delayed(run_agent)(lifetime=lifetime, weights=newagents[agent].nnet.params) for agent in range(num_agents))
            else:
                deadagents = Parallel(n_jobs=num_cores)(delayed(run_agent)(lifetime=lifetime) for agent in range(num_agents))
            sortedagents = sorted(deadagents, key=lambda x: x.eatenTokens, reverse=True)
            fitness = [o.eatenTokens for o in deadagents]
            time_elapsed = time.time() - start
            log_and_output(condition, current_generation, fitness, time_elapsed)
            newagents = []
            for agent in sortedagents[0:20]:
                for i in range(5):
                    newagents.append(Agent(weights=agent.nnet.params + np.random.normal(0, mut_par, 144)))