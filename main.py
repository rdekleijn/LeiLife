import experiments

# some default params
lifetime = 500
env_size = 50

nbseq = [4,2,3,1,3,2,4,3,2,1]
simpseq = [1,2,3,4]

experiments.run_experiment(lifetime=lifetime, env_size=env_size,
                                 num_cores=4, num_gens=1000, sequence=simpseq, load_last_agent=True)
