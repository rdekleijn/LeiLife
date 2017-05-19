import experiments

# some default params
lifetime = 1000
env_size = 50

nbseq = [4,2,3,1,3,2,4,3,2,1]
simpseq = [1,2,3,4]

experiments.run_mult_experiments(lifetime=lifetime, env_size=env_size,
                                 num_cores=7, num_gens=300, sequence=simpseq)
#experiments.run_mult_experiments(lifetime=lifetime, env_size=env_size, num_cores=7, num_gens=300)

# want to automatically save the best agent every X generations, and be able
# to run them again on the environment and output an animation

#experiments.run_mult_experiments(lifetime=600, num_cores=7, num_gens=300)
