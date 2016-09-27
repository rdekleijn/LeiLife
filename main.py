import alife
import experiments


experiments.run_mult_experiments(lifetime=1000, num_cores=2, num_gens=50)

# want to automatically save the best agent every X generations, and be able
# to run them again on the environment and output an animation

#experiments.run_mult_experiments(lifetime=600, num_cores=7, num_gens=300)
