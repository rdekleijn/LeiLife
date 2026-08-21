from alife.experiment import run_evolution

run_evolution(lifetime=1000, num_cores=7, num_gens=300)

# want to automatically save the best agent every X generations, and be able
# to run them again on the environment and output an animation
