import random
#This code defines a IWO() function that takes in several parameters:

def IWO(dim, fitness_function, n_pop, pop, rinitial, rfinal, modulation_index, itermax, iter):
    '''
    dim: The dimensionality of the search space
    fitness_function: The objective function to be optimized
    n_pop: The maximum number of weeds in the population
    rinitial: The initial value of the variance parameter
    rfinal: The final value of the variance parameter
    modulation_index: The non-linear modulation index
    itermax: The maximum number of iterations
    # Initialize the population of weeds
    '''

    population = pop

    # Evaluate the fitness of each weed
    fitness = [fitness_function(weed) for weed in population]

    # perform reproduction, spatial distribution, and competitive exclusion
    # Allow each weed to reproduce by generating a number of seeds based on its own fitness
    seeds = []
    for i, weed in enumerate(population):
        num_seeds = calculate_num_seeds(fitness[i], fitness)
        for j in range(num_seeds):
            seeds.append(weed)

    # Distribute the seeds over the search space using normally distributed random numbers
    for i, seed in enumerate(seeds):
        # Calculate the standard deviation for this time step
        r = (itermax - iter) ** modulation_index * (rinitial - rfinal) / itermax ** modulation_index + rfinal
        # Generate normally distributed random numbers
        displacement = [random.normalvariate(0, r) for _ in range(dim)]
        # Update the position of the seed
        seeds[i] = [x + y for x, y in zip(seed, displacement)]

    #  Perform competitive exclusion if the number of weeds exceeds the population size
    if len(seeds) > n_pop:
        # Combine the seeds and the original population
        population += seeds
        # Calculate the fitness of each plant
        fitness = [fitness_function(plant) for plant in population]
        # Sort the plants by fitness
        population = [x for _, x in sorted(zip(fitness, population), reverse=True)]
        # Keep only the top n_pop plants
        population = population[:n_pop]
        fitness = fitness[:n_pop]

    print('finish with IWO')

    return population, fitness

def calculate_num_seeds(fitness, fitness_values):
    # Calculate the number of seeds produced by a weed based on its own fitness
    # and the fitness of the other weeds in the population
    min_fitness = min(fitness_values)
    max_fitness = max(fitness_values)
    num_seeds = (fitness - min_fitness) / (max_fitness - min_fitness)
    return num_seeds