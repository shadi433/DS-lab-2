import random
import math

def levy_flight(position, beta, intervals):
    """Performs a Levy flight move from the current position."""
    # Calculate sigma value using the formula specified in the algorithm
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    # Number of variables
    dimensions = len(position) 
    # Steps to take in each dimension
    steps = [] 
    for i in range(dimensions):
        # Generate random values from normal distribution with mean 0 and standard deviation sigma
        u = random.normalvariate(0, sigma)
        v = random.normalvariate(0, sigma)
        # Calculate step in this dimension using the formula specified in the algorithm
        step = u / abs(v) ** (1 / beta)
        steps.append(step)
    # Return new position by adding the steps to the current position and make sure that each value did not pass thier interval
    new_position = []
    for x, step, interval in zip(position, steps, intervals):
        new_x = x + step
        if new_x < interval[0]:
            new_x = interval[0]
        elif new_x > interval[1]:
            new_x = interval[1]
        new_position.append(new_x)

    return new_position

def CSO(fitness_function, nest, n_pop, intervals, pa=0.25, beta=1.5):
    """
    Performs the Cuckoo Search algorithm to find the minimum of the cost function.
    
    Args:
        fitness_function: Function to minimize.
        nest: List of current positions (solutions) of the cuckoos.
        n_pop: Number of cuckoos in the population.
        intervals: List of lists of ranges for each variable.
        pa: Probability of abandoning a nest.
        beta: Levy flight exponent. 

    """
    fitnesses = []
    # Generate new position for the cuckoos
    new_nest = []
    for j in range(len(nest)):
        # Perform a Levy flight with probability (1 - pa)
        if random.random() > pa: 
            new_nest.append(levy_flight(nest[j], beta))
        # Abandon the old position and generate a new random position with probability pa
        else: 
            new_position = [random.uniform(interval[0], interval[1]) for interval in intervals]
            new_nest.append(new_position)
    # Add the new positions to the list of positions
    nest.extend(new_nest)
    # Sort the positions by their fitness (as determined by the cost function
    nest = sorted(nest, key=fitness_function, reverse=True)
    # Keep only the specified number of cuckoos
    nest = nest[:n_pop]  
    fitnesses = [fitness_function(pos) for pos in nest]

    print('finish with CSO')

    return nest, fitnesses