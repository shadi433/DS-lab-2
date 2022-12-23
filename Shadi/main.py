import random
import math

def levy_flight(x, beta):
    """Performs a Levy flight move from the current position x."""
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = random.normalvariate(0, sigma)
    v = random.normalvariate(0, sigma)
    step = u / abs(v) ** (1 / beta)
    return x + step

def cuckoo_search(nest, cost_function, pa, beta=1.5, num_iterations=100, population_size=10):
    """Performs the Cuckoo Search algorithm to find the minimum of the cost function.
    
    Args:
        nest: List of current positions (solutions) of the cuckoos.
        cost_function: Function to minimize.
        pa: Probability of abandoning a nest.
        beta: Levy flight exponent.
        num_iterations: Number of iterations to run the algorithm.
        population_size: Number of cuckoos in the population.
        
    Returns:
        Tuple (nest, fitnesses) containing the final list of positions and the fitness values of each position according to the cost function.
    """
    fitnesses = []
    for i in range(num_iterations):
        new_nest = []
        for j in range(len(nest)):
            if random.random() > pa:
                new_nest.append(levy_flight(nest[j], beta))
            else:
                new_nest.append(random.uniform(-1, 1))
        nest.extend(new_nest)
        nest = sorted(nest, key=cost_function)
        nest = nest[:population_size]  # Keep only the specified number of cuckoos
        fitnesses = [cost_function(pos) for pos in nest]
    return nest, fitnesses

# Example usage
def cost_function(x):
    return x ** 2

nest, fitnesses = cuckoo_search([random.uniform(-1, 1) for _ in range(10)], cost_function, 0.25, population_size=20)
print(nest)
print(fitnesses)
