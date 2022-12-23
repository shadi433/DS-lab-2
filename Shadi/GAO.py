import random

# Perform crossover between two parents to produce two offspring
def crossover(parent1, parent2):
	crossover_point = random.randint(1, len(parent1)-1)
	offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
	offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
	return offspring1, offspring2

# Perform mutation on an individual by randomly changing one of its genes
def mutate(individual, intervals):
	index = random.randint(0, len(individual)-1)
	inrerval = intervals[index]
	individual[index] = random.uniform(inrerval[0], inrerval[1])
	return individual

# Select the fittest individuals from the population to survive to the next generation
def selection(population, fitnesses):
    fittest = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:int(len(population)/2)]
    return [individual for individual, fitness in fittest]

# Run the genetic algorithm to find the optimal solution
def GAO(fitness_function, pop, intervals):
	population = pop
	print(pop)
	fitnesses = [fitness_function(individual) for individual in population]
	population = selection(population, fitnesses)
	population = [crossover(random.choice(population), random.choice(population)) for _ in range(len(population))]
	pop1 = []
	for sublist in population:
		pop1.extend(sublist)
	population = [mutate(individual, intervals) for individual in pop1]
	fitnesses = [fitness_function(individual) for individual in population]
	
	print('finish with GAO')

	return population, fitnesses

