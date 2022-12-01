from numpy.random import randint
from numpy.random import rand
import matplotlib.pyplot as plt


# for ploting

worst_fit_v = []
best_fit_v = []
avg_fit_v = []
# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		# extract the substring
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		# convert bitstring to a string of chars
		chars = ''.join([str(s) for s in substring])
		# convert string to integer
		integer = int(chars, 2)
		# scale integer to desired range
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		# store
		decoded.append(value)
	return decoded

# tournament selection
def selection(pop, Fitness, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if Fitness[ix] > Fitness[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

# genetic algorithm
def GAO(fitness_function, bounds, n_bits, Generations, n_pop, r_cross, r_mut):
	# initial population of random bitstring
	pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
	# keep track of best solution
	best_para, best_fit = 0, fitness_function(decode(bounds, n_bits, pop[0]))
	# enumerate generations
	for generation in range(Generations):
		# decode population
		decoded = [decode(bounds, n_bits, p) for p in pop]
		# evaluate all candidates in the population
		Fitness = [fitness_function(d) for d in decoded]
		# for ploting the fitness
		worst_fit_v.append(min(Fitness))
		best_fit_v.append(max(Fitness))
		avg_fit_v.append(sum(Fitness) / len(Fitness))
		# check for new best solution
		for i in range(n_pop):
			if Fitness[i] > best_fit:
				best_para, best_fit = pop[i], Fitness[i]
				print(">%d, new best C and gamma:%s, best_fit %f" % (generation,  decoded[i], Fitness[i]))
		# select parents
		selected = [selection(pop, Fitness) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
	return best_para

# Ploting 
# def plot():
# 	fig, ax = plt.subplots(figsize=(10, 10))
# 	ax.plot(best_fit_v, label='Best Fitness')
# 	ax.plot(worst_fit_v, label='Worst Fitness')
# 	ax.plot(avg_fit_v, label='Average Fitness')
# 	plt.xlabel('Generations')
# 	plt.ylabel('Fitness')
# 	plt.title('Fitness Evolution')
# 	plt.legend()
# 	plt.savefig('Fitness Evalution.png')
# 	plt.close()
