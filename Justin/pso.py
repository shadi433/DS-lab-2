# -*- coding: utf-8 -*-
"""Untitled7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OChRlsTjD5Xl66R-MaS3EmSyfjxt_E0B
"""

import random

# Calculate the fitness of a particle
def fitness(particle, fitness_function):
	return fitness_function(particle)

# Update the velocity of a particle based on its best position and the global best position
def update_velocity(particle, best_position, global_best_position, velocity, c1, c2):
	r1 = random.uniform(0, 1)
	r2 = random.uniform(0, 1)
	velocity = [v + c1 * r1 * (bp - p) + c2 * r2 * (gbp - p) for v, bp, p, gbp in zip(velocity, best_position, particle, global_best_position)]
	return velocity

# Update the position of a particle based on its velocity
def update_position(particle, velocity, intervals):
	for i in range(len(particle)):
		particle[i] = particle[i] + velocity[i]
		if particle[i] < intervals[i][0]:
			particle[i] = intervals[i][0]
		elif particle[i] > intervals[i][1]:
			particle[i] = intervals[i][1]
	return particle

# Run the particle swarm optimization algorithm to find the optimal solution
def PSO(fitness_function, pop_size, intervals, max_iter, c1, c2):
	# Initialize the population with random particles within the specified intervals
	population = [[random.uniform(intervals[i][0], intervals[i][1]) for i in range(len(intervals))] for _ in range(pop_size)]
	
	# Initialize the velocities with random values
	velocities = [[random.uniform(-1, 1) for i in range(len(intervals))] for _ in range(pop_size)]
	
	# Initialize the best positions with the current positions
	best_positions = population[:]
	
	# Initialize the global best position with the best position of the first particle
	global_best_position = best_positions[0]
	
	# Iterate over the maximum number of iterations
	for _ in range(max_iter):
		# Calculate the fitness of each particle
		fitnesses = [fitness(particle, fitness_function) for particle in population]
		
		# Update the best positions and global best position
		for i in range(pop_size):
			if fitnesses[i] > fitness(best_positions[i], fitness_function):
				best_positions[i] = population[i][:]
			if fitnesses[i] > fitness(global_best_position, fitness_function):
				global_best_position = population[i][:]
		
		# Update the velocity and position of each particle
		for i in range(pop_size):
			velocities[i] = update_velocity(population[i], best_positions[i], global_best_position, velocities[i], c1, c2)
			population[i] = update_position(population[i], velocities[i], intervals)
	
	# Return the global best position and the population and fitness of each particle
	return global_best_position, population, fitnesses