# Invasive Weed algorith
# Import necessary libraries
import numpy as np
import random
import time
import progressbar
import matplotlib.pyplot as plt
from Fit_fun import *




class IWO(): # Define class with model name
    def __init__(self, fitness_function, n_pop, pop): # Initiate class with required parameters
        seed_array = [] # Initiate empty list to contain tuples of x and y seed coordinates
        for i in range(n_pop):
            x = pop[i][0] 
            y = pop[i][1] 
            # Append the seed to the seed_array
            seed_array.append((x,y)) 
        self.fitness_function = fitness_function
        self.n_pop = n_pop
        
        # current_fitness_dict will contain the seed coordinates and their fitness scores
        current_fitness_dict = {}
        self.current_fitness_dict = current_fitness_dict
        # Initiate list that will contain best fitness score from each iteration
        best_fitness_list =  []
        self.best_fitness_list = best_fitness_list
        
        # Initiate list that will contain best scoring seeds from each iteration
        best_seed_dict = {}
        best_seed_list = []
        self.best_seed_list = best_seed_list
        self.best_seed_dict = best_seed_dict
        
        
        fitness_array = self.get_fitness(seed_array) # Get fitness array for current seeds
        fitness_dict = dict(zip(seed_array, fitness_array)) # Make a dictionary containing seed tuple as key and their fitness as values
        
        # Sort seed --> fitness dictionary
        # The highest scoring fitness value first (the best score)
        sorted_fitness_dict = {k: v for k, v in sorted(fitness_dict.items(), key=lambda item: item[1], reverse=True)} 
        best_seed_dict[list(sorted_fitness_dict.keys())[0]] = list(sorted_fitness_dict.values())[0]
        self.best_seed_dict = best_seed_dict
        
        # Append best seed to global list
        best_seed_list.append(list(sorted_fitness_dict.values())[0])
        
        # Normalize fitness value in the dictionary
        # Increase the distribution of fitness scores so they start with 0 instead of negative values

        # normalized_fitness_dict = self.normalize_fitness(sorted_fitness_dict)
        normalized_fitness_dict = sorted_fitness_dict
        
        # Compute number of children to be made from each seed
        # and output as a new directory with seed as key and number of children as values
        seed_children_dict = self.eval_fitness(normalized_fitness_dict)
        
        # Append best fitness score to the global fitness_array
        best_fitness_list.append(sorted(fitness_array, reverse=True)[0])
        #print(best_fitness_list,"\n")
        
        # Add seeds and their fitness scores from this iteration to the population (current_fitness_dict)
        for seed, fitness in fitness_dict.items():
            current_fitness_dict[seed] = fitness

        
        # If population is bigger than the set n_pop
        # we need to trim it by keeping only best scoring seeds
        if len(current_fitness_dict) > n_pop:
                self.prune_seeds()

        
        # Generate new seeds and add them to current array of seeds
        seed_array = seed_array + self.generate_population(seed_children_dict)
    
    
    # Function to generate the new seeds in each generation
    # Based on their previous fitness
    def generate_population(self,seed_children_dict):
        new_seeds = [] # Initiate list of new seeds
        
        # Based on how many new children each previous fitness parent produces
        # Inititate new seeds within 0.1 standard distributions from previous parent coordinates
        for seed, nchildren in seed_children_dict.items():
            for _ in range(nchildren): # this loop will run as long as nchildren for each parent
                x = seed[0] # X coordinate
                y = seed[1] # Y coordinate
                new_x = random.gauss(x,0.1) # Initiate new x
                new_y = random.gauss(y,0.1) # Initiate new y

                # If x was outside set bounds (0,10), Initiate new x until a correct one is produced
                while new_x < 1 or new_x > 10:
                    new_x=random.gauss(x,0.1)

                # If y was outside set bounds (0,10), Initiate new y until a correct one is produced
                while new_y < 0.0001 or new_y > 0.1:
                    new_y=random.gauss(y,0.1)

                # Append new seeds to the population
                new_seeds.append((new_x, new_y))

        # Return new seeds from the function
        return new_seeds
    
    # Get fitness of new seeds
    def get_fitness(self,seed_array):
        fitness_array = [] # Initiate fitness array for new seeds

        for seed in seed_array:
            fitness = self.fitness_function(seed)
            # Append fitness to the fitness array
            fitness_array.append(fitness)
            
        return fitness_array
    
    
    # Function to normalize fitness into values from 0 to max
    # Slide the values to positives based on the lowest negative values
    def normalize_fitness(self,sorted_fitness_dict):
        # Get the seed with the best/lowhest fitness
        lowest_fitness_element = list(sorted_fitness_dict.items())[0]
        
        new_vals = [] # Initiate list for new fitness values
        for val in sorted_fitness_dict.values():
             # Subtract value from the lowest value in sorted fitness dictionary
            new_val = val - list(sorted_fitness_dict.values())[0]
            new_vals.append(new_val)
        
        # Save sorted dictionary with new values of fitness
        sorted_fitness_dict = dict(zip(sorted_fitness_dict.keys(), new_vals))
        
        # Keep the lowest fitness element to be accessed
        self.lowest_fitness_element = lowest_fitness_element
        return sorted_fitness_dict

    # Generate variable number of seeds based on the fitness score of the parent seed
    def eval_fitness(self,normalized_fitness_dict):
        children_seed_list = [] # Initiate list for new seeds number
        maximum = max(normalized_fitness_dict.values()) # get maximum fitness value
        
        # Perform fitness to N children transformation based on each fitness value
        # and the maximum fitness value from this generation
        for val in normalized_fitness_dict.values(): 
            # (val+0.00001) makes sure that the 0 error does not occur
            # Other calculations were attempted but this one proved the most efficient for the task
            num_seed = round( ( (val+0.000001) ) / maximum)
            
            # Append the num of new children to the pre-initialized list
            children_seed_list.append(num_seed)

        # Create dictionary containing seeds and reversed number of children
        # That is because the lowest fitness score is one that should actually have most children
        seed_children_dict = dict(zip(normalized_fitness_dict.keys(), children_seed_list))

        return seed_children_dict

    # Function to prune the seeds to the number of population maximum (n_pop)
    def prune_seeds(self):
        # Sort the current dictionary with seeds and their fitness based on fitness
        sorted_current_fitness_dict = {k: v for k, v in sorted(self.current_fitness_dict.items(), key=lambda item: item[1], reverse=True)}
        
        # Prune the sorted dictionary to population max (n_pop) and return as class fitness dictionary
        self.current_fitness_dict = dict(list(sorted_current_fitness_dict.items())[0:self.n_pop]) 
    
    # Return best iteration from class
    def return_best_iteration_(self):
        return self.i
    
    # Return model runtime from class
    def return_runtime_(self):
        return self.runtime
    
    # Return best seed from class
    def return_best_seed_(self):
        # Make sure the current best fitness is the maximum of fitness values from the iterations
        best_seed_dict = {k: v for k, v in sorted(self.best_seed_dict.items(), key=lambda item: item[1], reverse=True)}
        
        # Return best seed
        return list(max(best_seed_dict.items(), key=lambda x: x[1]))
    
    # Return best fitness value from class
    def return_best_fitness_(self):
        return list(self.best_seed_list)
    
        # def get_best(self): #<------------------------coming back here
    #     return best_seed

    def get_best_fit(): #<------------------------coming back here
        return None

    

model = IWO(fitness_function, n_pop, pop) # Iniiate model with current parameters
best_fitness_score = model.return_best_fitness_() # Get best fitness score from the model
best_iteration = model.return_best_iteration_() # Get best iteration from the model ( Should be last/highest one )\
                    
sorted_results_dict = {k: v for k, v in sorted(res_dic.items(), key=lambda item: item[1])}  # Sort resulst dictionary
sorted_seeds = sorted(best_seeds, key=lambda x: x[1]) # sort best seeds dictionary by fitness

best_params = list(sorted_results_dict.keys())[0] # Get params for best seed from gridsearch
best_fitness = list(sorted_results_dict.values())[0] # Get best fitness from gridsearch

best_seed = sorted_seeds[0][0] # Get best seed from gridsearch

best_round_fitnesses = list(grid_search_dict[best_params]) # Get all fitnesses from the best model from gridsearch

# Print the best model as well as the current best parameters
# and best iteration, total runtime, best fitness and best seed from gridsearch
print(f"\nGridsearch results:")
print(f"\tBest model:")
print(f"\t\tIWO(n_pop = {best_params[0]}, Generations = {best_params[1]}, delta_cap = {best_params[2]}, num_exceeded_delta = {best_params[3]})\n")
print(f"\tStopped at iteration: {best_params[4]}\n")
print(f"\tBest fitness: {best_fitness}")
print(f"\tBest seed coordinates: {best_seed}")
    