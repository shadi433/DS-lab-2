# genetic algorithm search for continuous function optimization

from Fit_fun import *
from GA import *
from CS import *


                                   



# define range for input
bounds = [[1.0, 10.0], [0.0001, 0.1]]
# define the total iterations
Generations = 10
# bits per variable
n_bits = 16
# define the population size
n_pop = 10
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / (float(n_bits) * len(bounds))

#--------------------------------------------------------------------------------------------------------
# perform the genetic algorithm search

# best_para, best_fit = genetic_algorithm(fitness_function, bounds, n_bits, Generations, n_pop, r_cross, r_mut)
# print('Done!')
# decoded = decode(bounds, n_bits, best_para)
# print('best_parameters:%s' % decoded)

#--------------------------------------------------------------------------------------------------------

CSO(fitness=fitness_function, bound=[(1.0, 10.0),(0.0001, 0.1)], n_pop=n_pop, Generations=Generations, verbose=True).execute()
