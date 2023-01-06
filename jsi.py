from GAO import *
from CSO import *
from IWO import *

import sys
sys.path.insert(1, '../Sonia')
from abc_opt import ABC_OPT
from de import DE
from gso import GSO
from fss_opt import FSS

import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score


data = load_digits() 

n_samples = len(data.images)
X = data.images.reshape((n_samples, -1))
Y = data['target']

# modle implementation
def fitness_function1(x):
	# x[0] = C and x[1] = gamma
	clf = svm.SVC(kernel='rbf', C=x[0], gamma=x[1], random_state=42)
	scores = cross_val_score(clf, X, Y, cv=5)
	
	return scores.mean()

def fitness_function2(x):
	pass

def bubble_sort(lst1, lst2):
    # Set a flag to True to indicate that the list is not yet sorted
    sorted = False
    # Keep looping until the list is sorted
    while not sorted:
        # Set the flag to True to assume that the list is already sorted
        sorted = True
        # Iterate over the list and compare adjacent elements
        for i in range(len(lst1) - 1):
            # If the current element is smaller than the next element, swap them
            if lst1[i] < lst1[i + 1]:
                lst1[i], lst1[i + 1] = lst1[i + 1], lst1[i]
                lst2[i], lst2[i + 1] = lst2[i + 1], lst2[i]
                sorted = False
    # Return the sorted list
    return lst1, lst2
def clip_pop(pop, intervals):
    # IF BOUND IS SPECIFIED THEN CLIP 'pop' VALUES SO THAT THEY ARE IN THE SPECIFIED RANGE
    return [[random.uniform(lower_bound, upper_bound) if not lower_bound <= x <= upper_bound else x for x, (lower_bound, upper_bound) in zip(sublist, intervals)] for sublist in pop]



class JSI:
    
    def __init__(self, **kwargs):
        self.model = kwargs.get('model', 'svm')
        self.parameters = kwargs.get('parameters', 2)
        self.intervals = kwargs.get('intervals', [[1.0, 100.0], [0.0001, 0.1]])
        self.interval_dict = {'c': self.intervals[0], 'gamma': self.intervals[1]}
        self.Generations = kwargs.get('Generations', 10)
        self.n_pop = kwargs.get('n_pop', 10)

        if self.model == 'svm':
            self.fitness_function = fitness_function1
        elif self.model == 'decision tree':
            self.fitness_function = fitness_function2

        self.best_pop_from_all=[]
        self.best_fit_from_all=[]
        self.model_name=[]
        self.n = len(self.intervals)

        #initialize the population
        self.pop = [] 
        if self.intervals is not None:
            for i in range(self.n_pop):
                x = [random.uniform(interval[0], interval[1]) for interval in self.intervals]
                self.pop.append(x) # list of lists, for 2 dim: [[ , ], [ , ], [ , ],...,[ , ]]
            self.pop = clip_pop(self.pop, self.intervals)
        else:
            print('Please determine the intervals for the paremeters')
  
    def run(self):
        old_pop_GAO = self.pop.copy()
        old_pop_CSO = self.pop.copy()
        old_pop_IWO = self.pop.copy()

        old_pop_ABC = self.pop.copy()
        old_pop_DE = self.pop.copy()
        old_pop_GSO = self.pop.copy()
        old_pop_FSS = self.pop.copy()
        
        #params for fss
        #initial step
        Sinit = {p: self.interval_dict[p][0] for p in self.interval_dict.keys()}

        #final step
        Sfinal = {p: self.interval_dict[p][1] for p in self.interval_dict.keys()}

        for i in range(self.Generations):
            print('iter:',i)
            
            #new_pop_Algo, fit_Algo = Algo(...., pop=old_pop_Algo,...)
            #[ , , , ...], [ , , , ...] = Algo(..., pop = [ , , , ...], ...)

            new_pop_GAO, fit_GAO = GAO(fitness_function=self.fitness_function, pop=old_pop_GAO, intervals=self.intervals)
            new_pop_CSO, fit_CSO = CSO(fitness_function=self.fitness_function, nest=old_pop_CSO, n_pop=self.n_pop, intervals=self.intervals, pa=0.25, beta=1.5)
            new_pop_IWO, fit_IWO = IWO(dim=self.parameters, fitness_function=self.fitness_function, n_pop=self.n_pop, pop=old_pop_IWO, intervals=self.intervals, rinitial=2, rfinal=0.1, modulation_index=2, itermax=self.Generations, iter=i)
            
            
            new_pop_ABC, fit_ABC = ABC_OPT(bounds=self.interval_dict, n_pop=self.n_pop, cycles=1, fitness_function=self.fitness_function, population=None, old_pop = old_pop_ABC)()
            new_pop_DE, fit_DE = DE(cr=0.9, bounds=self.interval_dict, n_pop=self.n_pop, cycles=1, fitness_function=self.fitness_function, population=None, old_pop = old_pop_DE)()
            new_pop_GSO, fit_GSO = GSO(rho=0.3, gamma=0.65, s=0.5, rs=0.45, r0=4, betta=0.075, l0=0.25, bounds=self.interval_dict, n_pop=self.n_pop, cycles=1, fitness_function=self.fitness_function, population=None, old_pop = old_pop_GSO)()
            new_pop_FSS, fit_FSS = FSS(Sinit, Sfinal, bounds=self.interval_dict, n_pop=self.n_pop, cycles=1, fitness_function=self.fitness_function, population=None, old_pop = old_pop_FSS)()
            
            
            #sorting the fitness with the corresponding population
            sorted_fit_GAO, new_pop_GAO = bubble_sort(fit_GAO, new_pop_GAO)
            sorted_fit_CSO, new_pop_CSO = bubble_sort(fit_CSO, new_pop_CSO)
            sorted_fit_IWO, new_pop_IWO = bubble_sort(fit_IWO, new_pop_IWO)

            sorted_fit_ABC, new_pop_ABC = bubble_sort(fit_ABC, new_pop_ABC)
            sorted_fit_DE, new_pop_DE = bubble_sort(fit_DE, new_pop_DE)
            sorted_fit_GSO, new_pop_GSO = bubble_sort(fit_GSO, new_pop_GSO)
            sorted_fit_FSS, new_pop_FSS = bubble_sort(fit_FSS, new_pop_FSS)


            #change the n-1 worst nembers in each of the n population with the 
            #best member from each of the n-1 other population
            new_pop_GAO[self.n_pop-1] = new_pop_CSO[0].copy()
            new_pop_GAO[self.n_pop-2] = new_pop_IWO[0].copy()

            new_pop_CSO[self.n_pop-1] = new_pop_GAO[0].copy()
            new_pop_CSO[self.n_pop-2] = new_pop_IWO[0].copy()

            new_pop_IWO[self.n_pop-1] = new_pop_GAO[0].copy()
            new_pop_IWO[self.n_pop-2] = new_pop_CSO[0].copy()
            
            new_pop_ABC[self.n_pop-1] = new_pop_DE[0].copy()
            new_pop_ABC[self.n_pop-2] = new_pop_GSO[0].copy()
            new_pop_ABC[self.n_pop-3] = new_pop_FSS[0].copy()
            
            new_pop_DE[self.n_pop-1] = new_pop_ABC[0].copy()
            new_pop_DE[self.n_pop-2] = new_pop_GSO[0].copy()
            new_pop_DE[self.n_pop-3] = new_pop_FSS[0].copy()
            
            new_pop_GSO[self.n_pop-1] = new_pop_DE[0].copy()
            new_pop_GSO[self.n_pop-2] = new_pop_ABC[0].copy()
            new_pop_GSO[self.n_pop-3] = new_pop_FSS[0].copy()
            
            new_pop_FSS[self.n_pop-1] = new_pop_DE[0].copy()
            new_pop_FSS[self.n_pop-2] = new_pop_GSO[0].copy()
            new_pop_FSS[self.n_pop-3] = new_pop_ABC[0].copy()

            old_pop_GAO = new_pop_GAO
            old_pop_CSO = new_pop_CSO
            old_pop_IWO = new_pop_IWO
            
            old_pop_ABC = new_pop_ABC
            old_pop_DE = new_pop_DE
            old_pop_GSO = new_pop_GSO
            old_pop_FSS = new_pop_FSS

            #end of for loop
        #after the loop finishes store the best fitness and the corresponding population of each algorithm
        self.best_pop_from_all.append(new_pop_GAO[0])
        self.best_fit_from_all.append(sorted_fit_GAO[0])
        self.model_name.append('GAO')

        self.best_pop_from_all.append(new_pop_CSO[0])
        self.best_fit_from_all.append(sorted_fit_CSO[0])
        self.model_name.append('CSO')

        self.best_pop_from_all.append(new_pop_IWO[0])
        self.best_fit_from_all.append(sorted_fit_IWO[0])
        self.model_name.append('IWO')

        
        self.best_pop_from_all.append(new_pop_ABC[0])
        self.best_fit_from_all.append(sorted_fit_ABC[0])
        self.model_name.append('ABC')

        self.best_pop_from_all.append(new_pop_DE[0])
        self.best_fit_from_all.append(sorted_fit_DE[0])
        self.model_name.append('DE')

        self.best_pop_from_all.append(new_pop_GSO[0])
        self.best_fit_from_all.append(sorted_fit_GSO[0])
        self.model_name.append('GSO')

        self.best_pop_from_all.append(new_pop_FSS[0])
        self.best_fit_from_all.append(sorted_fit_FSS[0])
        self.model_name.append('FSS')
        
    # after haveing the best fitness and the corresponding population of each algorithm
    # this function will return the best among all
    def best_all(self): 
        # 3 lists and sortin based on the second one (fitness): [  ,  ,  ] , [  ,  ,  ], [  ,  ,  ]
        best = sorted(zip(self.best_pop_from_all, self.best_fit_from_all, self.model_name), key=lambda x: x[1], reverse=True)
        # best will be like: [( , , ),( , , ),( , , )]
        return best[0][0], best[0][1], best[0][2]