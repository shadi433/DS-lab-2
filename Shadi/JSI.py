from GAO import *
from CSO import *
from IWO import *

import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score


data = load_digits() 

n_samples = len(data.images)
X = data.images.reshape((n_samples, -1))
Y = data['target']

# modle implementation
def fitness_function(x):  
	# x[0] = C and x[1] = gamma
	clf = svm.SVC(kernel='rbf', C=x[0], gamma=x[1], random_state=42)
	scores = cross_val_score(clf, X, Y, cv=5)
	
	return scores.mean()

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



class JSI:
    
    def __init__(self, **kwargs):
        self.model = kwargs.get('model', 'svm')
        self.parameters = kwargs.get('parameters', 2)
        self.intervals = kwargs.get('intervals', [[1.0, 10.0], [0.0001, 0.1]])
        self.Generations = kwargs.get('Generations', 10)
        self.n_pop = kwargs.get('n_pop', 10)

        self.best_pop_from_all=[]
        self.best_fit_from_all=[]
        self.n = len(self.intervals)

        self.pop = []
        if self.intervals is not None:
            for i in range(self.n_pop):
                x = [random.uniform(interval[0], interval[1]) for interval in self.intervals]
                self.pop.append(x) # list of lists, for 2 dim: [[ , ], [ , ], [ , ],...,[ , ]]
            self.clip_pop()
        else:
            print('Please determine the intervals for the paremeters')
        
    def clip_pop(self):
        # IF BOUND IS SPECIFIED THEN CLIP 'pop' VALUES SO THAT THEY ARE IN THE SPECIFIED RANGE
        if self.intervals is not None:
            for i in range(self.n):
                xmin, xmax = self.intervals[i]
                self.pop[:][i] = np.clip(self.pop[:][i], xmin, xmax)
  
    def run(self):
        old_pop_GAO = self.pop
        old_pop_CSO = self.pop
        old_pop_IWO = self.pop


        for i in range(self.Generations):
            
            new_pop_GAO, fit_GAO = GAO(fitness_function=fitness_function, pop=old_pop_GAO, intervals=self.intervals)

            new_pop_CSO, fit_CSO = CSO(fitness_function=fitness_function, nest=old_pop_CSO, n_pop=self.n_pop, intervals=self.intervals, pa=0.25, beta=1.5)

            new_pop_IWO, fit_IWO = IWO(dim=self.parameters, fitness_function=fitness_function, n_pop=self.n_pop, pop=old_pop_IWO, rinitial=2, rfinal=0.1, modulation_index=2, itermax=self.Generations, iter=i)
            
            sorted_fit_GAO, sorted_pop_GAO = bubble_sort(fit_GAO, new_pop_GAO)
            sorted_fit_CSO, sorted_pop_CSO = bubble_sort(fit_CSO, new_pop_CSO)
            sorted_fit_IWO, sorted_pop_IWO = bubble_sort(fit_IWO, new_pop_IWO)

            sorted_pop_GAO[self.n_pop-1] = sorted_pop_CSO[0].copy()
            sorted_pop_GAO[self.n_pop-2] = sorted_pop_IWO[0].copy()

            sorted_pop_CSO[self.n_pop-1] = sorted_pop_GAO[0].copy()
            sorted_pop_CSO[self.n_pop-2] = sorted_pop_IWO[0].copy()

            sorted_pop_IWO[self.n_pop-1] = sorted_pop_GAO[0].copy()
            sorted_pop_IWO[self.n_pop-2] = sorted_pop_CSO[0].copy()

            old_pop_GAO = sorted_pop_GAO
            old_pop_CSO = sorted_pop_CSO
            old_pop_IWO = sorted_pop_IWO
        
        self.best_pop_from_all.append(sorted_pop_GAO[0])
        self.best_fit_from_all.append(sorted_fit_GAO[0])

        self.best_pop_from_all.append(sorted_pop_CSO[0])
        self.best_fit_from_all.append(sorted_fit_CSO[0])

        self.best_pop_from_all.append(sorted_pop_IWO[0])
        self.best_fit_from_all.append(sorted_fit_IWO[0])

        

    def best_all(self):

        sorted_fit, sorted_pop = bubble_sort(self.best_fit_from_all, self.best_pop_from_all)

        return sorted_pop[0], sorted_fit[0]
    