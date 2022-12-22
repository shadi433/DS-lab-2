from GAO import *
from ACO_from_scrach import *
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
    # Iterate over the list and compare adjacent elements
    for i in range(len(lst1) - 1):
        # If the current element is greater than the next element, swap them
        if lst1[i] > lst1[i + 1]:
            lst1[i], lst1[i + 1] = lst1[i + 1], lst1[i]
            lst2[i], lst2[i + 1] = lst2[i + 1], lst2[i]

    # Return the sorted list
    return lst1, lst2



class JSI:
    
    def __init__(self, **kwargs):
        self.model = kwargs.get('model', 'svm')
        self.parameters = kwargs.get('parameters', 2)
        self.bounds = kwargs.get('bounds', [[1.0, 10.0], [0.0001, 0.1]])
        self.Generations = kwargs.get('Generations', 10)
        self.n_pop = kwargs.get('n_pop', 10)

        self.n = len(self.bounds)
        self.pop = []
        if self.bounds is not None:
            self.pop = np.random.randn(self.n_pop,self.n) # list of lists, for 2 dim: [[ , ], [ , ], [ , ],...,[ , ]]
            self.clip_pop()
        else:
            print('Please determine the bounds for the paremeters')
        
    def clip_pop(self):
        # IF BOUND IS SPECIFIED THEN CLIP 'pop' VALUES SO THAT THEY ARE IN THE SPECIFIED RANGE
        if self.bounds is not None:
            for i in range(self.n):
                xmin, xmax = self.bounds[i]
                self.pop[:,i] = np.clip(self.pop[:,i], xmin, xmax)
  
    def run(self):
        old_pop_GAO = self.pop
        old_pop_CSO = self.pop
        pop_IWO = self.pop

        for i in range(self.Generations):
            
            new_pop_GAO, fit_GAO = GAO(fitness_function=fitness_function, bounds=self.bounds, n_bits=16, n_pop=self.n_pop, pop=old_pop_GAO, r_cross=0.9, r_mut = 1.0 / (float(n_bits=16) * len(self.bounds)))
            CSO_model = CSO(fitness_function=fitness_function, bounds=self.bounds, n_pop=self.n_pop, pop=old_pop_CSO, n=self.n)
            CSO_model.execute()
            new_pop_CSO = CSO_model.get_pop() 
            fit_CSO = CSO_model.get_fit() 
            # IWO_model = IWO(fitness_function=fitness_function, n_pop=self.n_pop, pop=self.pop, Generations=self.Generations, delta_cap=10**-6, num_exceeded_delta=6)
            # pop_IWO = IWO_model.get_best()
            # fit_IWO = IWO_model.get_best_fit()
            
            sorted_fit_GAO, sorted_pop_GAO = bubble_sort(fit_GAO, new_pop_GAO)
            sorted_fit_CSO, sorted_pop_CSO = bubble_sort(fit_CSO, new_pop_CSO)
            sorted_pop_GAO[self.n_pop-1] = sorted_pop_CSO[0].copy()
            sorted_pop_CSO[self.n_pop-1] = sorted_pop_GAO[0].copy()
            old_pop_GAO = sorted_pop_GAO
            old_pop_CSO = sorted_pop_CSO
        

        

    def best_all(self):

        

        return best_algo, best_para, best_score
    