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

class JSI:
    
    def __init__(self, **kwargs):
        self.model = kwargs.get('model', 'svm')
        self.parameters = kwargs.get('parameters', 2)
        self.bounds = kwargs.get('bounds', [[1.0, 10.0], [0.0001, 0.1]])
        self.fitness_fun = kwargs.get('fitness_fun', fitness_function)
        self.Generations = kwargs.get('Generations', 10)
        self.n_pop = kwargs.get('n_pop', 12)

        self.n = len(self.bounds)
        self.X = []
        if self.bounds is not None:
            self.X = np.random.randn(self.n_pop,self.n)
            self.clip_X()
        else:
            print('Please determine the bounds for the paremeters')
        
    def clip_X(self):
        # IF BOUND IS SPECIFIED THEN CLIP 'X' VALUES SO THAT THEY ARE IN THE SPECIFIED RANGE
        if self.bounds is not None:
            for i in range(self.n):
                xmin, xmax = self.bounds[i]
                self.X[:,i] = np.clip(self.X[:,i], xmin, xmax)
    
    def run(self):



        for i in range(self.Generations):
            
            best_pop_form_GAO = GAO(fitness_function=fitness_function, bounds, n_bits=16, Generations=self.Generations, n_pop=self.n_pop, r_cross=0.9, r_mut = 1.0 / (float(n_bits=16) * len(self.bounds)))
            CSO_model = CSO(fitness=fitness_function, bounds=[(1.0, 10.0),(0.001, 0.1)], n_pop=self.n_pop, Generations=self.Generations)
            best_pop_form_CSO = CSO_model.execute()
            IWO_model = IWO(pmax=self.n_pop, maxiter=self.Generations, delta_cap=10**-6, num_exceeded_delta=6)
            best_pop_form_IWO = IWO_model.return_best_seed_()
            