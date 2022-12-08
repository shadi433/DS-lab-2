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
        self.n_pop = kwargs.get('n_pop', 3)

        self.n = len(self.bounds)
        self.pop = []
        if self.bounds is not None:
            self.pop = np.random.randn(self.n_pop,self.n)
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

        for i in range(self.Generations):
            
            best_pop_form_GAO, best_fit_from_GAO = GAO(fitness_function=fitness_function, bounds=self.bounds, n_bits=16, Generations=self.Generations, n_pop=self.n_pop, pop=self.pop, r_cross=0.9, r_mut = 1.0 / (float(n_bits=16) * len(self.bounds)))
            CSO_model = CSO(fitness_function=fitness_function, bounds=self.bounds, n_pop=self.n_pop, pop=self.pop, n=self.n, Generations=self.Generations)
            CSO_model.execute()
            best_pop_form_CSO = CSO_model.get_best() 
            best_fit_form_CSO = CSO_model.get_best_fit() 
            IWO_model = IWO(fitness_function=fitness_function, n_pop=self.n_pop, pop=self.pop, Generations=self.Generations, delta_cap=10**-6, num_exceeded_delta=6)
            best_pop_form_IWO = IWO_model.get_best()
            best_fit_form_IWO = IWO_model.get_best_fit()
            
            
        self.pop = [best_pop_form_GAO, best_pop_form_CSO, best_pop_form_IWO]
        self.fitness= [best_fit_from_GAO, best_fit_form_CSO, best_fit_form_IWO]
        self.clip_pop()

    def best_all(self):

        return best_algo, best_para, best_score
    