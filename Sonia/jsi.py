# %%
import pandas as pd
import numpy as np
import random
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# %%
from abc_opt import ABC
from de import DE
from gso import GSO
from fss_opt import FSS
from params import *

# %%
#bounds
bounds = {"C": [0.001, 10.0], "gamma": [0.0001, 0.1]}

#population size
n_pop = 10

#maximum iteration
cycles = 10

# %%
data = load_digits()
n_samples = len(data.images)
X = data.images.reshape((n_samples, -1))
Y = data['target']

def fitness_function(x):
  # clf = SVC(kernel='rbf', C=x[0], gamma=x[1], random_state=42)
  scores = cross_val_score(SVC(kernel='rbf', C=x[0], gamma=x[1]), X, Y, cv=5)

  return scores.mean()

# %%
import params

# %%
class swarm:
    def __init__(self, bounds, n_pop, fitness_function, params):
        self.bounds = bounds
        self.n_pop = n_pop
        self.fitness_function = fitness_function
        self.population = self.get_population()
        self.params = params
    
    def get_population(self):
        self.pop_dict = self.init_pop()
        self.pop = pd.DataFrame.from_dict(self.pop_dict)
        #calculating the Fit of the initialised population (employed bees)
        self.pop['Fit'] = [self.fitness_function(x) for x in list(zip(*self.pop_dict.values()))]
        self.population = self.pop.copy()
        return self.population
    
    def init_pop(self):
        pop_dict = dict()
        for i in range(self.n_pop):
            for key in self.bounds.keys():
                if i == 0:
                    pop_dict[key] = [self.bounds[key][0] + random.uniform(0,1)*(self.bounds[key][-1] - self.bounds[key][0])]
                else:
                    pop_dict[key].append(self.bounds[key][0] + random.uniform(0,1)*(self.bounds[key][-1] - self.bounds[key][0]))
        return pop_dict
    
    
    def generation_rnd(self, alg):
        l = [self.abc_rnd, self.de_rnd, self.g_rnd, self.fss_rnd]
        l.remove(alg)
        
        new_pop = alg.population.copy()
        indexes = random.choices(list(new_pop.index), k=random.randint(1, len(new_pop)-1))
        
        for i in indexes:
            rep = l[random.randint(0, len(l)-1)].population
            new_pop.loc[i, self.bounds.keys()] = rep.loc[random.randint(0, len(rep)-1), self.bounds.keys()]
            new_pop.loc[i, 'Fit'] = self.fitness_function(new_pop[self.bounds.keys()].values[i])
        
        return new_pop
    
    def generation_best(self, alg):
        l = [self.abc_best, self.de_best, self.g_best, self.fss_best]
        l.remove(alg)
        
        new_pop = alg.population.copy()
        
        for i in l:
            new_pop.loc[new_pop['Fit'].idxmin(), self.bounds.keys()] = i.population.loc[i.population['Fit'].idxmax()][self.bounds.keys()]
            new_pop.loc[new_pop['Fit'].idxmin(), 'Fit'] = self.fitness_function(new_pop[self.bounds.keys()].values[new_pop['Fit'].idxmin()])
        return new_pop
    
    
    def run(self):
        p = self.population.copy()
        #first iteration
        abc = ABC(self.bounds, self.n_pop, 10, self.fitness_function, population=p)
        abc()

        de = DE(self.params.cr, self.bounds, self.n_pop, 10, self.fitness_function, population=p)
        de()

        g = GSO(self.params.rho, self.params.gamma, self.params.s, self.params.rs, self.params.r0, self.params.betta, self.params.l0, self.bounds, self.n_pop, 10, self.fitness_function, population=p)
        g()

        fss = FSS(self.params.Sinit, self.params.Sfinal, self.bounds, self.n_pop, 10, self.fitness_function, population=p)
        fss()
        
        #the first proposed approach
        self.abc_rnd = copy(abc)
        self.de_rnd = copy(de)
        self.g_rnd = copy(g)
        self.fss_rnd = copy(fss)
        
        # f.append(("f",)+ l)
        self.best_para_rnd = []
        for i in range(2):
            self.abc_rnd = ABC(self.bounds, self.n_pop, 1, self.fitness_function, population=self.generation_rnd(self.abc_rnd))
            self.best_para_rnd.append(("abc",)+ self.abc_rnd())
            self.de_rnd = DE(self.params.cr, self.bounds, self.n_pop, 1, self.fitness_function, population=self.generation_rnd(self.de_rnd))
            self.best_para_rnd.append(("de",)+ self.de_rnd())
            self.g_rnd = GSO(self.params.rho, self.params.gamma, self.params.s, self.params.rs, self.params.r0, self.params.betta, self.params.l0, self.bounds, self.n_pop, 1, self.fitness_function, population=self.generation_rnd(self.g_rnd))
            self.best_para_rnd.append(("gso",)+ self.g_rnd())
            self.fss_rnd = FSS(self.params.Sinit, self.params.Sfinal, self.bounds, self.n_pop, 1, self.fitness_function, population=self.generation_rnd(self.fss_rnd))
            self.best_para_rnd.append(("fss",)+ self.fss_rnd())
            
        #the second proposed approach
        self.abc_best = copy(abc)
        self.de_best = copy(de)
        self.g_best = copy(g)
        self.fss_best = copy(fss)
        
        self.best_para = []
        for i in range(2):
            self.abc_best = ABC(self.bounds, self.n_pop, 1, self.fitness_function, population=self.generation_best(self.abc_best))
            self.best_para.append(("abc",)+ self.abc_best())
            self.de_best = DE(self.params.cr, self.bounds, self.n_pop, 1, self.fitness_function, population=self.generation_best(self.de_best))
            self.best_para.append(("de",)+ self.de_best())
            self.g_best = GSO(self.params.rho, self.params.gamma, self.params.s, self.params.rs, self.params.r0, self.params.betta, self.params.l0, self.bounds, self.n_pop, 1, self.fitness_function, population=self.generation_best(self.g_best))
            self.best_para.append(("gso",)+ self.g_best())
            self.fss_best = FSS(Sinit, Sfinal, self.bounds, self.n_pop, 1, self.fitness_function, population=self.generation_best(self.fss_best))
            self.best_para.append(("fss",)+ self.fss_best())
            
    def best_all(self):
        print("best parameters and fitness according to the approach: ")
        print("replace random elements by other algorithms random elements: ")
        mr, bpr, bfr = sorted(
            self.best_para_rnd, 
            key=lambda x: x[2]
        )[-1]  
        print('model: ', mr, 'best parameters: ', bpr, 'best fitness: ', bfr)      
        
        print("replace worst elements by other algorithms best elements: ")
        m, bp, bf = sorted(
            self.best_para, 
            key=lambda x: x[2]
        )[-1]  
        print('model: ', m, 'best parameters: ', bp, 'best fitness: ', bf)  

# %%
s = swarm(bounds, n_pop, fitness_function, params)

# %%
s.run()

# %%
s.best_all()

# %%



