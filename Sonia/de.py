# %%
import pandas as pd
import numpy as np
import random
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# %%
class DE:
  def __init__(self, cr, bounds, n_pop, cycles, fitness_function, population=None, old_pop=None):
    self.bounds = bounds
    self.n_pop = n_pop
    self.cycles = cycles
    self.fitness_function = fitness_function
    self.old_pop = old_pop.copy()
    
    if type(old_pop)==list:
      self.population = pd.DataFrame(self.old_pop, columns=bounds.keys())
      self.population['Fit'] = [self.fitness_function(x) for x in list(zip(*self.old_pop))]
    else:
      self.population = population.copy()
    
    self.cr = cr
    
    # keep track of best solution
    self.best = 0
    self.best_fit = 0
    self.best_para = 0    
    
    self.dim = len(self.bounds.keys())

  def __call__(self):
    
    self.keep_track(self.population['Fit'].idxmax())

    for _ in range(self.cycles):
      for i in range(self.n_pop):
        x = self.population.loc[i].to_dict()
        y = self.generate_mutant(i, self.population)
        z = self.Binomial_crossover(x, y)

        fit = self.fitness_function(list(z.values()))
        if fit > self.population['Fit'].loc[i]:
          for param in z.keys():
            self.population.loc[i, param] = z[param]
          self.population.loc[i, 'Fit'] = fit
      
      if self.population['Fit'].max()>self.best_fit:
        self.best += 1
        self.keep_track(self.population['Fit'].idxmax())

    #return self.best_para, self.best_fit
    return self.population[self.bounds.keys()].values.tolist(), self.population['Fit'].values.tolist()

  def generate_mutant(self, i, population):
    #parent vector
    x = self.population.loc[i]

    #Selecting from population three random distinct vectors
    indexes = list(self.population.index)
    indexes.remove(i)
    trg_i, rnd1_i, rnd2_i = random.sample(indexes, 3)

    trg = self.population.loc[trg_i]
    rnd1 = self.population.loc[rnd1_i]
    rnd2 = self.population.loc[rnd2_i]

    #generation of the mutant vector
    y = dict()
    for param in self.bounds.keys():
      x = trg[param] + random.uniform(0,1)*(rnd1[param] - rnd2[param])
      y[param] = self.clip(param, x)
    return y


  def Binomial_crossover(self, x, y):
    #initialization of the output vector as x
    z = x.copy()

    #selecting random param (between 0 and len of keys)
    param = random.choice(list(self.bounds.keys()))

    for p in self.bounds.keys():
      if random.uniform(0,1) < self.cr or p!=param:
        z[p] = y[p]
  
    return z

  def keep_track(self, best_sol_index):
    self.best_para = [{key: self.population[key].loc[best_sol_index]} for key in self.bounds.keys()]
    self.best_fit = self.population["Fit"].loc[best_sol_index]
  
  def clip(self, param, x):
    return max(self.bounds[param][0], min(self.bounds[param][-1], x))

# %%
if __name__ == "__main__":
        
    data = load_digits()
    n_samples = len(data.images)
    X = data.images.reshape((n_samples, -1))
    Y = data['target']

    def fitness_function(x):
        # clf = SVC(kernel='rbf', C=x[0], gamma=x[1], random_state=42)
        scores = cross_val_score(SVC(kernel='rbf', C=x[0], gamma=x[1]), X, Y, cv=5)

        return scores.mean()
    
    #bounds
    bounds = {"C": [0.001, 10.0], "gamma": [0.0001, 0.1]}

    #population size
    n_pop = 10

    #maximum iteration
    cycles = 10
    
    #crossover rate
    cr = 0.9

    de = DE(cr, bounds, n_pop, cycles, fitness_function, p)
    de()
    de.population


