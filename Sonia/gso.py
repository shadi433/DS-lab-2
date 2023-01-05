# %%
import pandas as pd
import numpy as np
import random
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# %%
class GSO:
  def __init__(self, rho, gamma, s, rs, r0, betta, l0, bounds, n_pop, cycles, fitness_function, population=None, old_pop=None):

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
      
    self.rho = rho
    self.gamma = gamma
    self.s = s
    self.rs = rs
    self.r0 = r0
    self.betta = betta
    self.l0 = l0

    self.population['luciferin'] = l0
    self.population['neigh_rng'] = r0

    self.population['neigh'] = [[] for _ in range(self.population.shape[0])]
    self.population['prob_mov'] = None
    
    # keep track of best solution
    self.best = 0
    self.best_fit = 0
    self.best_para = 0    
    
    self.dim = len(self.bounds.keys())
  
  def __call__(self):
    
    self.keep_track(self.population["Fit"].idxmax())
    
    for _ in range(self.cycles):      
      self.update_luciferin()
      
      for idx in range(self.n_pop):
        self.define_neibhors(idx)
        self.max_prob_moving(idx)
        self.update_glowworm_mov(idx)
        self.update_nei_rng(idx)
    
    # return self.best_para, self.best_fit
    return self.population[self.bounds.keys()].values.tolist(), self.population['Fit'].values.tolist()
  
  def update_luciferin(self):
    for i in range(self.n_pop):
      self.population.loc[i, 'luciferin'] = float((1-self.rho)*(self.population.loc[i, ['luciferin']])
        + self.gamma*self.fitness_function(self.population.loc[i, self.bounds.keys()]))
  def define_neibhors(self, i):
    indexes = list(self.population.index)
    indexes.remove(i)
    for j in indexes:
      if self.population['luciferin'][i]<self.population['luciferin'][j]:
        #euclidean distance:
        a, b = self.population.loc[i, self.bounds.keys()], self.population.loc[j, self.bounds.keys()]
        dist = np.linalg.norm(a-b)

        if dist<self.population['neigh_rng'][i]:
          self.population['neigh'].loc[i].append(j)
  def max_prob_moving(self, i):
    sum_n = 0
    for k in self.population.loc[i, 'neigh']:
      sum_n += self.population['luciferin'][k] - self.population['luciferin'][i]
  
    max_prob = 0
    #indexes = list(self.population.index)
    #indexes.remove(i)
    #for j in indexes:
    for j in self.population.loc[i, 'neigh']:
      diff_ij = self.population['luciferin'][j] - self.population['luciferin'][i]
      # prob_ij = logsumexp(diff_ij)/logsumexp(sum_n)
      prob_ij = diff_ij/sum_n
      if prob_ij>max_prob:
        max_prob = prob_ij
        self.population.loc[i, 'prob_mov'] = j
    
  def update_glowworm_mov(self, i):

    #update glowworm movement:
    if self.population['prob_mov'][i] != None:
      j = self.population['prob_mov'][i]
      for param in self.bounds.keys():
        d = np.linalg.norm(self.population[param][j] - self.population[param][i])
        d = d if d!=0 else 1e-9
        x = self.population[param][i] + self.s*((self.population[param][j] - self.population[param][i])/
                                                              d)
        x = self.clip(param, x)
        self.population.loc[i, param] = x
      self.population.loc[i, 'Fit'] = self.fitness_function(self.population.loc[i, self.bounds.keys()])
      self.population.loc[i, 'prob_mov'] = None
      if self.population['Fit'].loc[i] > self.best_fit:
        self.best += 1
        self.keep_track(self.population['Fit'].idxmax())

  def update_nei_rng(self, i):

    #update the neighborhood range
    self.population.loc[i, 'neigh_rng'] = min(self.rs, max(0, self.population['neigh_rng'][i]+self.betta*(self.n_pop - np.abs(len(self.population['neigh'][i])))))
    #clear neighbors
    self.population['neigh'].loc[i].clear()

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
    
    # luciferin_decay_const
    rho = 0.3

    #luciferin_enhacement_const
    gamma = 0.65

    #step_size
    s = 0.5

    #sensor_range
    rs = 0.45

    #neighborhood_range
    r0 = 4

    #rate of range of neighborhood
    betta = 0.075

    #intial luciferin
    l0 = 0.25


    g = GSO(rho, gamma, s, rs, r0, betta, l0, bounds, n_pop, cycles, fitness_function, p)
    g()
    g.population


