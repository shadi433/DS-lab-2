# %%
import pandas as pd
import numpy as np
import random
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# %%
class FSS:
  def __init__(self, Sinit, Sfinal, bounds, n_pop, cycles, fitness_function, population=None, old_pop=None):
    
    self.bounds = bounds
    self.n_pop = n_pop
    self.cycles = cycles
    self.fitness_function = fitness_function
    self.ret1 = False
    
    self.old_pop = old_pop.copy()
    
    if type(self.old_pop)==list:
      self.population = pd.DataFrame(self.old_pop, columns=self.bounds.keys())
      self.population['Fit'] = [self.fitness_function(x) for x in self.old_pop]
      self.ret1 = True
    else:
      self.population = population.copy()

    self.population['w'] = 1 
    
    # keep track of best solution
    self.best = 0
    self.best_fit = 0
    self.best_para = 0    
    
    self.dim = len(self.bounds.keys())

    self.Sinit = Sinit
    self.Sfinal = Sfinal
    self.Sind = self.Sinit

    self.old_population = self.population.copy()

  def __call__(self):
    self.keep_track(self.population['Fit'].idxmax())

    for _ in range(self.cycles):

      self.update_position_sind()
      delta_fit = self.update_weight()
      self.update_position_col_ins_mov(delta_fit)
      B = self.baryCenter()
      self.update_position_col_vol_mov(B)
    
    # return self.best_para, self.best_fit
    # return self.population[self.bounds.keys()].values.tolist(), self.population['Fit'].values.tolist()
    
    if self.ret1 == True: 
      return self.population[self.bounds.keys()].values.tolist(), self.population['Fit'].values.tolist()
    else: 
      return self.best_para, self.best_fit
    
  def update_position_sind(self):
    #update position based on Sind
    self.Sind = {p: self.Sind[p] - (self.Sinit[p]- self.Sfinal[p])/self.cycles for p in self.bounds.keys()}
    for i in range(self.n_pop):
      para = {}
      for p in self.bounds.keys():
        x = self.population[p][i] + random.uniform(0,1)*self.Sind[p]
        para[p] = self.clip(p, x)
      fit = self.fitness_function(list(para.values()))
      # fit = self.fitness_function(self.population[self.bounds.keys()].values[i])
      if fit > self.population.loc[i, 'Fit']:
        for p in para.keys():
          self.population.loc[i, p] = para[p]
        self.population.loc[i, 'Fit'] = fit
    
    if self.population['Fit'].max()>self.best_fit:
      self.best += 1
      self.keep_track(self.population['Fit'].idxmax())
  
  def update_weight(self):
    #update weight
    delta_fit = [np.abs(self.population['Fit'][i] - self.old_population['Fit'][i]) for i in range(self.n_pop)]
    for i in range(self.n_pop):
        m = max(delta_fit) if max(delta_fit)!=0 else 1
        w = self.population['w'][i] + delta_fit[i]/m
        self.population.loc[i, 'w'] = self.clip('w', w)
    return delta_fit

    
  def update_position_col_ins_mov(self, delta_fit):
    #update position based on their collective instinctive mov
    #avrg ind mov
    delta_fit_population = dict()
    I = dict()
    for p in self.bounds.keys():
      delta_fit_population[p] = [(self.population[p][i] - self.old_population[p][i])*delta_fit[i] for i in range(self.n_pop)]
      s = sum(delta_fit) if sum(delta_fit) != 0 else 1
      I[p] = sum(delta_fit_population[p])/s
      
    # for i in range(self.n_pop):
    #   for p in self.bounds.keys():
    #     x = self.population[p][i] + I[p]
    #     self.population.loc[i, p] = clip(p, x)
    #   self.population.loc[i, 'Fit'] = self.fitness_function(self.population[self.bounds.keys()].values[i])
        
    
    for i in range(self.n_pop):
      para = dict()
      for p in self.bounds.keys():
        x = self.population[p][i] + I[p]
        para[p] = self.clip(p, x)        
      fit = self.fitness_function(list(para.values()))
      if fit >= self.population.loc[i, 'Fit']:
        for p in para.keys():
          self.population.loc[i, p] = para[p]
        self.population.loc[i, 'Fit']= fit
        self.fitness_function(self.population[self.bounds.keys()].values[i])

    if self.population['Fit'].max()>self.best_fit:
      self.best += 1
      self.keep_track(self.population['Fit'].idxmax())
    self.old_population = self.population.copy()
      

  def baryCenter(self):
    #BaryCenter
    pop_weight = dict()
    B = dict()
    for p in self.bounds.keys():
      pop_weight[p] = [(self.population[p][i] * self.population['w'][i]) for i in range(self.n_pop)]
      B[p] = sum(pop_weight[p])/sum(self.population['w'])
    return B
  
  def update_position_col_vol_mov(self, B):
    #collective volatile mov
    Svol = {p: 2*self.Sind[p] for p in self.Sind.keys()}
    op = 1 if sum(self.old_population['w']) > sum(self.population['w']) else -1
    # #updates fishes position
    # for i in range(self.n_pop):
    #   for p in self.bounds.keys():
    #     x = self.population[p][i] + op*Svol[p]*random.uniform(0,1)*((self.population[p][i] - B[p])/(np.sqrt((self.population[p][i] - B[p])**2)))
    #     self.population.loc[i, p] = self.clip(p, x)
    #   self.population.loc[i, 'Fit']= self.fitness_function(self.population[self.bounds.keys()].values[i])
      
    for i in range(self.n_pop):
      para = dict()
      for p in self.bounds.keys():
        x = self.population[p][i] + op*Svol[p]*random.uniform(0,1)*((self.population[p][i] - B[p])/(np.sqrt((self.population[p][i] - B[p])**2)))
        para[p] = self.clip(p, x)        
      fit = self.fitness_function(list(para.values()))
      if fit >= self.population.loc[i, 'Fit']:
        for p in para.keys():
          self.population.loc[i, p] = para[p]
        self.population.loc[i, 'Fit']= fit
        self.fitness_function(self.population[self.bounds.keys()].values[i])
    
    if self.population['Fit'].max()>self.best_fit:
      self.best += 1
      self.keep_track(self.population['Fit'].idxmax())

  def keep_track(self, best_sol_index):
    self.best_para = [{key: self.population[key].loc[best_sol_index]} for key in self.bounds.keys()]
    self.best_fit = self.population["Fit"].loc[best_sol_index]
  
  def clip(self, param, x):
    if param == 'w':
      return max(1, min(x, 2))
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
    
    #initial step
    Sinit = {p: bounds[p][0] for p in bounds.keys()}

    #final step
    Sfinal = {p: bounds[p][1] for p in bounds.keys()}

    fss = FSS(Sinit, Sfinal, bounds, n_pop, cycles, fitness_function, p)
    fss()
    fss.population


