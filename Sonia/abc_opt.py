# -*- coding: utf-8 -*-
"""abc.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KmHEQKl5Rp7pZcARsXk4xN1Bbtha2vX_
"""

import pandas as pd
import numpy as np
import random
from swarm import swarm_opt

class ABC(swarm_opt):
  def __init__(self, swarm):
    super().__init__(swarm.bounds, swarm.n_pop, swarm.cycles, swarm.fitness_function)
    self.population['trial'] = 0
    self.population['prob'] = 0

    self.dim = len(self.bounds.keys())
    #defining the limit of the trial vector = (Np * D)
    self.limit = self.n_pop * self.dim 

  def __call__(self):
    
    self.keep_track(self.population['Fit'].idxmax())
    
    for _ in range(self.cycles):

      #produce new solution Vij for employed beed
      self.produce_new_sol()
      
      #we suppose that the onlooker bees will be in the same position
      #as the employed bees
      
      #produce new solutions Vij for the onlookers
      #from the solutions selected by Prob
      self.produce_new_sol(onlookers=True)

      #generate new solution for the scout bees if exists:
      scout_indexes = self.population[self.population['trial']>=self.limit].index.values
      if len(scout_indexes) == 0:
        continue
      
      self.produce_new_sol_scout(scout_indexes)
    return self.best_para, self.best_fit

  def produce_new_sol(self, onlookers=False):
    for i, _ in self.population.iterrows():
        if onlookers == True and random.uniform(0,1) > self.population.loc[i, 'prob']:
          continue
        indexes = list(self.population.index)
        indexes.remove(i)
        k = random.choice(indexes)
        param = list(set(random.choices(list(self.bounds.keys()), k = len(self.bounds.keys()))))
        v = self.population[list(self.bounds.keys())].loc[i].to_dict()
        for p in param:
          x = self.population[p].loc[i] + random.uniform(0,1)*(self.population[p].loc[i] - self.population[p].loc[k])
          v[p] = self.clip(p, x)
        
        #evaluation of the new generated solution and update of the trial vector
        fit = self.fitness_function(list(v.values()))
        if fit > self.population['Fit'].loc[i]:
          for p in param:
            self.population.loc[i, [p]] = v[p]
          self.population.loc[i, 'Fit'] = fit
          self.population.loc[i, 'trial'] = 0
        else:
          self.population.loc[i, 'trial'] = self.population['trial'].loc[i] + 1
      
    if self.population['Fit'].max()> self.best_fit:
      self.best += 1
      self.keep_track(self.population['Fit'].idxmax())
      
    #Probability values for the solutions Xij
    self.population['prob'] = [f/sum(self.population['Fit']) for f in self.population['Fit']] 
  
  def produce_new_sol_scout(self, scout_indexes):
    for idx in scout_indexes:
      for param in self.bounds.keys():
        x_min = self.population[param].min()
        x_max = self.population[param].max()
        x = x_min + random.uniform(0,1)*(x_max - x_min)
        self.population.loc[idx, [param]] = self.clip(param, x)
      self.population.loc[idx, 'Fit'] = self.fitness_function(self.population.loc[idx, [self.bounds.keys()]])
      self.population.loc[idx, 'trial'], self.population.loc[idx, 'prob'] = 0,0
    
    if self.population['Fit'].max()> self.best_fit:
      self.best += 1
      self.keep_track(self.population['Fit'].idxmax())

if __name__ == '__main__':
  from sklearn.datasets import load_digits
  from sklearn.svm import SVC
  from sklearn.model_selection import cross_val_score

  data = load_digits()
  n_samples = len(data.images)
  X = data.images.reshape((n_samples, -1))
  Y = data['target']

  def fitness_function(x):
    clf = SVC(kernel='rbf', C=x[0], gamma=x[1], random_state=42)
    scores = cross_val_score(clf, X, Y, cv=5)

    return scores.mean()
  
  bounds = {"C": [0.001, 10.0], "gamma": [0.0001, 0.1]}
  n_pop = 10
  cycles = 10

  swarm = swarm_opt(bounds, n_pop, cycles, fitness_function)

  abc = ABC(swarm)
  abc()