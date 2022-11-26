# -*- coding: utf-8 -*-
"""swarm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17_SCeg36kLI_kTV26Ho8eNCbpHFi55aI
"""

import pandas as pd
import numpy as np
import random

class swarm_opt:
  def __init__(self, bounds, n_pop, cycles, fitness_function, population=None):
    self.bounds = bounds
    self.n_pop = n_pop
    self.cycles = cycles
    self.fitness_function = fitness_function

    if population == None:
      #populaiton generation
      self.pop_dict = self.init_pop()
      self.pop = pd.DataFrame.from_dict(self.pop_dict)
      #calculating the Fit of the initialised population (employed bees)
      self.pop['Fit'] = [fitness_function(x) for x in list(zip(*self.pop_dict.values()))]
      
    else:
      self.pop = pd.DataFrame.from_dict(self.population)
      # self.pop['Fit'] = [fitness_function(x) for x in list(zip(*self.population.values()))]
      # self.pop['trial'] = 0
    self.population = self.pop

    # keep track of best solution
    self.best = 0
    self.best_fit = 0
    self.best_para = 0
    # self.population["prob"] = 0
  
  def init_pop(self):
    pop_dict = dict()
    for i in range(self.n_pop):
      for key in self.bounds.keys():
        if i == 0:
          pop_dict[key] = [self.bounds[key][0] + random.uniform(0,1)*(self.bounds[key][-1] - self.bounds[key][0])]
        else:
          pop_dict[key].append(self.bounds[key][0] + random.uniform(0,1)*(self.bounds[key][-1] - self.bounds[key][0]))
    return pop_dict

  def keep_track(self, best_sol_index):
    self.best_para = [{key: self.population[key].loc[best_sol_index]} for key in self.bounds.keys()]
    self.best_fit = self.population["Fit"].loc[best_sol_index]
    # print(f"{self.best}, best fit: {self.best_fit}.")
    # print(f'best params: ', self.best_para)

  def clip(self, param, x):
    return max(self.bounds[param][0], min(self.bounds[param][-1], x))

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
