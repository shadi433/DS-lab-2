# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DkUAQuRzUxKhrQPzGMP7w2oT7mk-4YFM
"""

import pandas as pd
import numpy as np
import random
from scipy.special import logsumexp

from swarm import swarm_opt

from abc_opt import ABC
from de import DE
from gso import GSO
from fss import FSS

from params import *

np.seterr(all="ignore")

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
  data = load_digits()
  n_samples = len(data.images)
  X = data.images.reshape((n_samples, -1))
  Y = data['target']

  def fitness_function(x):
    clf = SVC(kernel='rbf', C=x[0], gamma=x[1], random_state=42)
    scores = cross_val_score(clf, X, Y, cv=5)
    return scores.mean()
  
  swarm_alg = swarm_opt(bounds, n_pop, cycles, fitness_function)

  # abc = ABC(swarm_alg)
  # de = DE(cr, swarm_alg)
  # g = GSO(rho, gamma, s, rs, r0, betta, l0, swarm_alg)
  # fss = FSS(Sinit, Sfinal, swarm_alg)

  best_para = []
  best_fit = []

  print("ABC")
  best_para_abc, best_fit_abc = ABC(swarm_alg)()

  best_para.extend(best_para_abc)
  best_fit.extend([best_fit_abc])

  print("de")
  best_para_de, best_fit_de = DE(cr, swarm_alg)()

  best_para.extend(best_para_de)
  best_fit.extend([best_fit_de])

  print("gso")
  best_para_gso, best_fit_gso = GSO(rho, gamma, s, rs, r0, betta, l0, swarm_alg)()

  best_para.extend(best_para_gso)
  best_fit.extend([best_fit_gso])

  print("fss")
  best_para_fss, best_fit_fss = FSS(Sinit, Sfinal, swarm_alg)()

  best_para.extend(best_para_fss)
  best_fit.extend([best_fit_fss])

 
  new_population = dict()
  
  for param in best_para:
    p = list(param.keys())[0]
    new_population[p] = new_population.get(p, [])+[param[p]]
  
  new_population['Fit'] = best_fit

  print("####################################################")
  print("final iteration")
  print("####################################################")

  new_swarm_alg = swarm_opt(bounds, n_pop, cycles, fitness_function, new_population)

  new_best_para = []
  new_best_fit = []

  print("ABC")
  new_best_para_abc, new_best_fit_abc = ABC(new_swarm_alg)()

  new_best_para.extend(new_best_para_abc)
  new_best_fit.extend([new_best_fit_abc])

  print("de")
  new_best_para_de, new_best_fit_de = DE(cr, new_swarm_alg)()

  new_best_para.extend(new_best_para_de)
  new_best_fit.extend([new_best_fit_de])

  print("gso")
  new_best_para_gso, new_best_fit_gso = GSO(rho, gamma, s, rs, r0, betta, l0, new_swarm_alg)()

  new_best_para.extend(new_best_para_gso)
  new_best_fit.extend([new_best_fit_gso])

  print("fss")
  new_best_para_fss, new_best_fit_fss = FSS(Sinit, Sfinal, new_swarm_alg)()

  new_best_para.extend(new_best_para_fss)
  new_best_fit.extend([new_best_fit_fss])


  new_new_population = dict()

  for param in new_best_para:
    p = list(param.keys())[0]
    new_new_population[p] = new_new_population.get(p, [])+[param[p]]

  new_new_population['Fit'] = new_best_fit


  print("####################################################")

  result = []
  for p in bounds.keys():
    result.append(np.mean(new_new_population[p]))
  
  return tuple(result)