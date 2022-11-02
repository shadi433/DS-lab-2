# -*- coding: utf-8 -*-
"""ABC.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1q5MAJOIvm5Yv28E3pQAJggRNEyOw1WYt
"""

import pandas as pd
import numpy as np
import random
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

def Fit(fx):
  return (1/(1+fx)) if fx>=0 else (1+np.abs(fx))

# define range for input
bounds = [[1.0, 10.0], [0.0001, 0.1]]

# define the total iterations
Cycles = 10

# define the population size
n_pop = 10

def ABC(bounds, n_pop, Cycles, fitness_function, Fit=Fit):

  #creation of the population dataframe
  emp_pop = pd.DataFrame()
  
  #defining the limit of the trial vector = (Np * D)

  limit = n_pop * len(bounds)
  
  #population generation
  XCmin, XGmin = bounds[0][0], bounds[1][0]
  XCmax, XGmax  = bounds[0][1], bounds[1][1]

  emp_pop_list = list()
  for _ in range(n_pop):

    xc = XCmin + random.uniform(0,1)*(XCmax - XCmin)
    xg = XGmin + random.uniform(0,1)*(XGmax - XGmin)
    emp_pop_list.append((xc, xg))
  
  #creation of the dataframe with the employed bees
  emp_pop[["C", "gamma"]] = list(emp_pop_list)

  #initialization of the trial vector
  emp_pop['trial'] = 0

  #calculating the Fit of the initialised population (employed bees)
  emp_pop["Fit"] = [fitness_function(params) for params in emp_pop[["C", "gamma"]].values]

  # keep track of best solution
  best = 0
  best_sol_index = emp_pop["Fit"].idxmax()
  best_para, best_fit = (emp_pop["C"].loc[best_sol_index],
                         emp_pop["gamma"].loc[best_sol_index]), emp_pop["Fit"].loc[best_sol_index]

  print(">%d, new best C and gamma:%s, best_fit %f" % (best,  best_para, best_fit))
  #setting all the population as employed bees
  #emp_pop["scout"] = 0
  emp_pop["prob"] = 0

  params = ["C","gamma"]
  
  print("init done, cycle entering")
  for _ in range(Cycles):
    #produce new solution Vij for employed beed
    for i, (c, gamma, trial, f, p) in emp_pop.iterrows():
      vij = [c,gamma]
      #selection of random candidate
      indexes = list(emp_pop.index)
      indexes.remove(i)
      k = random.choice(indexes)
      param = list(set(random.choices(params, k = 2)))
      for p in param:
        if p == "C":
          vj = max(bounds[0][0],min(bounds[0][1],emp_pop[p].loc[i] + random.uniform(0,1)*(emp_pop[p].loc[i] - emp_pop[p].loc[k])))
        else:
          vj = max(bounds[1][0], min(bounds[1][1], emp_pop[p].loc[i] + random.uniform(0,1)*(emp_pop[p].loc[i] - emp_pop[p].loc[k])))
        vij[params.index(p)] = vj
      #evaluation of the new generated solution and update of the trial vector
      fit = fitness_function(vij)
      if fit > f:
        emp_pop['C'].loc[i], emp_pop['gamma'].loc[i] = vij[0], vij[1]
        emp_pop['trial'].loc[i] = 0
      else:
        emp_pop['trial'].loc[i] = trial + 1
        # if emp_pop['trial'].loc[i]>=limit:
        #   emp_pop['scout'].loc[i] = 1
    
    #keeping tracking the best solution:
    if emp_pop["Fit"].max()>best_fit:
      best += 1
      best_sol_index = emp_pop["Fit"].idxmax()
      best_para, best_fit = (emp_pop["C"].loc[best_sol_index],
                            emp_pop["gamma"].loc[best_sol_index]), emp_pop["Fit"].loc[best_sol_index]
      print(">%d, new best C and gamma:%s, best_fit %f" % (best,  best_para, best_fit))
      
    #Probability values for the solutions Xij
    emp_pop["prob"] = [f/sum(emp_pop['Fit']) for f in emp_pop['Fit']]

    #we suppose that the onlooker bees will be in the same position
    #as the employed bees
  

    #produce new solutions Vij for the onlookers
    #from the solutions selected by Prob
    for i, (c, gamma, trial, f, p) in emp_pop.iterrows():
      if random.uniform(0,1) > p:
        continue
      vij = [c,gamma]
      #selection of random candidate
      indexes = list(emp_pop.index)
      indexes.remove(i)
      k = random.choice(indexes)
      param = list(set(random.choices(params, k = 2)))
      for p in param:
        if p == "C":
          vj = max(bounds[0][0],min(bounds[0][1],emp_pop[p].loc[i] + random.uniform(0,1)*(emp_pop[p].loc[i] - emp_pop[p].loc[k])))
        else:
          vj = max(bounds[1][0], min(bounds[1][1], emp_pop[p].loc[i] + random.uniform(0,1)*(emp_pop[p].loc[i] - emp_pop[p].loc[k])))
        vij[params.index(p)] = vj
      #evaluation of the new generated solution and update of the trial vector
      fit = fitness_function(vij)
      if fit > f:
        emp_pop['C'].loc[i], emp_pop['gamma'].loc[i] = vij[0], vij[1]
        emp_pop['trial'].loc[i] = 0
      else:
        emp_pop['trial'].loc[i] = trial + 1
        # if emp_pop['trial'].loc[i]>=limit:
        #   emp_pop['scout'].loc[i] = 1
    
    #keeping tracking the best solution:
    if emp_pop["Fit"].max()>best_fit:
      best += 1
      best_sol_index = emp_pop["Fit"].idxmax()
      best_para, best_fit = (emp_pop["C"].loc[best_sol_index],
                            emp_pop["gamma"].loc[best_sol_index]), emp_pop["Fit"].loc[best_sol_index]
      
      print(">%d, new best C and gamma:%s, best_fit %f" % (best,  best_para, best_fit))
    


    #generate new solution for the scout bees if exists:
    scout_indexes = emp_pop[emp_pop['trial']>=limit].index.values
    if len(scout_indexes) == 0:
      continue
    
    XCmin = emp_pop['C'].min()
    XGmin = emp_pop['gamma'].min()
    
    XCmax = emp_pop['C'].max()
    XGmax = emp_pop['gamma'].max()

    for index in scout_indexes:
      emp_pop["C"].loc[index] = max(XCmin,min(XCmax,XCmin + random.uniform(0,1)*(XCmax - XCmin)))
      c = emp_pop["C"].loc[index]
      emp_pop["gamma"].loc[index] = max(XGmin, min(XGmax,XGmin + random.uniform(0,1)*(XGmax - XGmin)))
      gamma = emp_pop["gamma"].loc[index]
      emp_pop["Fit"].loc[index] = Fit(fitness_function([c, gamma]))
      emp_pop["trial"].loc[index], emp_pop["prob"].loc[index] = 0,0

    #keeping tracking the best solution:
    if emp_pop["Fit"].max()>best_fit:
    
      best += 1
      best_sol_index = emp_pop["Fit"].idxmax()
      best_para, best_fit = (emp_pop["C"].loc[best_sol_index],
                            emp_pop["gamma"].loc[best_sol_index]), emp_pop["Fit"].loc[best_sol_index]
      print(">%d, new best C and gamma:%s, best_fit %f" % (best,  best_para, best_fit))
  return best_para