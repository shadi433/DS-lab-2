# -*- coding: utf-8 -*-
"""params.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dayHUOxyD1Zrp7joQWUxRTdp-02KtO25
"""

#bounds
bounds = {"C": [0.001, 10.0], "gamma": [0.0001, 0.1]}

#population size
n_pop = 10

#maximum iteration
cycles = 10

#DE params
#crossover rate
cr = 0.9

#GSO params
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

#fss params
#initial step
Sinit = 0.1

#final step
Sfinal = 0.0001