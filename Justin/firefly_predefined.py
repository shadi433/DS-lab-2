# -*- coding: utf-8 -*-
"""firefly_predefined.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1u9Uf382gvVlLOVl-ZfRpSVuMnfnDGjQ_
"""

pip install sklearn-nature-inspired-algorithms

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

data = load_digits()

n_samples = len(data.images)
X = data.images.reshape((n_samples, -1))
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle= True, random_state= 42)

from sklearn_nature_inspired_algorithms.model_selection import NatureInspiredSearchCV

clf = svm.SVC()

param_grid = {'C':[0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001], 'kernel': ('linear', 'rbf')}

nia_search = NatureInspiredSearchCV(
    clf,
    param_grid,
    algorithm='fa', # hybrid bat algorithm
    population_size=50,
    max_n_gen=100,
    max_stagnating_gen=10,
    runs=5,
    random_state=None,
    cv=5 # or any number if you want same results on each run
)

nia_search.fit(X_train, y_train)

from sklearn.metrics import classification_report
clf = svm.SVC(**nia_search.best_params_)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, digits=4))

from sklearn_nature_inspired_algorithms.helpers import score_by_generation_lineplot, score_by_generation_violinplot

# line plot will plot all of the runs, you can specify the metric to be plotted ('min', 'max', 'median', 'mean')
score_by_generation_lineplot(nia_search, metric='max')

# in violin plot you need to specify the run to be plotted
score_by_generation_violinplot(nia_search, run=0)