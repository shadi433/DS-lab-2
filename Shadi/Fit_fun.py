from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score

import numpy as np

bounds = [[1.0, 10.0], [0.0001, 0.1]]        
def clip_pop(pop):
	# IF BOUND IS SPECIFIED THEN CLIP 'pop' VALUES SO THAT THEY ARE IN THE SPECIFIED RANGE
	if bounds is not None:
		for i in range(2):
			xmin, xmax = bounds[i]
			pop[:,i] = np.clip(pop[:,i], xmin, xmax)

data = load_digits() 

n_samples = len(data.images)
X = data.images.reshape((n_samples, -1))
Y = data['target']

# modle implementation
def fitness_function(x):  
	# x[0] = C and x[1] = gamma
	clf = svm.SVC(kernel='rbf', C=x[0], gamma=x[1], random_state=42)
	scores = cross_val_score(clf, X, Y, cv=5)
	
	return scores.mean()  