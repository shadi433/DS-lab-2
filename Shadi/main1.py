from ACO_from_scrach import *
import random
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn import svm

# data importing 
data = load_digits() 

n_samples = len(data.images)
X = data.images.reshape((n_samples, -1))
Y = data['target']

nodes = []
for _ in range(20):
  x = random.uniform(1.0, 10.0)
  y = random.uniform(0.0001, 0.1)
  nodes.append((x, y))

def fitness_function(x):  
	# x[0] = C and x[1] = gamma
	clf = svm.SVC(kernel='rbf', C=x[0], gamma=x[1], random_state=42)
	scores = cross_val_score(clf, X, Y, cv=5)
	
	return scores.mean()  
#------------------------------------------------------------------------------------------------------------
world = World(nodes, fitness_function)

solver = Solver()

solution = solver.solve(world)

print(solution.distance)