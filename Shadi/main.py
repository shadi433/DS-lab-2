from JSI import *

model = JSI(Generations=10, n_pop= 10)
model.run()
best_para, best_score, best_model = model.best_all()
print(f'best parameters:{best_para}, best score:{best_score}, best_model:{best_model}')