import numpy as np
import matplotlib.pyplot as plt
from math import gamma

class CSO:

    def __init__(self, fitness_function, bound=None, n_pop=10, pop=[], n=2, pa=0.25, beta=1.5, 
                plot=False, verbose=False, Generations=20):

        '''
        PARAMETERS:
        
        fitness_function: A FUNCTION WHICH EVALUATES COST (OR THE FITNESS) VALUE
        bound: AXIS BOUND FOR EACH DIMENSION
        n_pop: POPULATION SIZE
        n: TOTAL DIMENSIONS
        pa: ASSIGNED PROBABILITY
        beta: LEVY PARAMETER
        Generations: MAXIMUM ITERATION
        best: GLOBAL BEST POSITION OF SHAPE (n,1)
        
        '''
        self.fitness_function = fitness_function
        self.bound = bound
        self.n_pop = n_pop 
        self.pop = pop
        self.n = n
        self.pa = pa
        self.beta = beta
        self.plot = plot
        self.verbose = verbose
        self.Generations = Generations

        self.clip_pop()  #<-------------------------------------I don't think I need it

    def update_position_1(self):
        
        '''
        TO CALCULATE THE CHANGE OF POSITION 'X = X + rand*C' USING LEVY FLIGHT METHOD
        C = 0.01*S*(X-best) WHERE S IS THE RANDOM STEP
        
        '''

        num = gamma(1+self.beta)*np.sin(np.pi*self.beta/2)
        den = gamma((1+self.beta)/2)*self.beta*(2**((self.beta-1)/2))
        segma_u = (num/den)**(1/self.beta)
        segma_v = 1
        u = np.random.normal(0, segma_u, self.n)
        v = np.random.normal(0, segma_v, self.n)
        S = u/(np.abs(v)**(1/self.beta))

        # DEFINING GLOBAL BEST SOLUTION BASED ON FITNESS VALUE

        for i in range(self.n_pop):
            if i==0:
                self.best = self.pop[i,:].copy()
            else:
                self.best = self.optimum(self.best, self.pop[i,:])

        Xnew = self.pop.copy()
        for i in range(self.n_pop):
            Xnew[i,:] += np.random.randn(self.n)*0.01*S*(Xnew[i,:]-self.best)
            if self.bound is not None:
                for j in range(self.n):
                    xmin, xmax = self.bound[j]
                    Xnew[:,j] = np.clip(Xnew[:,j], xmin, xmax)
            self.pop[i,:] = self.optimum(Xnew[i,:], self.pop[i,:])
            self.clip_pop()

    def update_position_2(self):
        
        '''
        TO REPLACE SOME NEST WITH NEW SOLUTIONS
        HOST BIRD CAN THROW EGG AWAY (ABANDON THE NEST) WITH FRACTION
        pa ∈ [0,1] (ALSO CALLED ASSIGNED PROBABILITY) AND BUILD A COMPLETELY 
        NEW NEST. FIRST WE CHOOSE A RANDOM NUMBER r ∈ [0,1] AND IF r < pa,
        THEN 'pop' IS SELECTED AND MODIFIED ELSE IT IS KEPT AS IT IS. 
        '''

        Xnew = self.pop.copy()
        Xold = self.pop.copy()
        for i in range(self.n_pop):
            d1,d2 = np.random.randint(0,5,2)
            for j in range(self.n):
                r = np.random.rand()
                if r < self.pa:
                    Xnew[i,j] += np.random.rand()*(Xold[d1,j]-Xold[d2,j])
                    if self.bound is not None:
                        for k in range(self.n):
                            xmin, xmax = self.bound[k]
                            Xnew[:,k] = np.clip(Xnew[:,k], xmin, xmax)
            self.pop[i,:] = self.optimum(Xnew[i,:], self.pop[i,:])
    
    def optimum(self, best, pop):

        if self.fitness_function(best) < self.fitness_function(pop):
            best = pop.copy()
            
        return best

    def clip_pop(self):

        # IF BOUND IS SPECIFIED THEN CLIP 'X' VALUES SO THAT THEY ARE IN THE SPECIFIED RANGE
        
        if self.bound is not None:
            for i in range(self.n):
                xmin, xmax = self.bound[i]
                self.pop[:,i] = np.clip(self.pop[:,i], xmin, xmax)

    def execute(self):

        '''
        t: ITERATION NUMBER
        fitness_time: LIST STORING FITNESS (OR COST) VALUE FOR EACH ITERATION
        time: LIST STORING ITERATION NUMBER ([0,1,2,...])
        
        THIS FUNCTION EXECUTES CUCKOO SEARCH ALGORITHM
        '''

        self.fitness_time, self.time = [], []

        for t in range(self.Generations):
            self.update_position_1()
            self.clip_pop()
            self.update_position_2()
            self.clip_pop()
            self.fitness_time.append(self.fitness_function(self.best))
            self.time.append(t)
            if self.verbose:
                print('Iteration:  ',t,'| best global fitness (cost):',round(self.fitness_function(self.best),7))

        print('\nOPTIMUM SOLUTION\n  >', np.round(self.best.reshape(-1),7).tolist())
        print('\nOPTIMUM FITNESS\n  >', np.round(self.fitness_function(self.best),7))
        print()
        if self.plot:
            self.Fplot()

    def get_best(self):
        return self.best  # <-------------------------------------maybe it needs reshape

    def get_best_fit(self):
        return np.round(self.fitness_function(self.best),7)
        
    def Fplot(self):

        # PLOTS GLOBAL FITNESS (OR COST) VALUE VS ITERATION GRAPH
        
        plt.plot(self.time, self.fitness_time)
        plt.title('Fitness value vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness value')
        plt.show()