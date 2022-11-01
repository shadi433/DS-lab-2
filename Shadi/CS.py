import numpy as np
import matplotlib.pyplot as plt
from math import gamma

class CSO:

    def __init__(self, fitness, n_pop=10, n=2, pa=0.25, beta=1.5, bound=None, 
                plot=False, verbose=False, Generations=20):

        '''
        PARAMETERS:
        
        fitness: A FUNCTION WHICH EVALUATES COST (OR THE FITNESS) VALUE
        P: POPULATION SIZE
        n: TOTAL DIMENSIONS
        pa: ASSIGNED PROBABILITY
        beta: LEVY PARAMETER
        bound: AXIS BOUND FOR EACH DIMENSION
        X: PARTICLE POSITION OF SHAPE (P,n)
        Generations: MAXIMUM ITERATION
        best: GLOBAL BEST POSITION OF SHAPE (n,1)
        
        '''
        self.fitness = fitness
        self.P = n_pop 
        self.n = n
        self.Generations = Generations
        self.pa = pa
        self.beta = beta
        self.bound = bound
        self.plot = plot
        self.verbose = verbose

        # X = (U-L)*rand + L (U AND L ARE UPPER AND LOWER BOUND OF X)
        # U AND L VARY BASED ON THE DIFFERENT DIMENSION OF X

        self.X = []

        if bound is not None:
            for (U, L) in bound:
                x = (U-L)*np.random.rand(n_pop,) + L 
                self.X.append(x)
            self.X = np.array(self.X).T
        else:
            self.X = np.random.randn(n_pop,n)

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

        for i in range(self.P):
            if i==0:
                self.best = self.X[i,:].copy()
            else:
                self.best = self.optimum(self.best, self.X[i,:])

        Xnew = self.X.copy()
        for i in range(self.P):
            Xnew[i,:] += np.random.randn(self.n)*0.01*S*(Xnew[i,:]-self.best) 
            self.X[i,:] = self.optimum(Xnew[i,:], self.X[i,:])

    def update_position_2(self):
        
        '''
        TO REPLACE SOME NEST WITH NEW SOLUTIONS
        HOST BIRD CAN THROW EGG AWAY (ABANDON THE NEST) WITH FRACTION
        pa ∈ [0,1] (ALSO CALLED ASSIGNED PROBABILITY) AND BUILD A COMPLETELY 
        NEW NEST. FIRST WE CHOOSE A RANDOM NUMBER r ∈ [0,1] AND IF r < pa,
        THEN 'X' IS SELECTED AND MODIFIED ELSE IT IS KEPT AS IT IS. 
        '''

        Xnew = self.X.copy()
        Xold = self.X.copy()
        for i in range(self.P):
            d1,d2 = np.random.randint(0,5,2)
            for j in range(self.n):
                r = np.random.rand()
                if r < self.pa:
                    Xnew[i,j] += np.random.rand()*(Xold[d1,j]-Xold[d2,j]) 
            self.X[i,:] = self.optimum(Xnew[i,:], self.X[i,:])
    
    def optimum(self, best, particle_x):

        if self.fitness(best) < self.fitness(particle_x):
            best = particle_x.copy()
            
        return best

    def clip_X(self):

        # IF BOUND IS SPECIFIED THEN CLIP 'X' VALUES SO THAT THEY ARE IN THE SPECIFIED RANGE
        
        if self.bound is not None:
            for i in range(self.n):
                xmin, xmax = self.bound[i]
                self.X[:,i] = np.clip(self.X[:,i], xmin, xmax)

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
            self.clip_X()
            self.update_position_2()
            self.clip_X()
            self.fitness_time.append(self.fitness(self.best))
            self.time.append(t)
            if self.verbose:
                print('Iteration:  ',t,'| best global fitness (cost):',round(self.fitness(self.best),7))

        print('\nOPTIMUM SOLUTION\n  >', np.round(self.best.reshape(-1),7).tolist())
        print('\nOPTIMUM FITNESS\n  >', np.round(self.fitness(self.best),7))
        print()
        if self.plot:
            self.Fplot()
        
    def Fplot(self):

        # PLOTS GLOBAL FITNESS (OR COST) VALUE VS ITERATION GRAPH
        
        plt.plot(self.time, self.fitness_time)
        plt.title('Fitness value vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness value')
        plt.show()