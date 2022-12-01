import random
import bisect
import functools
import itertools
from copy import copy


#--------------------------------------------------------------------------------------------------------------
class World:

    def __init__(self, nodes, fit_fun):
        self._nodes = nodes
        self.fit_fun = fit_fun
        self.edges = self.create_edges()

    @property
    def nodes(self):
        """Node IDs."""
        return list(range(len(self._nodes)))

    def create_edges(self):
        
        edges = {}
        for m in self.nodes:
            for n in self.nodes:
                a, b = self.data(m), self.data(n)
                if a != b:
                    edge = Edge(a, b, length=self.fit_fun(b))
                    edges[m, n] = edge
        return edges
        
    def reset_pheromone(self, level=0.01):
    
        for edge in self.edges.values():
            edge.pheromone = level
        
    def data(self, idx, idy=None):

        try:
            if idy is None:
                return self._nodes[idx]
            else:
                return self.edges[idx, idy]
        except IndexError:
            return None

#--------------------------------------------------------------------------------------------------------------
class Edge:
   
    def __init__(self, start, end, length=None, pheromone=None):
        self.start = start
        self.end = end
        self.length = 1 if length is None else length
        self.pheromone = 0.1 if pheromone is None else pheromone
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

#--------------------------------------------------------------------------------------------------------------
@functools.total_ordering
class Ant:
    uid = 0

    def __init__(self, alpha=1, beta=3):
        """Create a new Ant for the given world.
        float alpha: the relative importance of pheromone (default=1)
        float beta: the relative importance of distance(fittness) (default=3)
        """
        self.world = None
        self.alpha = alpha
        self.beta = beta
        self.start = None
        self.distance = 0
        self.visited = []
        self.unvisited = []
        self.traveled = []

    def initialize(self, world, start=None):
        """Reset everything so that a new solution can be found.

        :param World world: the world to solve
        :param Node start: the starting node (default is chosen randomly)
        :return: `self`
        :rtype: :class:`Ant`
        """
        self.world = world
        if start is None:
            self.start = random.randrange(len(self.world.nodes))
        else:
            self.start = start
        self.distance = 0
        self.visited = [self.start]
        self.unvisited = [n for n in self.world.nodes if n != self.start]
        self.traveled = []
        return self

    @property
    def node(self):
        # Most recently visited node.
        try:
            return self.visited[-1]
        except IndexError:
            return None

    @property
    def path(self):
        # Edges traveled by the :class:`Ant` in order.
        return [edge for edge in self.traveled] 

    def __eq__(self, other):
        """Return ``True`` if the distance is equal to the other distance.
        :param Ant other: right-hand argument
        :rtype: bool
        """
        return self.distance == other.distance

    def __lt__(self, other):
        """Return ``True`` if the distance is less than the other distance.
        :param Ant other: right-hand argument
        :rtype: bool
        """
        return self.distance > other.distance       

    def can_move(self):
        """Return ``True`` if there are moves that have not yet been made.
        :rtype: bool
        """
        # This is only true after we have made the move back to the starting
        # node.
        return len(self.traveled) != len(self.visited)

    def move(self):
        """Choose, make, and return a move from the remaining moves.
        :return: the :class:`Edge` taken to make the move chosen
        :rtype: :class:`Edge`
        """
        remaining = self.remaining_moves()
        choice = self.choose_move(remaining)
        return self.make_move(choice)

    def remaining_moves(self):
        """Return the moves that remain to be made.
        :rtype: list
        """
        return self.unvisited

    def choose_move(self, choices):
        """Choose a move from all possible moves.
        :param list choices: a list of all possible moves
        :return: the chosen element from *choices*
        :rtype: node
        """
        if len(choices) == 0:
            return None
        if len(choices) == 1:
            return choices[0]
        
        # Find the weight of the edges that take us to each of the choices.
        weights = []
        for move in choices:
            edge = self.world.edges[self.node, move]
            weights.append(self.weigh(edge))
        
        # Choose one of them using a weighted probability.
        total = sum(weights)
        cumdist = list(itertools.accumulate(weights)) + [total]
        return choices[bisect.bisect(cumdist, random.random() * total)]

    def make_move(self, dest):
        # Since self.node simply refers to self.visited[-1], which will be
        # changed before we return to calling code, store a reference now.
        ori = self.node

        # When dest is None, all nodes have been visited but we may not
        # have returned to the node on which we started. If we have, then
        # just do nothing and return None. Otherwise, set the dest to the
        # node on which we started and don't try to move it from unvisited
        # to visited because it was the first one to be moved.
        if dest is None:
            if self.can_move() is False:
                return None
            dest = self.start   # last move is back to the start
        else:
            self.visited.append(dest)
            self.unvisited.remove(dest)
        
        edge = self.world.edges[ori, dest]
        self.traveled.append(edge)
        self.distance += edge.length
        return edge

    def weigh(self, edge):
        pre = 1 / (edge.length or 1)
        post = edge.pheromone
        return post ** self.alpha * pre ** self.beta
#--------------------------------------------------------------------------------------------------------------
class Solver:
    
    def __init__(self, **kwargs):
        self.alpha = kwargs.get('alpha', 1)
        self.beta = kwargs.get('beta', 3)
        self.rho = kwargs.get('rho', 0.8)
        self.q = kwargs.get('Q', 1)
        self.t0 = kwargs.get('t0', .01)
        self.Generations = kwargs.get('Generations', 10)
        self.ant_count = kwargs.get('ant_count', 10)
        self.elite = kwargs.get('elite', .5)
        
    def create_colony(self, world):
        
        if self.ant_count < 1:
            return self.round_robin_ants(world, len(world.nodes))
        return self.random_ants(world, self.ant_count)
        
    def reset_colony(self, colony):
        
        for ant in colony:
            ant.initialize(ant.world)
        
    def aco(self, colony):
       
        self.find_solutions(colony)
        self.global_update(colony)
        return sorted(colony, reverse=True)[0]
        
    def solve(self, world):
        
        world.reset_pheromone(self.t0)
        global_best = None
        colony = self.create_colony(world)
        for i in range(self.Generations):
            self.reset_colony(colony)
            local_best = self.aco(colony)
            if global_best is None or local_best > global_best:
                global_best = copy(local_best)
            self.trace_elite(global_best)
        return global_best
    
    def solutions(self, world):
        
        world.reset_pheromone(self.t0)
        global_best = None
        colony = self.create_colony(world)
        for i in range(self.Generations):
            self.reset_colony(colony)
            local_best = self.aco(colony)
            if global_best is None or local_best > global_best:
                global_best = copy(local_best)
                yield global_best
            self.trace_elite(global_best)
    
    def round_robin_ants(self, world, count):
        
        starts = world.nodes
        n = len(starts)
        return [Ant(self.alpha, self.beta).initialize(world, start=starts[i % n]) for i in range(count)]
        
    def random_ants(self, world, count, even=False):
        
        ants = []
        starts = world.nodes
        n = len(starts)
        if even:
            # Since the caller wants an even distribution, use a round-robin 
            # method until the number of ants left to create is less than the
            # number of nodes.
            if count > n:
                for i in range(self.ant_count // n):
                    ants.extend([Ant(self.alpha,self.beta).initialize(world, start=starts[j]) for j in range(n)])
            # Now (without choosing the same node twice) choose the reamining
            # starts randomly.
            ants.extend([Ant(self.alpha, self.beta).initialize(world, start=starts.pop(random.randrange(n - i))) for i in range(count % n)])
        else:
            # Just pick random nodes.
            ants.extend([Ant(self.alpha, self.beta).initialize(world, start=starts[random.randrange(n)]) for i in range(count)])
        return ants

    def find_solutions(self, ants):
        
        # This loop occurs exactly as many times as there are ants times nodes,
        # but that is only because every ant must visit every node. It may be
        # more efficient to convert it to a counting loop
        ants_done = 0
        while ants_done < len(ants):
            ants_done = 0
            for ant in ants:
                if ant.can_move():
                    edge = ant.move()
                    self.local_update(edge)
                else:
                    ants_done += 1

    def local_update(self, edge):
        
        edge.pheromone = max(self.t0, edge.pheromone * self.rho)

    def global_update(self, ants):
        
        ants = sorted(ants)[:len(ants) // 2]
        for a in ants:
            p = self.q / a.distance
            for edge in a.path:
                edge.pheromone = max(self.t0, (1 - self.rho) * edge.pheromone + p)

    def trace_elite(self, ant):
        
        if self.elite:
            p = self.elite * self.q / ant.distance
            for edge in ant.path:
                edge.pheromone += p

#-------------------------------------------------------------------------------------------------------------