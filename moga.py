import numpy as np
import random as rnd

from dataclasses import dataclass
from typing import List, Tuple
from numpy.typing import NDArray

from archive import Archive
from interfaces import Customer, Solution

@dataclass
class MOGAOptimizer:
    # Customers
    customers: List[Customer]

    # Population size 
    population_size: int = 50

    # Number of generations
    generations: int = 100

    # Crossover rate
    crossover_rate: float = 0.8

    # Mutation rate
    mutation_rate: float = 0.2

    def __post_init__(self):
        # Add starting point indexed 0 to customer list
        self.customers = [Customer(id=0, demand=0, x=0.0, y=0.0)] + self.customers

        self.customer_ids = [customer.id for customer in self.customers]

        self.number_of_customers = len(self.customers)

        self.customers_dictionary = {customer.id: customer for customer in self.customers}

        self.distances = self.compute_distance()

    def compute_distance(self) -> NDArray[np.float64]:
        distances = np.zeros((self.number_of_customers, self.number_of_customers))
        for a in range(self.number_of_customers):
            for b in range(self.number_of_customers):
                if a != b:
                    ac = self.customers[a]
                    bc = self.customers[b]
                    distances[a][b] = np.linalg.norm(ac.location - bc.location)
                else:
                    distances[a][b] = np.inf
        return distances

    def initialize_population(self) -> List[Solution]:
        population = []
        starting_point = self.customer_ids[:1]

        for _ in range(self.population_size):
            ids = self.customer_ids[1:].copy() # Remove starting point indexed at 0 
            rnd.shuffle(ids)
            path = starting_point + ids + starting_point # All paths have to start and end at the starting point
            distance, demand_score = self.evaluate(path)
            population.append(Solution(path=path, distance=distance, demand=demand_score))
        return population

    def evaluate(self, path: List[int]) -> Tuple[float, float]:
        total_distance = sum(self.distances[path[i]][path[i+1]] for i in range(len(path) - 1))
        N = len(path) - 1
        demand_score = 0.0
        for idx, customer_id in enumerate(path):
            weight = 1 - (idx / N)
            demand_score += self.customers_dictionary[customer_id].demand * weight
        return total_distance, demand_score

    def select_parents(self, population: List[Solution]) -> Tuple[Solution, Solution]:
        return (self.select(population), self.select(population))

    def select(self, population: List[Solution]) -> Solution:
        # Using tournament selection
        subset = rnd.sample(population, k=3) # Select a subset of 3 individuals
        return min(subset, key=lambda sol: sol.distance)

    def crossover(self, p1: Solution, p2: Solution) -> Tuple[Solution, Solution]:
        if rnd.random() > self.crossover_rate:
            return p1, p2
        
        starting_point = p1.path[:1]
        length = len(p1.path) - 1

        p1_path = p1.path[1:length]
        p2_path = p2.path[1:length]
        
        start, end = sorted(rnd.sample(range(len(p1_path)), 2))

        def order_crossover(path1: List[int], path2: List[int]):
            child = [None] * len(path1)
            child[start:end+1] = path1[start:end+1]

            path2_idx = 0
            for i in range(len(child)):
                if child[i] is None:
                    while path2[path2_idx] in child:
                        path2_idx += 1
                    child[i] = path2[path2_idx]
            return child

        c1_path = order_crossover(p1_path, p2_path)
        c2_path = order_crossover(p2_path, p1_path)

        child1 = Solution(path=starting_point + c1_path + starting_point, distance=0.0, demand=0.0)
        child2 = Solution(path=starting_point + c2_path + starting_point, distance=0.0, demand=0.0)
        child1.distance, child1.demand = self.evaluate(child1.path)
        child2.distance, child2.demand = self.evaluate(child2.path)
        return child1, child2

    def mutate(self, solution: Solution) -> Solution:
        if rnd.random() > self.mutation_rate:
            return solution

        starting_point = solution.path[:1]
        length = len(solution.path) - 1

        path = solution.path[1:length].copy()
        idx1, idx2 = rnd.sample(range(len(path)), 2)
        path[idx1], path[idx2] = path[idx2], path[idx1]

        mutated = Solution(path=starting_point + path + starting_point, distance=0.0, demand=0.0)
        mutated.distance, mutated.demand = self.evaluate(mutated.path)
        return mutated

    def optimize(self) -> Archive:
        archive = Archive()

        population = self.initialize_population()

        for _ in range(self.generations):
            next_generation = []

            while len(next_generation) < self.population_size:
                parent1, parent2 = self.select_parents(population)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                next_generation.append(child1)
                next_generation.append(child2)

            population = next_generation[:self.population_size]
            archive.update(population)
        return archive