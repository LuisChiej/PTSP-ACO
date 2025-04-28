from enum import Enum, unique
import numpy as np
import random as rnd

from dataclasses import dataclass, field
from typing import List, Tuple
from numpy.typing import NDArray
from matplotlib import pyplot as plt, cm

from archive import Archive
from interfaces import Customer, Solution

@unique
class SelectionStrategy(Enum):
    TOURNAMENT = 1
    ROULETTE = 2

@dataclass
class  AntColonyOptimizer:
    # Customers
    customers: List[Customer] = field(default_factory=[])

    # Pheromone influence
    alpha: float = 1.0

    # Distance influence
    beta: float = 2.0

    # Pheromone evaporation rate (rho)
    evaporation_rate: float = 0.5

    # Deposit factor
    deposit_factor: int = 100

    # Selection strategy
    selection_strategy: SelectionStrategy = SelectionStrategy.ROULETTE

    # Number of ants
    number_of_ants: int = 30

    # Visualize path construction
    visualize: bool = False

    def __post_init__(self):

        # Add starting point
        customers = [Customer(id=0, demand=0, x=0.0, y=0.0)] + self.customers
        self.customers = customers

        # Number of customers
        self.number_of_customers = len(self.customers)

        # Make a dictionary of customers for better search
        self.customers_dictionary = {customer.id: customer for customer in self.customers}

        # Pheromones
        self.pheromones = np.ones((self.number_of_customers, self.number_of_customers))

        # Distance matrix representing distances between customers
        self.distances = self.compute_distance()

    def compute_distance(self) -> NDArray[np.float64]:
        distances = np.zeros((self.number_of_customers, self.number_of_customers))
                    
        for a in range(self.number_of_customers):
            for b in range(self.number_of_customers):
                if a != b:
                    a_customer: Customer = self.customers[a]
                    b_customer: Customer = self.customers[b]
                    distances[a][b] = np.linalg.norm(a_customer.location - b_customer.location)
                else:
                    distances[a][b] = np.inf
        return distances
    
    # Select a population using the selection strategy selected (default = roulette)
    def select(self, probabilities: List[Tuple[int, float]]) -> int:
        match self.selection_strategy:
            case SelectionStrategy.ROULETTE:
                r = rnd.random()
            
                cumulative = 0.0
                for id, probability in probabilities:
                    cumulative += probability
                    if r <= cumulative:
                        return id
                return probabilities[-1][0]
            case SelectionStrategy.TOURNAMENT:
                # TODO - Implement tournament selection strategy
                return 0
            
    # Evaluates the path and returns the total distance and total demand satisfied
    def evaluate(self, path: List[int], distances: NDArray[np.float64]) -> Tuple[float, float]:
        total_distance = sum(distances[path[i]][path[i+1]] for i in range(len(path) - 1))

        # Compute demand score based on weighted priority
        N = len(path) - 1
        demand_score = 0.0
        for idx, customer_id in enumerate(path):
            weight = 1 - (idx / N)
            demand_score += self.customers_dictionary[customer_id].demand * weight
        return total_distance, demand_score

    def construct_path(self, customers: List[int], distances: NDArray[np.float64]) -> List[int]:
        path = customers[:1]
        unvisited = set(customers[1:])

        while unvisited:
            current_node = path[-1]
            probabilities: List[Tuple[int, float]] = []

            # Compute denominator of all unvisited customers
            denominator: float = sum(
                (self.pheromones[current_node][i] ** self.alpha) *
                (1/distances[current_node][i]) ** self.beta
                for i in unvisited
            )

            # Calculate the probabilities of each unvisited customer
            for i in unvisited:
                pheromone = self.pheromones[current_node][i] ** self.alpha
                heuristic = (1/distances[current_node][i]) ** self.beta
                probabilities.append((i, (pheromone * heuristic) / denominator))

            # Using a selection strategy return the next probable customer
            next_node = self.select(probabilities=probabilities)
            path.append(next_node)
            unvisited.remove(next_node)
        return path

    def visualize_path(self, path: List[int], iteration: int, ant: int):
        plt.clf()
        plt.title(f"Iteration {iteration + 1} - {'Ant path construction starting' if ant == 0 else 'Ant path construction ending'}")

        coordinates = [(customer.x, customer.y) for customer in self.customers]
        x_coords = [x for x, _ in coordinates]
        y_coords = [y for _, y in coordinates]

        plt.scatter(x_coords, y_coords, color='black', s=10)
    
        for i in range(len(path) - 1):
            x0, y0 = coordinates[path[i]]
            x1, y1 = coordinates[path[i + 1]]
            strength = self.pheromones[path[i]][path[i + 1]] if self.pheromones is not None else 1
            plt.plot([x0, x1], [y0, y1], color=cm.tab20(ant % 20), linewidth=1 + 0.1 * strength)
            plt.pause(0.05)
        plt.pause(0.05)


    def update_pheromones(self, solutions: List[List[int]]) -> None:
        self.pheromones *= (1 - self.evaporation_rate)
        for path in solutions:
            for i in range(len(path) - 1):
                self.pheromones[path[i]][path[i + 1]] += 1.0

    def optimize(self, niter: int) -> Archive:
        # Initiate archive
        archive = Archive()

        plt.figure(figsize=(7, 7))

        for iter in range(niter):
            solutions: List[Solution] = [] # Contains (path, distance, demand_score)

            for ant in range(self.number_of_ants):
                path = self.construct_path(customers=[customer.id for customer in self.customers], distances=self.distances)
                distance, demand_score = self.evaluate(path=path, distances=self.distances)
                solutions.append(Solution(path=path, distance=distance, demand=demand_score))

                # Visualize path
                if(self.visualize):
                    if (iter == 0 or (iter + 1) % 10 == 0) and (ant == 0 or ant == self.number_of_ants - 1):
                        self.visualize_path(path, iter, ant)
            
            # Update pheromones based on path in solution
            self.update_pheromones([solution.path for solution in solutions])

            archive.update(solutions)
        return archive