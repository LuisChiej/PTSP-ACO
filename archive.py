import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import List

from interfaces import Solution

@dataclass
class Archive:
    # Maximum size of solutions
    max_size: int = 1000

    # Dominated solutions
    objective_vectors: List[Solution] = field(default_factory=list)

    # Non dominated solutions
    non_dominated: List[Solution] = field(default_factory=list)

    def dominates(self, u: Solution, v: Solution) -> bool:
        return u.distance <= v.distance and u.demand >= v.demand and (u.distance < v.distance or u.demand > v.demand)

    # TODO - Remove filter the non_dominated from the objective-vectors and plot
    def update(self, y: List[Solution]) -> None:
        combined = self.objective_vectors + y

        pareto_front = []
        for s in combined:
            dominated = False
            for other in combined:
                if self.dominates(other, s):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(s)

        self.non_dominated = pareto_front

        self.objective_vectors = combined

    def plot(self):
        if not self.non_dominated:
            print("Archive is empty, nothing to plot.")
            return
        
        distances = [s.distance for s in self.non_dominated]
        demands = [s.demand for s in self.non_dominated]

        pareto_distances = [sol.distance for sol in self.objective_vectors]
        pareto_priorities = [sol.demand for sol in self.objective_vectors]
        
        plt.figure(figsize=(8, 6))

        plt.scatter(distances, demands, c='blue', label='Pareto Front')

        plt.scatter(pareto_distances, pareto_priorities, color='lightgray', label='Dominated Solutions', s=30)


        plt.xlabel('Total Distance (to minimize)')
        plt.ylabel('Total Demand (to maximize)')
        plt.title('Pareto Front of Solutions')
        plt.grid(True)
        plt.legend()
        plt.show()
