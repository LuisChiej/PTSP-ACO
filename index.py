from typing import List

from interfaces import Customer
from moga import AntColonyOptimizer

def main():
    vpr = "vpr.txt"
    data = open(vpr, "r").read()

    customers: List[Customer] = []
    for line in data.strip().split('\n'):
        parts = list(map(int, line.split()))
        customers.append(
            Customer(id=parts[0], x=parts[1], y=parts[2], demand=parts[3]))

    iteration = 50
    aco = AntColonyOptimizer(
        customers=customers, number_of_ants=30, visualize=True)

    # Optimize
    archive = aco.optimize(niter=iteration)

    # Plot archive
    archive.plot()

if __name__ == "__main__":
    main()