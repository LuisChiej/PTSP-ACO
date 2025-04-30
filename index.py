from typing import List

from interfaces import Customer
from moga import MOGAOptimizer

def main():
    vrp = "vrp.txt"
    data = open(vrp, "r").read()

    customers: List[Customer] = []
    for line in data.strip().split('\n'):
        parts = list(map(int, line.split()))
        customers.append(
            Customer(id=parts[0], x=parts[1], y=parts[2], demand=parts[3]))

    moga = MOGAOptimizer(customers=customers)

    # Optimize
    archive = moga.optimize()

    # Plot
    archive.plot()

if __name__ == "__main__":
    main()