import numpy as np

from dataclasses import dataclass, field
from typing import List

# Solution type
@dataclass
class Solution:
    # Order in which salesman visits cities
    path: List[int] = field(default_factory=list)

    # Distance covered
    distance: float = 0.0

    # Demand satisfied score
    demand: float = 0.0


# Customer type
@dataclass
class Customer:
    # Serial number/ID
    id: int

    # Priority of the customer
    demand: int

    # Customer longitude
    x: float

    # Customer latitude
    y: float

    def __post_init__(self):
        object.__setattr__(self, 'location', np.array([self.x, self.y]))

