from math import log
from random import random
from typing import List
import numpy as np
from Edge import Edge



class Geometric():
    def __init__(self, beta: float, safety_stock: float):
        self.beta = beta
        self.safety_stock = safety_stock

    def get_position(self, n: int) -> int:
        #np.random
        return int((log(np.random.random()) // log(1 - self.beta)) % n)

    def reorganize_savings_list(self, savings_list: List[Edge]) -> List[Edge]:
        monte_carlo = []
        while savings_list:
            n = len(savings_list)
            position = self.get_position(len(savings_list))
            monte_carlo.insert(0, savings_list.pop(n - position - 1))

        return monte_carlo

    def name(self):
        return f'Geometric_beta_{self.beta}'

    def __str__(self):
        return f'{self.beta};{self.safety_stock}'

    def __repr__(self):
        return f'{self.beta};{self.safety_stock}'