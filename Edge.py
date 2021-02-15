import Node
import numpy as np
import random

class Edge:

    def __init__(self, a: Node, b: Node):
        self.cost = self.distance(a, b)
        self.saving = 0
        self.stochastic_saving = 0
        self.orig = a
        self.dest = b

    def distance(self, a: Node, b: Node) -> float:
        return np.power(np.power(a.x-b.x, 2) + np.power(a.y-b.y, 2), 1/2)

    def var_calc(self):
        if self.orig.id[0] == "d":
            random.seed(int(self.orig.id[1:])+int(self.dest.id))
        elif self.dest.id[0] == "d":
            random.seed(int(self.dest.id[1:])+int(self.orig.id))
        else:
            random.seed(int(self.dest.id) + int(self.orig.id))

        var = random.random()
        return var

    def __str__(self):
        return self.orig.id + "-" + self.dest.id