from Route import Route
import random as rand
class Node:

    def __init__(self, x, y, demand):
        self.x = x
        self.y = y
        self.demand = demand
        self.depot = None
        self.Route_in = Route()
        self.is_interior = False
        self.id = "0"

    def insert_lam(self):
        if self.id[0] == "d":
            lam = 0
        else:
            rand.seed(self.id)
            lam = self.demand
        return lam

    def insert_id(self, id):
        self.id = id

    def __str__(self):
        return self.id + " " + str(self.x) + " " +str(self.y)