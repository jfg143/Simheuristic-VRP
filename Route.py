from Edge import Edge

class Route:

    def __init__(self):
        self.cost = 0
        self.edges = []
        self.demand = 0
        self.stochastic_cost = 0

    def reverse(self):
        reverse_edges = []
        for e in self.edges:
            reverse_edges.insert(0, Edge(e.dest, e.orig))
            reverse_edges[0].saving = e.saving
        self.edges = reverse_edges

    def __str__(self):
        route_str = self.edges[0].orig.id
        for j in self.edges:
            route_str += "-" + j.dest.id

        return route_str
