from typing import List
from Edge import Edge
from Node import *
from Route import Route
import numpy as np
from geom import Geometric
from Distribution import Distribution
from stochastic import LogNormalDistribution
from stochastic import PoissonDistribution
from copy import deepcopy
from datetime import datetime

class VRP:

    def __init__(self, nodes, depots, vehicle_capacity, Algorithm: Geometric):
        self.nodes = nodes
        self.depots = depots
        self.routes = []
        self.vehicle_capacity = vehicle_capacity
        self.Geom = Algorithm
        self.best_mean = 0

    def input_depot(self):
        for i in self.nodes:
            cost = []
            for j in self.depots:
                edgedi = Edge(i, j)
                cost.append(edgedi)
                sorted(cost, key=lambda e: e.cost)
            i.depot = self.select_depot(cost)

    def select_depot(self, cost: []):
        edge_s = cost.pop()
        if self.depots.__contains__(edge_s.orig):
            return edge_s.orig
        return edge_s.dest

    def Dummy_sol(self):
        for i in self.nodes:
            route = Route()
            route.edges.append(Edge(i.depot, i))
            route.edges.append(Edge(i, i.depot))
            route.demand = i.demand
            route.cost = 2 * Edge(i.depot, i).cost
            i.Route_in = route
            self.routes.append(route)

    def Saving_List(self, Edge_distribution: Distribution = LogNormalDistribution) -> (List[Edge], List[Edge]):

        def mean_val(edge) -> float:
            return Edge_distribution(np.log(edge.cost), edge.var_calc()).mean_value()

        savings = []
        for i in range(len(self.nodes)):
            node_a = self.nodes[i]
            for j in range(i + 1, len(self.nodes)):
                node_b = self.nodes[j]
                Edgeij = Edge(node_a, node_b)
                Edgedi = Edge(node_a, node_a.depot)
                Edgedj = Edge(node_b, node_b.depot)
                Edgeij.saving = Edgedi.cost + Edgedj.cost - Edgeij.cost
                Edgeij.stochastic_saving = mean_val(Edgedi) + mean_val(Edgedj) - mean_val(Edgeij)
                savings.append(Edgeij)

        return sorted(savings, key=lambda e: e.saving), sorted(savings, key=lambda e: e.stochastic_saving)

    def Solution(self, savings):
        savings = self.Geom.reorganize_savings_list(savings)
        while (len(savings) > 1):
            edge_s = savings.pop(0)
            if self.merge(edge_s):
                Route_a = edge_s.orig.Route_in
                Route_b = edge_s.dest.Route_in

                if Route_a.edges[0].dest == edge_s.orig:
                    Route_a.reverse()

                if Route_b.edges[-1].orig == edge_s.dest:
                    Route_b.reverse()

                if len(Route_b.edges) > 2:
                    edge_s.dest.is_interior = True

                if len(Route_a.edges) > 2:
                    edge_s.orig.is_interior = True

                Route_a.edges.pop()
                Route_a.edges.append(edge_s)
                Route_a.cost += edge_s.cost
                edge_s.dest.Route_in = Route_a
                Route_a.demand += Route_b.demand
                while len(Route_b.edges) > 1:
                    edge_out = Route_b.edges.pop(1)
                    edge_out.dest.Route_in = Route_a
                    Route_a.edges.append(edge_out)
                    Route_a.cost += edge_out.cost

                self.routes.remove(Route_b)

    def merge(self, edge: Edge):
        if (edge.orig.Route_in != edge.dest.Route_in and not edge.orig.is_interior and not
        edge.dest.is_interior):
            return edge.orig.Route_in.demand + edge.dest.Route_in.demand <= self.vehicle_capacity * (
                        1 - self.Geom.safety_stock)
        return False

    def swap_nodes_deterministic_intra(self, route, i, j, together) -> (float, float):
        node_i = route.edges[i].orig
        node_i_prev = route.edges[i - 1].orig
        node_i_next = route.edges[i].dest

        node_j = route.edges[j].orig
        node_j_prev = route.edges[j - 1].orig
        node_j_next = route.edges[j].dest

        old_cost = route.edges[j].cost + route.edges[j - 1].cost + \
                   route.edges[i].cost + route.edges[i - 1].cost

        if together:
            new_cost = self.norm_2(node_j, node_i_prev) + 2 * self.norm_2(node_j, node_i) + \
                       self.norm_2(node_i, node_j_next)
        else:
            new_cost = self.norm_2(node_j, node_i_prev) + self.norm_2(node_j, node_i_next) + \
                       self.norm_2(node_i, node_j_prev) + self.norm_2(node_i, node_j_next)

        return old_cost, new_cost, new_cost - old_cost

    def swap_nodes_deterministic_between(self, route_t, route_k, i, j) -> (float, float):
        node_i = route_k.edges[i].orig
        node_i_prev = route_k.edges[i - 1].orig
        node_i_next = route_k.edges[i].dest

        node_j = route_t.edges[j].orig
        node_j_prev = route_t.edges[j - 1].orig
        node_j_next = route_t.edges[j].dest

        old_cost_t = route_t.edges[j].cost + route_t.edges[j - 1].cost
        old_cost_k = route_k.edges[i].cost + route_k.edges[i - 1].cost

        new_cost_k = self.norm_2(node_j, node_i_prev) + self.norm_2(node_j, node_i_next)
        new_cost_t = self.norm_2(node_i, node_j_prev) + self.norm_2(node_i, node_j_next)

        return old_cost_t, old_cost_k, new_cost_t, new_cost_k, new_cost_k - old_cost_k, new_cost_t - old_cost_t

    def swap_nodes_stochastic_intra(self, route, i, j, together, dist: Distribution = LogNormalDistribution) -> (
    float, float):

        def mean_val(edge) -> float:
            return dist(edge.cost, edge.var_calc()).mean_value()

        node_i = route.edges[i].orig
        node_i_prev = route.edges[i - 1].orig
        node_i_next = route.edges[i].dest

        node_j = route.edges[j].orig
        node_j_prev = route.edges[j - 1].orig
        node_j_next = route.edges[j].dest

        determinist_old_cost = route.edges[j].cost + route.edges[j - 1].cost + \
                               route.edges[i].cost + route.edges[i - 1].cost

        old_cost = mean_val(route.edges[j]) + mean_val(route.edges[j - 1]) + mean_val(route.edges[i]) + mean_val(
            route.edges[i - 1])

        if together:
            new_cost = mean_val(Edge(node_j, node_i_prev)) + 2 * mean_val(Edge(node_j, node_i)) + mean_val(
                Edge(node_i, node_j_next))

            determinist_new_cost = self.norm_2(node_j, node_i_prev) + 2 * self.norm_2(node_j, node_i) + \
                                   self.norm_2(node_i, node_j_next)
        else:
            new_cost = mean_val(Edge(node_j, node_i_prev)) + mean_val(Edge(node_j, node_i_next)) + \
                       mean_val(Edge(node_i, node_j_prev)) + mean_val(Edge(node_i, node_j_next))

            determinist_new_cost = self.norm_2(node_j, node_i_prev) + self.norm_2(node_j, node_i_next) + \
                                   self.norm_2(node_i, node_j_prev) + self.norm_2(node_i, node_j_next)

        return old_cost, new_cost, determinist_new_cost - determinist_old_cost

    def swap_nodes_stochastic_between(self, route_t, route_k, i, j, dist: Distribution = LogNormalDistribution) -> (
    float, float):

        def mean_val(edge) -> float:
            return dist(edge.cost, edge.var_calc()).mean_value()

        node_i = route_k.edges[i].orig
        node_i_prev = route_k.edges[i - 1].orig
        node_i_next = route_k.edges[i].dest

        node_j = route_t.edges[j].orig
        node_j_prev = route_t.edges[j - 1].orig
        node_j_next = route_t.edges[j].dest

        # Stochastic
        old_cost_t = mean_val(route_t.edges[j]) + mean_val(route_t.edges[j - 1])
        old_cost_k = mean_val(route_k.edges[i]) + mean_val(route_k.edges[i - 1])

        new_cost_k = mean_val(Edge(node_j, node_i_prev)) + mean_val(Edge(node_j, node_i_next))
        new_cost_t = mean_val(Edge(node_i, node_j_prev)) + mean_val(Edge(node_i, node_j_next))

        # Deterministic
        Deterministic_old_cost_t = route_t.edges[j].cost + route_t.edges[j - 1].cost
        Deterministic_old_cost_k = route_k.edges[i].cost + route_k.edges[i - 1].cost

        Deterministic_new_cost_k = self.norm_2(node_j, node_i_prev) + self.norm_2(node_j, node_i_next)
        Deterministic_new_cost_t = self.norm_2(node_i, node_j_prev) + self.norm_2(node_i, node_j_next)

        return old_cost_t, old_cost_k, new_cost_t, new_cost_k, Deterministic_new_cost_k - Deterministic_old_cost_k, Deterministic_new_cost_t - Deterministic_old_cost_t

    def local_search_intra(self, distance):
        global_improve = False
        for route in self.routes:
            improve = True
            while improve:
                improve = False
                for i in range(1, len(route.edges) - 1):

                    for j in range(i + 1, len(route.edges) - 1):
                        node_i = route.edges[i].orig
                        node_i_prev = route.edges[i - 1].orig
                        node_i_next = route.edges[i].dest

                        node_j = route.edges[j].orig
                        node_j_prev = route.edges[j - 1].orig
                        node_j_next = route.edges[j].dest

                        if node_i_next != node_j:
                            old_cost, new_cost, cost_improve = distance(route, i, j, False)

                            if new_cost < old_cost:
                                improve = True
                                global_improve = True
                                route.edges[i - 1] = Edge(node_i_prev, node_j)
                                route.edges[i] = Edge(node_j, node_i_next)

                                route.edges[j - 1] = Edge(node_j_prev, node_i)
                                route.edges[j] = Edge(node_i, node_j_next)

                                route.cost += cost_improve
                        else:
                            old_cost, new_cost, cost_improve = distance(route, i, j, True)

                            if new_cost < old_cost:
                                improve = True
                                global_improve = True
                                route.edges[i - 1] = Edge(node_i_prev, node_j)
                                route.edges[i] = Edge(node_j, node_i)
                                route.edges[j] = Edge(node_i, node_j_next)

                                route.cost += cost_improve
        return global_improve

    def local_search_between(self, distance):
        global_improve = False
        for k in range(len(self.routes)):
            route_k = self.routes[k]
            for t in range(k + 1, len(self.routes)):
                route_t = self.routes[t]
                improve = True
                while improve:
                    improve = False
                    for i in range(1, len(route_k.edges) - 1):
                        for j in range(1, len(route_t.edges) - 1):

                            node_i = route_k.edges[i].orig
                            node_i_prev = route_k.edges[i - 1].orig
                            node_i_next = route_k.edges[i].dest

                            node_j = route_t.edges[j].orig
                            node_j_prev = route_t.edges[j - 1].orig
                            node_j_next = route_t.edges[j].dest

                            if (node_i.Route_in.demand + node_j.demand - node_i.demand <= self.vehicle_capacity
                                    and node_j.Route_in.demand + node_i.demand - node_j.demand <= self.vehicle_capacity
                                    and node_j.Route_in != node_i.Route_in):

                                old_cost_t, old_cost_k, new_cost_t, new_cost_k, total_cost_k, total_cost_t = distance(
                                    route_t, route_k, i, j)

                                if new_cost_k + new_cost_t < old_cost_t + old_cost_k:
                                    improve = True
                                    global_improve = True
                                    node_j.Route_in, node_i.Route_in = node_i.Route_in, node_j.Route_in

                                    route_k.edges[i - 1] = Edge(node_i_prev, node_j)
                                    route_k.edges[i] = Edge(node_j, node_i_next)

                                    route_t.edges[j - 1] = Edge(node_j_prev, node_i)
                                    route_t.edges[j] = Edge(node_i, node_j_next)

                                    route_k.cost += total_cost_k
                                    route_t.cost += total_cost_t
        return global_improve

    def norm_2(self, a: Node, b: Node):
        return np.power(np.power(a.x - b.x, 2) + np.power(a.y - b.y, 2), 1 / 2)

    def simulation(self, times=500, dist_edge_cost: Distribution = LogNormalDistribution,
                   dist_node_demand: Distribution = PoissonDistribution):
        data = []
        for _ in range(times):
            for route in self.routes:
                stochastic_capacity = 0
                for edge in route.edges:
                    stochastic_demand = dist_node_demand(edge.dest.insert_lam()).random_value()
                    if stochastic_demand > 10: stochastic_demand = 10
                    if stochastic_capacity + stochastic_demand <= self.vehicle_capacity:
                        stochastic_capacity += stochastic_demand
                        stochastic_edge_cost = dist_edge_cost(edge.cost, edge.var_calc()).random_value()

                        route.stochastic_cost += stochastic_edge_cost
                    else:
                        return_to_depot = Edge(edge.orig, edge.orig.depot)
                        return_to_customer = Edge(edge.orig.depot, edge.dest)

                        stochastic_edge_return = dist_edge_cost(return_to_depot.cost,
                                                                return_to_depot.var_calc()).random_value()

                        stochastic_edge_customer = dist_edge_cost(return_to_customer.cost,
                                                                  return_to_customer.var_calc()).random_value()

                        stochastic_capacity = stochastic_demand
                        route.stochastic_cost += stochastic_edge_return + stochastic_edge_customer
                    data.append(route.stochastic_cost)
        return data

    def deterministic_solve(self, time):

        # Determinista
        t0 = datetime.utcnow().timestamp()

        VRP_copy = deepcopy(self)
        i = 0
        best = -1
        while datetime.utcnow().timestamp() - t0 < time:
            self = deepcopy(VRP_copy)
            self.input_depot()
            self.Dummy_sol()
            self.Solution(self.Saving_List()[0])

            improve_1 = True
            improve_2 = True
            while improve_1 or improve_2:
                improve_1 = self.local_search_intra(self.swap_nodes_deterministic_intra)
                improve_2 = self.local_search_between(self.swap_nodes_deterministic_between)
            i += 1
            val_simulation = np.mean(self.simulation())
            if best == -1 or best > val_simulation:
                best = val_simulation
                self.best_mean = best
                best_vrp = deepcopy(self)
                #print(val_simulation)

        print("El valor de la simulación en la metodología determinista es "+str(best))
        return best_vrp

    def stochastic_solve(self, time):
        # Stochastic
        t0 = datetime.utcnow().timestamp()

        VRP_copy = deepcopy(self)
        i = 0
        best = -1
        while datetime.utcnow().timestamp() - t0 < time:
            self = deepcopy(VRP_copy)
            self.input_depot()
            self.Dummy_sol()
            self.Solution(self.Saving_List()[1])

            improve_1 = True
            improve_2 = True
            while improve_1 or improve_2:
                improve_1 = self.local_search_intra(self.swap_nodes_stochastic_intra)
                improve_2 = self.local_search_between(self.swap_nodes_stochastic_between)

            i += 1
            val_simulation = np.mean(self.simulation())
            if best == -1 or best > val_simulation:
                best = val_simulation
                self.best_mean = best
                best_vrp = deepcopy(self)
                #print(val_simulation)

        print("El valor de la simulación en la metodología estocástica es "+str(best))
        return best_vrp

    def stochastic_local_search_solve(self, time):

        # Determinista
        t0 = datetime.utcnow().timestamp()

        VRP_copy = deepcopy(self)
        i = 0
        best = -1
        while datetime.utcnow().timestamp() - t0 < time:
            self = deepcopy(VRP_copy)
            self.input_depot()
            self.Dummy_sol()
            self.Solution(self.Saving_List()[0])

            improve_1 = True
            improve_2 = True
            while improve_1 or improve_2:
                improve_1 = self.local_search_intra(self.swap_nodes_stochastic_intra)
                improve_2 = self.local_search_between(self.swap_nodes_stochastic_between)
            i += 1
            val_simulation = np.mean(self.simulation())
            if best == -1 or best > val_simulation:
                best = val_simulation
                self.best_mean = best
                best_vrp = deepcopy(self)
                # print(val_simulation)

        print("El valor de la simulación en la metodología local search estocástico es " + str(best))
        return best_vrp

    def stochastic_savings_solve(self, time):

        # Determinista
        t0 = datetime.utcnow().timestamp()

        VRP_copy = deepcopy(self)
        i = 0
        best = -1
        while datetime.utcnow().timestamp() - t0 < time:
            self = deepcopy(VRP_copy)
            self.input_depot()
            self.Dummy_sol()
            self.Solution(self.Saving_List()[1])

            improve_1 = True
            improve_2 = True
            while improve_1 or improve_2:
                improve_1 = self.local_search_intra(self.swap_nodes_deterministic_intra)
                improve_2 = self.local_search_between(self.swap_nodes_deterministic_between)
            i += 1
            val_simulation = np.mean(self.simulation())
            if best == -1 or best > val_simulation:
                best = val_simulation
                self.best_mean = best
                best_vrp = deepcopy(self)
                #print(val_simulation)

        print("El valor de la simulación en la metodología savings estocásticos es "+str(best))
        return best_vrp

    def show(self):
        cost = 0
        for i in self.routes:
            route_str = i.edges[0].orig.id
            for j in i.edges:
                route_str += "-" + j.dest.id
            print(route_str)
            print(i.cost)
            print("")
            cost += i.cost

        print("Global cost: ")
        print(cost)
        print("")
        print("")
