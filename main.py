import csv
import pickle
from collections import defaultdict

import numpy as np
from Node import Node
from VRP import VRP
from geom import Geometric
from scipy import stats

def t_test():
    a_file = open("Datos diccionario 2.pkl", "rb")
    output = pickle.load(a_file)
    N = len(output["Determinista"])
    for key in output.keys():
        print(np.mean(output[key]))

    determinista = output["Determinista"]
    stocastico = output["Savings estocástico"]
    var_a = np.var(determinista)
    var_b = np.var(stocastico)
    s = np.sqrt((var_a + var_b) / 2)
    t = (np.mean(determinista) - np.mean(stocastico)) / (s * np.sqrt(2 / N))
    df = 2 * N - 2
    print("p value del t-test: "+ str(1-stats.t.cdf(t, df = df)))

def start(number_nodes, max_demand):
    list = []
    depot = []
    depot.append(Node(5, 5, 0))
    depot[0].insert_id("d0")
    k = 0
    for i in range(number_nodes):
            node = Node(np.random.random()*20, np.random.random()*20, np.random.randint(1, max_demand))
            node.insert_id(str(k))
            list.append(node)
            k += 1
    return list, depot

if __name__ == '__main__':
    Geom = Geometric(0.2, 0.3)
    time = 1
    iteraciones = 10


    resultados = {"Determinista": [], "Estocástico": [], "Savings estocástico": [], "Local Search estocástico": []}
    i = 0
    while i < iteraciones:
        list_original, depot_original = start(20, 5)

        list, depot = list_original, depot_original
        Sol1 = VRP(list, depot, 10, Geom)
        Sol1 = Sol1.deterministic_solve(time)
        resultados["Determinista"].append(Sol1.best_mean)
        #Sol1.show()
        del Sol1


        list, depot = list_original, depot_original

        Sol2 = VRP(list, depot, 10, Geom)
        Sol2 = Sol2.stochastic_solve(time)
        resultados["Estocástico"].append(Sol2.best_mean)
        #Sol2.show()
        del Sol2

        list, depot = list_original, depot_original

        Sol3 = VRP(list, depot, 10, Geom)
        Sol3 = Sol3.stochastic_savings_solve(time)
        resultados["Savings estocástico"].append(Sol3.best_mean)
        #Sol3.show()
        del Sol3

        list, depot = list_original, depot_original

        Sol4 = VRP(list, depot, 10, Geom)
        Sol4 = Sol4.stochastic_local_search_solve(time)
        resultados["Local Search estocástico"].append(Sol4.best_mean)
        #Sol4.show()
        del Sol4
        print("\n")
        i += 1


    a_file = open("Datos diccionario.pkl", "wb")
    pickle.dump(resultados, a_file)
    a_file.close()

    '''
    a_file = open("Datos diccionario 2.pkl", "rb")
    output = pickle.load(a_file)
    print(output)
    '''

    #print(resultados)
    t_test()

