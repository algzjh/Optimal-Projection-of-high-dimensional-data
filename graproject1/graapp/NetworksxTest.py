from itertools import combinations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def k_cliques(graph):
    # 2-cliques
    cliques = [{i, j} for i, j in graph.edges() if i != j]
    k = 2

    while cliques:
        # result
        yield k, cliques
        # merge k-cliques into (k+1)-cliques
        cliques_1 = set()
        for u, v in combinations(cliques, 2):
            w = u ^ v  # Symmetric difference
            if len(w) == 2 and graph.has_edge(*w):
                # print("u: \n", u)
                # print("v: \n", v)
                # print("w: \n", *w)
                cliques_1.add(tuple(u | w))
        # remove duplicates
        cliques = list(map(set, cliques_1))
        k += 1


def print_cliques(graph):
    result_dict = {}
    for k, cliques in k_cliques(graph):
        print('%d-cliques: #%d, %s ' % (k, len(cliques), cliques[:]))
        result_dict[k] = cliques[:]
    return result_dict


def getMinSet(my_dict):
    result_list = []
    key_list = list(my_dict.keys())
    key_list.sort(reverse=True)
    print("key_list: \n", key_list)
    k_num = key_list[0]
    k_list = my_dict[k_num]
    for it in k_list:
        result_list.append(it)
    k_total = len(key_list)
    for i in range(1, k_total):
        k_num = key_list[i]
        k_list = my_dict[k_num]
        tmp_list = []
        for it1 in k_list:
            issubset = False
            for it2 in result_list:
                if it1.issubset(it2):
                    issubset = True
                    break
            if not issubset:
                tmp_list.append(it1)
        result_list = result_list + tmp_list
    return result_list

"""
if __name__ == "__main__":
    # my_dict = {0: (0, 1), 1: (0, 2), 2: (0, 3), 3: (0, 4), 4: (1, 2),
    #            5: (1, 3), 6: (1, 4), 7: (2, 3), 8: (2, 4), 9: (3, 4)}
    nodes = 10
    edges = 30
    graph = nx.Graph()
    # my_add_nodes = [i for i in range(10)]
    my_add_nodes = range(nodes)
    # print("my_add_nodes: \n", my_add_nodes)
    # my_add_nodes2 = [my_dict[it] for it in my_add_nodes]
    # print("my_add_nodes2: \n", my_add_nodes2)
    graph.add_nodes_from(my_add_nodes)
    my_add_edges = np.random.randint(0, nodes, (edges, 2))
    # print("my_add_edges: \n", my_add_edges)
    #     # my_add_edges2 = []
    #     # for it1 in my_add_edges:
    #     #     one_edge = []
    #     #     for it2 in it1:
    #     #         one_edge.append(my_dict[it2])
    #     #     my_add_edges2.append(one_edge)
    #     # print("my_add_edges2: \n", my_add_edges2)
    graph.add_edges_from(my_add_edges)
    # plt.figure(figsize=(12, 12))
    nx.draw_networkx(graph)
    # plt.show()
    my_dict = print_cliques(graph)
    print("my_dict: \n", my_dict)
    key_list = my_dict.keys()
    print("key_list: \n", key_list)
"""

"""
my_add_nodes = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
                    (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
"""
