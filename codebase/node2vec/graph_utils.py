import networkx as nx
import numpy as np


def read_facebook_data(file_location):
    f = open(file_location, "r")
    edges = f.readlines()
    G = nx.parse_edgelist(edges, nodetype=int)

    return G


def delete_random_edge(G, random = True, edge_to_remove = None):
    if not random and edge_to_remove is None:
        raise RuntimeError("Invalid input combination - if not random then need to specify an edge to remove")

    edges = list(G.edges)

    if random:
        edge_to_remove = np.random.choice(edges)

    G.remove_edge(edge_to_remove[0], edge_to_remove[1])

    return G