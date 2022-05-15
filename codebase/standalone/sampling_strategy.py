import networkx as nx

def sampling_strategy(graph):
    ### Computing degree centrality 1 of the graph
    d_1_centrality = nx.degree_centrality(graph)

    ### Computing degree centrality 2 of the graph
    nodes_generator = graph.nodes()
    d_2_centrality = dict()
    for node_1 in nodes_generator:
        d2 = 0
        n_neighbors = 0
        for node_2 in graph.neighbors(node_1):
            d2 += d_1_centrality[node_2]
            n_neighbors +=1
        if n_neighbors>0:
            d_2_centrality[node_1] = d2/n_neighbors
        else:
            d_2_centrality[node_1] = n_neighbors

    d1_max = max(d_1_centrality.values())
    d2_max = max(d_2_centrality.values())

    ### Design of the sampling strategy
    sampling_strategy_graph = dict()
    for node in nodes_generator:
        dict_node = dict()
        dict_node["p"] = d1_max/d_1_centrality[node]
        dict_node["q"] = d2_max/d_2_centrality[node]
        sampling_strategy_graph[node] = dict_node

    return sampling_strategy_graph