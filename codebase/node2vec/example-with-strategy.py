import networkx as nx
from node2vec import Node2Vec

# FILES
EMBEDDING_FILENAME = './embeddings.emb'
EMBEDDING_MODEL_FILENAME = './embeddings.model'


# Create a graph
graph = nx.fast_gnp_random_graph(n=100, p=0.5)

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

# Precompute probabilities and generate walks
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4,sampling_strategy= sampling_strategy_graph)

## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
# Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
#node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")

# Embed
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

# Look for most similar nodes
model.wv.most_similar('2')  # Output node names are always strings

# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
model.save(EMBEDDING_MODEL_FILENAME)
