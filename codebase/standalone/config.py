from graph import data_selector, test_splitter, train_splitter
from node2vec import Node2Vec
from sampling_strategy import sampling_strategy
import multiprocessing
import numpy as np

### Graph and sampling strategy ###
# Define custom nx graph by hand or
# 1: FB, 2: PPI, 3: arXiv
graph = data_selector(2)

graph_test, examples_test, labels_test = test_splitter(graph)
graph_train, examples, labels, examples_train, examples_model_selection, labels_train, labels_model_selection = \
    train_splitter(graph, graph_test)

graph_nx = graph.to_networkx()
graph_test_nx = graph_test.to_networkx()
graph_train_nx = graph_train.to_networkx()

# Define whether to use sampling strategy
sampling_strategy_graph = sampling_strategy(graph_nx)

### Hyperparameters ###
p = 1.0
q = 1.0
dimensions = 128
num_walks = 10
walk_length = 80
window_size = 10
num_iter = 1
workers = multiprocessing.cpu_count()

node2vec_train = Node2Vec(graph_train_nx, p = p, q= q, dimensions=dimensions, walk_length=walk_length,
                           num_walks=num_walks, workers=workers, sampling_strategy= sampling_strategy_graph)
model_train = node2vec_train.fit()
def embedding_train(u):
        return model_train.wv[u]

# Only use Hadamard for simplicity
def operator_hadamard(u, v):
    return u * v

def operator_l1(u, v):
    return np.abs(u - v)

def operator_l2(u, v):
    return (u - v) ** 2

def operator_avg(u, v):
    return (u + v) / 2.0
# binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]
binary_operators = [operator_hadamard]