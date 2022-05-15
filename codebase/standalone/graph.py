import networkx as nx
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
from sklearn.model_selection import train_test_split

### Load dataset/define graph ###
def data_selector(i):
    if i == 1: # Facebook
        f = open("./data/facebook_combined.txt", "r")
        edges = f.readlines()
        dataset = StellarGraph.from_networkx(nx.parse_edgelist(edges, nodetype=int))

    elif i == 2: # PPI
        f = open("./data/PPI.csv", "r")
        Graphtype = nx.Graph()
        g = nx.parse_edgelist(f, delimiter=',', create_using=Graphtype, nodetype=int)
        relabeled_G = nx.relabel.convert_node_labels_to_integers(g, first_label=0, ordering='default',label_attribute=None)
        #dataset = StellarGraph.from_networkx(nx.parse_edgelist(f, delimiter=',', create_using=Graphtype,nodetype=int))
        dataset = StellarGraph.from_networkx(relabeled_G)

    elif i == 3: # arXiv
        f = open("./data/arXiv.txt", "r")
        edges = f.readlines()
        g = nx.parse_edgelist(edges, nodetype=int)
        relabeled_G = nx.relabel.convert_node_labels_to_integers(g, first_label=0, ordering='default', label_attribute=None)
        dataset = StellarGraph.from_networkx(relabeled_G)

    else:
        raise RuntimeError("Invalid dataset selection")

    print("INITIAL GRAPH")
    print(dataset.info())
    #graph = dataset.subgraph(next(dataset.connected_components()))
    graph = dataset
    print("REDUCED GRAPH OF ONE COMPONENT")
    print(graph.info())

    return graph


### Construct splits of the input data ###
def test_splitter(graph):
    ## TEST GRAPH ##
    # Define an edge splitter on the original graph:
    edge_splitter_test = EdgeSplitter(graph, graph)

    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph, and obtain the
    # reduced graph graph_test with the sampled links removed:
    graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global"
    )
    print(graph_test.info())

    return graph_test, examples_test, labels_test

def train_splitter(graph, graph_test):
    ## TRAIN GRAPH ##
    # Do the same process to compute a training subset from within the test graph
    edge_splitter_train = EdgeSplitter(graph_test, graph)
    graph_train, examples, labels = edge_splitter_train.train_test_split(
        p=0.1, method="global"
    )
    (
        examples_train,
        examples_model_selection,
        labels_train,
        labels_model_selection,
    ) = train_test_split(examples, labels, train_size=0.75, test_size=0.25)

    print(graph_train.info())

    return graph_train, examples, labels, examples_train, examples_model_selection, labels_train, labels_model_selection