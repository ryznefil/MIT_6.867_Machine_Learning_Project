import pandas as pd
from utils import run_link_prediction, evaluate_link_prediction_model
from config import *

results = [run_link_prediction(op, embedding_train) for op in binary_operators]
best_result = max(results, key=lambda result: result["score"])

print(f"Result from '{best_result['binary_operator'].__name__}'")

pd.DataFrame(
    [(result["binary_operator"].__name__, result["score"]) for result in results],
    columns=("name", "ROC AUC score"),
).set_index("name")

### Evaluate the best model using the test set ###
node2vec_test = Node2Vec(graph_test_nx, p=p, q=q, dimensions=dimensions, walk_length=walk_length,
                          num_walks=num_walks, workers=workers, sampling_strategy=sampling_strategy_graph)
model_test = node2vec_test.fit()
def embedding_test(u):
        return model_test.wv[u]

test_score = evaluate_link_prediction_model(
    best_result["classifier"],
    examples_test,
    labels_test,
    embedding_test,
    best_result["binary_operator"],
)
print(
    f"ROC AUC score on test set using '{best_result['binary_operator'].__name__}': {test_score}"
)
