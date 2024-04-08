# Enhanced node2vec: Incorporating Node Centrality for Optimized Graph Embeddings

## Introduction
This project presents an enhancement to the node2vec graph embedding algorithm, which traditionally uses biased random walks to encode nodes into a low-dimensional feature space. The original node2vec algorithm, while effective, applies constant hyperparameters across all nodes, neglecting the unique structural roles individual nodes may play within the graph. Our research introduces a novel approach that adapts the random walk hyperparameters based on node centrality, aiming to improve the representation power of node2vec embeddings.

## Methodology
- **Centrality-Based Parameterization**: We propose a linear-time expansion of node2vec that utilizes first and second-degree centrality measures to dynamically adjust the random walk hyperparameters (p and q) for each node.
- **Hyperparameter Optimization**: The method includes fine-tuning of the centrality-based approach, allowing for out-of-the-box deployment on any graph, and optimization of the original node2vec hyperparameters on baseline graphs for comparison.
- **Validation**: The enhanced node2vec is validated against a random hyperparameter approach and the optimized original node2vec on baseline graphs.

## Key Findings
- **Performance**: Our centrality-adjusted node2vec outperformed the original algorithm on the Cora dataset, demonstrating the potential of leveraging node centrality for graph embeddings.
- **Computation Time**: The additional computation time required for centrality-based parameterization is negligible compared to the overall time for random walk generation.
- **Tuning of α and β**: Empirical fine-tuning on the Cora dataset led to the selection of α = 0.1 and β = 0.01 for the mapping of centrality measures to p and q.

## Impact and Applications
The enhanced node2vec algorithm has the potential to improve the quality of graph embeddings, particularly for graphs with pronounced structural patterns. This can benefit various applications, including link prediction, node classification, and network analysis, by providing more accurate representations of nodes based on their roles within the graph.

## Repository Structure
- **Algorithm Implementation**: Code for the enhanced node2vec algorithm with centrality-based parameterization.
- **Benchmark Comparisons**: Scripts for comparing the enhanced node2vec against the original and randomized hyperparameter selection methods.
- **Datasets**: The Cora and CiteSeer datasets used for tuning and evaluation.
- **Results**: Detailed results and analysis of the performance comparisons.

## Collaborators and Acknowledgments
- Filip Ryzner, El Ghali Zerhouni, and Makram Chahine contributed to the research, with each member focusing on different aspects such as implementation, theoretical backing, and empirical testing.
- This work was supported by [Funding Source], and we thank [Institution] for providing the computational resources.

## Related Resources
- Original node2vec Paper: [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653)
- DeepWalk Paper: [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)

---

For more information on how to use the enhanced node2vec algorithm and to contribute to the project, please visit the [GitHub repository](https://github.com/EnhancedNode2Vec).

*Note: The repository link above is a placeholder and should be replaced with the actual URL of the GitHub repository.*
