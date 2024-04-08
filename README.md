# Enhancing the Random Walk Parametrization in node2vec Graph Embedding Algorithm

This repository houses the implementation and results of an improved version of the node2vec graph embedding algorithm, aimed at leveraging neighborhood structural information in graph data.

## Project Overview

- **Objective**: To enhance the node2vec algorithm's representation capabilities by customizing transition probability hyperparameters (p and q) using node centrality measures.
- **Methods**: Our method modifies node2vec by introducing a linear time preprocessing step that calculates p and q based on first and second-degree node centralities, thus allowing for node-specific bias adjustment in random walks.
- **Results**: The approach demonstrates improved link prediction performance on the Cora dataset but shows mixed results on the CiteSeer dataset, indicating the need for structure-aware tuning.
- **Significance**: The method points to the potential benefits of considering graph structural information in random walk-based embedding algorithms, especially for graphs with discernible patterns.

## Implementation Details

- We propose an expansion of the node2vec algorithm that takes the centrality of nodes into account, defining an algorithmic mapping that outputs node-specific hyperparameters p and q.
- Fine tuning of constants α and β over a variety of graph architectures is performed, enabling the out-of-the-box deployment on any graph without the need for re-tuning of p and q.
- Optimization of the original node2vec on three baseline graphs is conducted to provide a comparison benchmark.
- Validation of the method is done against a set of baseline graphs, comparing the performance of standard node2vec with constant hyperparameters and a randomized selection approach.

## Potential Impact and Applications

- Our method is shown to significantly improve link prediction tasks on graphs that exhibit strong structural patterns, suggesting effectiveness in networks where centrality measures can reflect meaningful topology.
- The linear time computation of the modified node2vec makes it suitable for large-scale graph analytics, preserving efficiency while adding enhanced performance.
- This research provides insights into the importance of node-centric parameter tuning in graph embeddings, potentially influencing future works on graph-based machine learning models and applications involving network analysis.

## Project Structure

- `Centrality_Based_Parametrization`: The core algorithm modification for the centrality-based computation of random walk parameters p and q.
- `Performance_Evaluation`: Scripts and notebooks for evaluating the performance of the modified algorithm against benchmark models.
- `Datasets_Tuning_Results`: Empirical results following the tuning of α and β parameters on Cora and CiteSeer datasets and their performance comparison against original node2vec and randomized selection approaches.

## Collaborators and Acknowledgments

- Filip Ryzner, El Ghali Zerhouni, and Makram Chahine led the project development, each contributing to critical aspects including implementation, theoretical backing, and experimental setup.
- This work does not cite external funding sources.
- Related work includes references to prior studies on node2vec and random walk-based graph embeddings.

---

For complete documentation and usage instructions, refer to the project Wiki and issue tracking for active discussions on improvements and community feedback. 

Explore the repository to learn more about how node centrality can enhance graph embeddings in tasks such as link prediction and network analysis. Your constructive contributions and insights are highly valued in this ongoing research endeavor.