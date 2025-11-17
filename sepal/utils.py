import numpy as np
import scipy.sparse
from collections import deque
import hashlib


def bfs_path_to_subgraph_csr(graph, source, subgraph_nodes):
    """
    Perform BFS on a graph represented as a csr_matrix to find the path from
    the source node to the closest node in the subgraph.

    Parameters:
        graph (csr_matrix): Sparse adjacency matrix of the graph (unweighted).
        source (int): Index of the source node.
        subgraph_nodes (set of int): Set of nodes in the target subgraph.

    Returns:
        list: The path from the source to the closest subgraph node, or an empty list if no path exists.
    """
    # Queue for BFS: stores (current_node, path_so_far)
    queue = deque([(source, [source])])

    # Track visited nodes
    visited = set()

    while queue:
        current_node, path = queue.popleft()

        # Check if we've reached a node in the subgraph
        if current_node in subgraph_nodes:
            return path

        # Mark the current node as visited
        visited.add(current_node)

        # Get neighbors of the current node (non-zero entries in the sparse matrix row)
        neighbors = graph[current_node].indices

        # Enqueue unvisited neighbors
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

    return []  # Return an empty path if no connection exists


def subgraph_hash(ctrl, dataset):
    subgraph_parameters = [
        # Dataset parameters
        dataset,
        # Core selection parameters
        ctrl.core_selection,
        ctrl.core_prop,
        ctrl.core_edge_proportions,
        ctrl.core_node_proportions,
        # Partitioning parameters
        ctrl.subgraph_max_size,
        ctrl.diffusion_stop,
    ]

    subgraphs_hash = hashlib.sha256(
        str(subgraph_parameters).encode("utf-8")
    ).hexdigest()
    return subgraphs_hash


def pagerank(adj_matrix, alpha=0.85, tol=1e-6, max_iter=100):
    """
    Compute PageRank using power iteration on a sparse matrix.

    Parameters:
        adj_matrix: scipy.sparse matrix (CSR preferred)
        alpha: damping factor
        tol: convergence tolerance
        max_iter: max number of iterations

    Returns:
        PageRank vector (numpy array)
    """
    n = adj_matrix.shape[0]

    # Normalize columns (out-degree)
    out_degree = np.array(adj_matrix.sum(axis=0)).flatten()
    out_degree[out_degree == 0] = 1  # prevent division by zero
    stochastic_matrix = adj_matrix / out_degree

    # Ensure it's sparse
    M = scipy.sparse.csr_matrix(stochastic_matrix)

    # Personalization vector (uniform)
    teleport = np.ones(n) / n

    # Initialize rank vector
    rank = np.ones(n) / n

    for _ in range(max_iter):
        new_rank = alpha * (M @ rank) + (1 - alpha) * teleport
        if np.linalg.norm(new_rank - rank, 1) < tol:
            break
        rank = new_rank

    return rank
