import numpy as np
import torch
from tqdm import tqdm
import scipy.sparse
from pykeen.triples import TriplesFactory

from knowledge_graph import KnowledgeGraph
from utils import bfs_path_to_subgraph_csr, pagerank


def extract_subgraph(ctrl, graph):
    if ctrl.core_selection == "degree":
        node_list = central_extraction(ctrl.core_prop, graph)
    elif ctrl.core_selection == "pagerank":
        node_list = central_extraction(
            ctrl.core_prop, graph, centrality_measure="pagerank"
        )
    elif ctrl.core_selection == "hybrid":
        node_list = hybrid_extraction(
            ctrl.core_node_proportions, ctrl.core_edge_proportions, graph
        )
    else:
        raise ValueError(
            "core selection strategy must be in {'degree', 'pagerank', 'hybrid'}"
        )
    if ctrl.handle_disconnected:
        # Add all nodes not in the largest connected component to the node list
        n_components, labels = scipy.sparse.csgraph.connected_components(
            graph.adjacency, directed=False, return_labels=True
        )
        largest_cc_label = np.argmax(np.bincount(labels))
        non_lcc_nodes = np.where(labels != largest_cc_label)[0]
        node_list.extend(non_lcc_nodes.tolist())
        node_list = list(set(node_list))
        graph.non_lcc_nodes = non_lcc_nodes.tolist()

    ctrl.core_subgraph_idx = [node_list]
    ctrl.core_size = len(node_list)
    subgraph = extract_subgraph_from_node_list(graph, node_list)
    return subgraph


def hybrid_extraction(node_prop, edge_prop, graph):
    """
    Select the entities to create the core subgraph by:
    1. Sampling the nodes with the top k% degrees.
    2. Sampling the edges with the top k% degrees (sum of degrees of head and tail) of each relation type. Getting the involved nodes.
    3. Taking the union of these two sets of nodes.
    4. If necessary, adding the nodes required to have only one connected component.
    """
    print("Hybrid core extraction:")

    ## 1. Sample the nodes with the top k% degrees.
    print(f"    1. Sampling the nodes with top {100*node_prop}% degrees", end="\r")
    node_list_tensor = torch.argsort(torch.IntTensor(graph.degrees), descending=True)[
        : int(node_prop * graph.num_entities) + 1
    ]
    degree_node_set = set(node_list_tensor.tolist())

    ## 2. Sample the edges with the top k% degrees for each relation type.
    print(
        f"    2. Sampling the edges with top {100*edge_prop}% degrees for each relation type",
        end="\r",
    )
    edges = graph.get_mapped_triples()

    # Collect nodes for each relation
    rel_node_set = set()
    for rel in tqdm(range(graph.num_relations)):
        # Filter edges for the current relation
        rel_edges = edges[edges[:, 1] == rel]

        # Calculate the sum of degrees for each edge
        edge_degrees = graph.degrees[rel_edges[:, [0, 2]]].sum(axis=1)

        # Calculate the number of edges to consider based on edge_prop
        n_edges = int(edge_prop * len(rel_edges)) + 1

        # Indices of the k largest elements
        indices = np.argpartition(edge_degrees, -n_edges)[-n_edges:]

        # Collect unique nodes from the selected edges
        rel_node_set.update(np.unique(rel_edges[indices][:, [0, 2]]))

    ## 3. Take the union of these two sets of nodes.
    print("    3. Taking the union of these two sets of nodes.", end="\r")
    node_set = degree_node_set | rel_node_set
    node_list = list(node_set)

    # Add all the edges linking those entities
    node_list_tensor = torch.IntTensor(node_list)
    mask = torch.isin(edges[:, [0, 2]], node_list_tensor).all(axis=1)
    subgraph_edges = edges[mask]
    old_to_new_reindex = {node_list[i]: i for i in range(len(node_list))}
    subgraph_edges[:, [0, 2]] = torch.tensor(
        np.vectorize(old_to_new_reindex.__getitem__)(subgraph_edges[:, [0, 2]])
    )

    ## 4. If necessary, add the nodes required to have only one connected component.
    print(
        "    4. Adding the nodes required to have only one connected component.",
        end="\r",
    )
    # Compute connected components
    num_nodes = len(node_list)
    subgraph_adjacency = scipy.sparse.csr_matrix(
        (
            np.ones(len(subgraph_edges)),
            (subgraph_edges[:, 0], subgraph_edges[:, 2]),
        ),
        shape=(num_nodes, num_nodes),
    )
    n_components, labels = scipy.sparse.csgraph.connected_components(
        subgraph_adjacency, directed=False, return_labels=True
    )
    new_to_old_reindex = {v: k for k, v in old_to_new_reindex.items()}

    # Get largest connected component (lcc)
    lcc_id = np.argmax(np.bincount(labels))
    lcc_entities = set(
        np.vectorize(new_to_old_reindex.__getitem__)(np.where(labels == lcc_id)[0])
    )

    # Iterate over the other components
    for i in tqdm(range(n_components)):
        if i != lcc_id:
            # Get the current connected component ids
            cc_entities = np.vectorize(new_to_old_reindex.__getitem__)(
                np.where(labels == i)[0]
            )

            # Perform a BFS from the highest degree node in current connected component to the lcc
            source = cc_entities[np.argmax(graph.degrees[cc_entities])]
            path_nodes = bfs_path_to_subgraph_csr(graph.adjacency, source, lcc_entities)

            # Add the nodes of the path and the connected component to the lcc
            lcc_entities.update(path_nodes)
            lcc_entities.update(cc_entities)

            # Add the nodes of the path to the core subgraph
            node_set.update(path_nodes)

    node_list = list(node_set)

    return node_list


def central_extraction(prop, graph, centrality_measure="degree"):
    """
    Select entities of high centrality to create the core subgraph.
    Keep only the resulting largest connected component.
    """
    print("Central core extraction:")
    ## Select the core subgraph
    # Select high degree nodes
    print("    1. Getting high centrality nodes", end="\r")
    edges = graph.get_mapped_triples()[:, [0, 2]]
    if centrality_measure == "degree":
        node_list_tensor = torch.argsort(
            torch.IntTensor(graph.degrees), descending=True
        )[: int(prop * graph.num_entities) + 1]
    elif centrality_measure == "pagerank":
        pageranks = pagerank(graph.adjacency, alpha=0.85)
        node_list_tensor = torch.argsort(torch.FloatTensor(pageranks), descending=True)[
            : int(prop * graph.num_entities) + 1
        ]
    node_list = node_list_tensor.tolist()
    mask = torch.isin(edges, node_list_tensor).all(axis=1)
    subgraph_edges = edges[mask]
    num_nodes = len(node_list)
    reindex = {node_list[i]: i for i in range(len(node_list))}
    subgraph_edges = torch.tensor(np.vectorize(reindex.__getitem__)(subgraph_edges))

    # Keep only the largest connected component of the subgraph
    print("    2. Keeping largest connected component", end="\r")
    subgraph_adjacency = scipy.sparse.csr_matrix(
        (np.ones(len(subgraph_edges)), (subgraph_edges[:, 0], subgraph_edges[:, 1])),
        shape=(num_nodes, num_nodes),
    )
    n_components, labels = scipy.sparse.csgraph.connected_components(
        subgraph_adjacency, directed=False, return_labels=True
    )
    selected_nodes = np.where(labels == np.argmax(np.bincount(labels)))[0]
    reindex_inv = {v: k for k, v in reindex.items()}
    node_list = [reindex_inv[i] for i in selected_nodes]

    ## Extract subgraph
    return node_list


def extract_subgraph_from_node_list(graph, node_list):
    ## Build triples_factory
    print("    ... building triples factory         ", end="\r")
    # Remove entities that are outside of the subgraph
    old_tf = graph.triples_factory
    mask = torch.isin(old_tf.mapped_triples[:, [0, 2]], torch.IntTensor(node_list)).all(
        axis=1
    )
    mapped_triples = old_tf.mapped_triples[mask]

    # Reindex subgraph entities between 0 and n-1
    d = {node_list[i]: i for i in range(len(node_list))}
    mapped_triples[:, [0, 2]] = torch.tensor(
        np.vectorize(d.__getitem__)(mapped_triples[:, [0, 2]])
    )

    # Build the new entity_to_id dictionnary
    old_id_to_entity = {v: k for k, v in old_tf.entity_to_id.items()}
    d_inv = {v: k for k, v in d.items()}
    id_to_entity = {i: old_id_to_entity[d_inv[i]] for i in range(len(node_list))}
    entity_to_id = {v: k for k, v in id_to_entity.items()}

    # Create triple factory object
    triples_factory = TriplesFactory(
        mapped_triples=mapped_triples,
        entity_to_id=entity_to_id,
        relation_to_id=old_tf.relation_to_id,
        create_inverse_triples=old_tf.create_inverse_triples,
    )

    ## Build knowledge_graph instance
    print("    ... building knowledge graph object", end="\r")
    subgraph = KnowledgeGraph(triples_factory, graph.name + "_core")
    graph.core_subgraph_idx = node_list

    ## Print number of entities
    print(
        f"Core subgraph contains {subgraph.num_entities} entities ({subgraph.num_entities/graph.num_entities:.1%} of total graph)"
    )

    return subgraph
