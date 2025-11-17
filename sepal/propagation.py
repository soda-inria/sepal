from tqdm import tqdm
import torch
import numpy as np
from pykeen.nn.compositions import composition_resolver


def propagate_embeddings(ctrl, graph, core_embed, relations_embed, subgraphs, **kwargs):
    """Propagates embeddings from a set of core entities to the entire graph.

    This function implements an iterative propagation mechanism. It starts with
    known embeddings for a "core" subgraph and iteratively updates the embeddings
    of other nodes. The process is performed on a series of specified subgraphs
    to manage memory and computation, especially for large graphs.

    For each subgraph, it performs a fixed number of propagation steps. In each
    step, node embeddings are updated based on their neighbors' embeddings and the
    connecting relation embeddings, using a specified composition function. After
    the update, embeddings are normalized. The embeddings of the core entities are
    reset to their original values after each step, treating them as fixed anchors.

    Args:
        ctrl (object): A control object containing configuration parameters such as
            `embed_dim`, `embed_method`, `propagation_device`, `device`,
            `n_propagation_steps`, `propagation_lr`, and `embed_setting`.
        graph (object): A graph object that provides access to graph properties
            like the number of entities, core subgraph indices, and methods to
            extract subgraph edges and types.
        core_embed (torch.Tensor): A tensor containing the pre-computed embeddings
            for the core entities.
        relations_embed (torch.Tensor): A tensor containing the embeddings for all
            relation types in the graph.
        subgraphs (list of np.ndarray): A list of arrays, where each array contains
            the entity indices of a subgraph to be processed.
        **kwargs: Additional keyword arguments to be passed to the composition
            function.

    Returns:
        torch.Tensor: A tensor of shape (num_entities, embed_dim) containing the
            propagated embeddings for all entities in the graph, located on the CPU.
    """
    # Initialize embeddings with zeros outside the core subgraph
    embeddings = torch.zeros((graph.num_entities, ctrl.embed_dim))
    if ctrl.embed_method == "rotate":
        embeddings = torch.zeros(
            (graph.num_entities, ctrl.embed_dim), dtype=torch.cfloat
        )

    # Define the device (if ctrl has attribute propagation_device, use it; otherwise, use ctrl.device)
    if hasattr(ctrl, "propagation_device"):
        device = ctrl.propagation_device
        relations_embed = relations_embed.to(device)
    else:
        device = ctrl.device

    # Sort core_embed based on core_subgraph_idx
    core_embed = core_embed[np.argsort(graph.core_subgraph_idx)].to(device)

    for subgraph_idx in tqdm(subgraphs, desc="Propagating through subgraphs"):
        # Add core subgraph
        subgraph_idx = np.union1d(subgraph_idx, graph.core_subgraph_idx)
        core_idx = np.where(np.isin(subgraph_idx, graph.core_subgraph_idx))[0]

        # Load subgraph embeddings on device
        outer_idx = np.where(~np.isin(subgraph_idx, graph.core_subgraph_idx))[0]
        subgraph_embed = torch.zeros_like(embeddings[subgraph_idx], device=device)
        subgraph_embed[outer_idx] = embeddings[subgraph_idx[outer_idx]].to(device)
        subgraph_embed[core_idx] = core_embed

        # Create edge_index and edge_type variables
        subgraph_edge_index, subgraph_edge_type = (
            graph.get_subgraph_edge_index_and_type(subgraph_idx)
        )
        subgraph_edge_index = subgraph_edge_index.to(device)
        subgraph_edge_type = subgraph_edge_type.to(device)

        for _ in range(ctrl.n_propagation_steps):
            # Propagate
            subgraph_embed += ctrl.propagation_lr * composition(
                subgraph_embed,
                relations_embed,
                subgraph_edge_index,
                subgraph_edge_type,
                kind=ctrl.embed_setting.composition,
                **kwargs,
            )

            # Normalize
            subgraph_embed = torch.nn.functional.normalize(subgraph_embed, p=2, dim=1)

            # Reset core subgraph
            subgraph_embed[core_idx] = core_embed

        # Load outer embeddings back to CPU
        embeddings[subgraph_idx[outer_idx]] = subgraph_embed[outer_idx].cpu()

    # Load core embeddings on CPU
    embeddings[np.sort(graph.core_subgraph_idx)] = core_embed.cpu()
    return embeddings


def composition(
    entity_embed,
    relations_embed,
    edge_index,
    edge_type,
    kind="multiplication",
    batch_size=100000,
    **kwargs,
):
    source, target = edge_index

    # Create an empty tensor that has same shape as entity_embed
    new_embed = torch.zeros_like(entity_embed)

    # Loop over the edges in batches
    for start in range(0, len(edge_type), batch_size):
        stop = min(start + batch_size, len(edge_type))

        # compose
        message = composition_resolver.make(kind)(
            entity_embed[source[start:stop]], relations_embed[edge_type[start:stop]]
        )

        # aggregate by sum
        new_embed = new_embed.index_add(dim=0, index=target[start:stop], source=message)

    return new_embed
