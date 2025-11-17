from pathlib import Path
import pickle
import numpy as np
import scipy.sparse
import warnings

from utils import subgraph_hash


def make_subgraphs(ctrl, graph):
    """Generates a list of subgraphs from the knowledge graph.
    The subgraphs are generated based on the BLOCS algorithm.
    If the number of entities in the graph is less than the maximum subgraph size,
    a single subgraph containing all entities is returned.
    Args:
        ctrl: Control parameters containing partitioning method and maximum subgraph size.
        graph: The knowledge graph from which to generate subgraphs.
    Returns:
        subgraphs: A list of subgraphs, each represented as a numpy array of entity indices.
    """
    # Check if partitioning is needed
    if graph.num_entities < ctrl.subgraph_max_size:
        subgraphs = [np.arange(graph.num_entities)]

    # If so, check if subgraphs have already been generated
    subgraphs_hash = subgraph_hash(ctrl, graph.name)
    subgraphs_dir = Path(__file__).resolve().parents[1] / "subgraphs_files"
    subgraphs_dir.mkdir(parents=True, exist_ok=True)
    subgraphs_file = subgraphs_dir / f"{subgraphs_hash}.pkl"

    if Path(subgraphs_file).is_file():
        ctrl.load_subgraphs = True
        with open(subgraphs_file, "rb") as f:
            subgraphs = pickle.load(f)

    else:
        ctrl.load_subgraphs = False
        # If not, generate subgraphs
        subgraphs = generate_blocs(
            graph,
            ctrl.subgraph_max_size,
            diffusion_stop=ctrl.diffusion_stop,
            handle_disconnected=ctrl.handle_disconnected,
        )
        # Save subgraphs for next time
        with open(subgraphs_file, "wb") as f:
            pickle.dump(subgraphs, f)

    ctrl.num_subgraphs = len(subgraphs)

    return subgraphs


def generate_blocs(
    graph, max_size, diffusion_stop=0.8, verbose=True, handle_disconnected=False
):
    """Generates a list of overlapping connected subgraphs covering the graph.

    This function implements the BLOCS (Balanced Overlapping Connected Subgraphs)
    algorithm to partition a large graph into smaller, manageable subgraphs.
    The process involves several steps:
    1. Seeding subgraphs from high-degree nodes ("super-spreaders").
    2. Growing subgraphs using a diffusion process.
    3. Expanding subgraphs via dilation to cover all remaining nodes.
    4. Iteratively merging and splitting subgraphs to adhere to size constraints.

    Args:
        graph: The graph object to partition.
        max_size (int): The desired maximum size for each subgraph.
        diffusion_stop (float, optional): The proportion of nodes to assign
            during the initial diffusion phase. Defaults to 0.8.
        verbose (bool, optional): Whether to print progress information.
            Defaults to True.
        handle_disconnected (bool, optional): If True, operates only on the
            largest connected component of the graph. Defaults to False.

    Returns:
        list[np.ndarray]: A list of subgraphs, where each subgraph is
            represented as a numpy array of node indices.
    """
    # Graph adjacency
    A = graph.get_adjacency_with_self_loops().tocsc()

    # Unassigned entities
    unassigned = set(np.arange(graph.num_entities))

    # List of subgraphs
    subgraphs = []

    # If the graph is disconnected, deal only with the largest connected component
    if handle_disconnected:
        unassigned = unassigned - set(graph.non_lcc_nodes)

    # Deal with very high degree nodes
    print("Assigning super-spreaders' neighbors...")
    subgraphs, unassigned = split_super_spreaders_neighbors(
        graph, subgraphs, unassigned, max_size, verbose
    )

    print("Diffusion on the graph...")
    subgraphs, unassigned = subgraph_diffusion(
        diffusion_stop, subgraphs, A, graph, unassigned, 0.8 * max_size, verbose
    )

    # Merge small subgraphs
    print("Merging subgraphs...")
    subgraphs = merge_subgraphs(subgraphs, 0.8 * max_size)

    # Subgraph dilation
    print("Subgraph dilation...")
    m = 1
    surrounded = (
        []
    )  # indices of subgraphs that do not have unassigned entities in their neighborhood
    while len(unassigned) > 0:
        # Create more subgraphs if needed
        if m % 5 == 1 and m > 1:
            for _ in range(10):
                subgraphs, unassigned = diffusion_step(
                    subgraphs, A, graph, unassigned, 0.8 * max_size, verbose=False
                )
                if len(unassigned) == 0:
                    break

        # Grow the existing subgraphs by dilation
        if len(unassigned) > 0:
            subgraphs, unassigned, surrounded = subgraph_dilation(
                subgraphs, graph, unassigned, surrounded
            )
            m += 1
    print("")

    print("Merging small subgraphs...")
    subgraphs = merge_small_subgraphs(subgraphs, max_size)

    print("Splitting large subgraphs...")
    subgraphs = split_large_subgraphs(graph, subgraphs, max_size)
    subgraphs = merge_subgraphs(subgraphs, max_size)

    sizes = np.array([s.size for s in subgraphs])
    print(f"Subgraph sizes: min: {min(sizes)}, max: {max(sizes)}")

    return subgraphs


def split_super_spreaders_neighbors(graph, subgraphs, unassigned, max_size, verbose):
    neigh_mat = graph.adjacency.tocoo()

    high_degree_nodes = np.where(graph.degrees > 0.2 * max_size)[0]
    for i in high_degree_nodes:
        neighbors = neigh_mat.row[neigh_mat.col == i]
        subgraphs += [
            np.append(arr, [i])
            for arr in np.array_split(
                neighbors, int(graph.degrees[i] / (0.2 * max_size)) + 1
            )
        ]
        unassigned = unassigned - set(neighbors)
        if verbose:
            print(f"{1 - len(unassigned)/graph.num_entities:.1%} assigned", end="\r")
    unassigned = unassigned - set(list(high_degree_nodes))
    print("")
    return subgraphs, unassigned


def merge_small_subgraphs(subgraphs, max_size):
    sizes = np.array([s.size for s in subgraphs])
    small_subgraphs_idx = np.where(sizes < 0.4 * max_size)[0]
    to_remove = []
    for i in small_subgraphs_idx:
        for j, s2 in enumerate(subgraphs):
            if j != i:
                if np.isin(subgraphs[i], s2).any():
                    subgraphs[i] = np.union1d(subgraphs[i], s2)
                    to_remove.append(j)
                    break
            else:
                continue
    # Remove the redundant subgraphs
    subgraphs = [s for idx, s in enumerate(subgraphs) if idx not in to_remove]
    return subgraphs


def split_large_subgraphs(graph, subgraphs, max_size):
    sizes = np.array([s.size for s in subgraphs])
    large_subgraphs_idx = np.where(sizes > max_size)[0]
    for i in large_subgraphs_idx:
        subgraph = subgraphs[i]
        # Remove core subgraph
        entities = np.array(list(set(subgraph) - set(graph.core_subgraph_idx)))
        if entities.size == subgraph.size:
            warnings.warn(f"Subgraph {i} is not connected to the core subgraph")
        # Count connected components
        nb_of_cc, labels = scipy.sparse.csgraph.connected_components(
            graph.adjacency[entities, :][:, entities], return_labels=True
        )
        cc_sizes = np.bincount(labels)
        # Add connected components until it reaches limit size
        sorted_cc_idx = np.argsort(cc_sizes)[::-1]
        cumsum = cc_sizes[sorted_cc_idx].cumsum()
        limit = subgraph.size // (subgraph.size // max_size + 1) + 1
        limit = (limit + max_size) // 2

        pointer = 0  # all entities left to pointer have been dealt with at all time
        previous_bound = 0
        entities_done = 0
        first = True
        while pointer < cc_sizes.size:
            if cumsum[pointer] - entities_done > limit:
                # Add a new subgraph
                if pointer != previous_bound:
                    # In this case the subgraph size will be smaller than limit
                    cc_idx = sorted_cc_idx[previous_bound:pointer]
                    new_subgraph = entities[np.where(np.isin(labels, cc_idx))[0]]
                    previous_bound = pointer
                    # in this case pointer is NOT incremented

                else:
                    # In this case the subgraph size will be greater than limit
                    # This will be the case if a connected component has size greater than limit
                    new_subgraph = entities[
                        np.where(labels == sorted_cc_idx[pointer])[0]
                    ]
                    previous_bound = pointer + 1
                    pointer += 1

                if first:
                    subgraphs[i] = new_subgraph
                else:
                    subgraphs.append(new_subgraph)

                entities_done += new_subgraph.size
                first = False
                if new_subgraph.size > max_size:
                    warnings.warn(f"Created subgraph with size {new_subgraph.size}")
            else:
                pointer += 1
        # Add the remaining entities
        cc_idx = sorted_cc_idx[previous_bound:]
        new_subgraph = entities[np.where(np.isin(labels, cc_idx))[0]]
        subgraphs.append(new_subgraph)
    return subgraphs


def subgraph_diffusion(
    diffusion_stop, subgraphs, A, graph, unassigned, max_size, verbose
):
    while 1 - len(unassigned) / graph.num_entities < diffusion_stop:
        subgraphs, unassigned = diffusion_step(
            subgraphs, A, graph, unassigned, max_size, verbose
        )
        if len(unassigned) == 0:
            break
    print("")
    return subgraphs, unassigned


def diffusion_step(subgraphs, A, graph, unassigned, max_size, verbose):
    # Set diffuser to the highest degree unassigned entity
    unassigned_lst = list(unassigned)
    diffuser = unassigned_lst[np.argmax(graph.degrees[unassigned_lst])]

    # other option: Set a random diffuser
    # diffuser = np.random.choice(list(unassigned))

    # Diffusion
    previous_x = scipy.sparse.csc_matrix(
        ([1], ([diffuser], [0])), shape=(graph.num_entities, 1), dtype="float32"
    )
    x = A * previous_x

    while x.nnz < max_size and previous_x.nnz < x.nnz:
        previous_x = x
        x = A * x

    x = previous_x
    x = x.tocoo()
    subgraphs.append(x.row)
    unassigned = unassigned - set(x.row)

    # Verbose
    if verbose:
        print(f"{1 - len(unassigned)/graph.num_entities:.1%} assigned", end="\r")
    return subgraphs, unassigned


def subgraph_dilation(subgraphs, graph, unassigned, surrounded):
    # Permute subgraphs to make smaller subgraphs grow first
    sizes = [s.shape[0] for s in subgraphs]
    permutation = np.argsort(sizes)

    adj = graph.adjacency

    for i in permutation:
        if i not in surrounded:
            subgraph = subgraphs[i]
            x = scipy.sparse.csc_matrix(
                (np.ones(subgraph.size), (subgraph, np.zeros(subgraph.size))),
                shape=(adj.shape[0], 1),
                dtype="float32",
            )
            x = adj * x
            unassigned_neighbors = set(x.tocoo().row).intersection(unassigned)

            if len(unassigned_neighbors) > 0:
                subgraphs[i] = np.hstack([subgraph, list(unassigned_neighbors)])
                unassigned = unassigned - unassigned_neighbors

            else:
                surrounded.append(i)

            print(f"{len(unassigned)} remaining     ", end="\r")
            if len(unassigned) == 0:
                break

    return subgraphs, unassigned, surrounded


def merge_subgraphs(subgraphs, max_size):
    """
    Aggregates small subgraphs that share common entities.
    """
    print(f"{len(subgraphs)} subgraphs before merging.")
    # Sort subgraphs by size
    sizes = [s.shape[0] for s in subgraphs]
    subgraphs = [subgraphs[i] for i in np.argsort(sizes)]

    # Check subgraphs with sizes up to max_size/2
    subgraphs_to_check = subgraphs[: (np.sort(sizes) < max_size / 2).sum()]

    # Merge compatible subgraphs
    new_subgraphs = []
    while len(subgraphs_to_check) > 0:
        s = subgraphs_to_check.pop(0)
        i = 0
        while (
            i < len(subgraphs_to_check)
            and s.shape[0] + subgraphs_to_check[i].shape[0] < max_size
        ):
            if np.isin(s, subgraphs_to_check[i]).any():
                s = np.union1d(s, subgraphs_to_check.pop(i))
            else:
                i += 1
        new_subgraphs.append(s)
    subgraphs = new_subgraphs + subgraphs[(np.sort(sizes) < max_size / 2).sum() :]
    print(f"{len(subgraphs)} subgraphs after merging.")
    return subgraphs
