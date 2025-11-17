from time import time

from core_extraction import extract_subgraph
from blocs import make_subgraphs
from propagation import propagate_embeddings
from core_embed import embed


def run_sepal(ctrl, graph):
    """Executes the SEPAL embedding pipeline on a given graph.

    This function orchestrates the SEPAL (Scalable Embedding Propagation
    ALgorithm) process. It follows these main steps:
    1.  Generates a core subgraph and a set of overlapping subgraphs from the
        main graph using BLOCS decomposition.
    2.  Computes initial embeddings for the nodes and relations within the
        core subgraph using the method specified in the control object.
    3.  Propagates the learned embeddings from the core to the rest of the
        nodes in the full graph.

    The total execution time is calculated and stored in the control object.

    Args:
        ctrl: The control object containing configuration parameters (e.g.,
              `embed_method`) and for storing results (e.g., `total_time`).
        graph: The main graph object to be processed.

    Returns:
        A tuple containing:
        - torch.Tensor: The final entity embeddings for the entire graph.
        - torch.Tensor: The embeddings for the relations.
        - float: The total time taken for the process in seconds.
    """
    ### Compute Embeddings
    start = time()

    ## Step 1: Generate subgraphs
    # 1.a. Select core subgraph
    core_graph = extract_subgraph(ctrl, graph)
    # 1.b. BLOCS decomposition
    subgraphs = make_subgraphs(ctrl, graph)

    ## Step 2: Embed core subgraph
    core_embed, relations_embed = embed(ctrl, core_graph, ctrl.embed_method)

    ## Step 3: Propagate embeddings
    embeddings = propagate_embeddings(
        ctrl, graph, core_embed, relations_embed, subgraphs
    )

    end = time()
    ctrl.total_time = end - start
    print(f"Total time: {ctrl.total_time:.2f} seconds")

    return embeddings, relations_embed.cpu(), ctrl.total_time
