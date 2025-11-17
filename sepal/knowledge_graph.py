import torch
import scipy.sparse
import numpy as np


# A class to represent multi relational graphs
class KnowledgeGraph(object):
    def __init__(self, triples_factory, name) -> None:
        self.name = name
        ## Useful attributes for base embedding
        self.triples_factory = triples_factory

        self.num_entities = triples_factory.num_entities
        self.num_triples = triples_factory.num_triples
        self.num_relations = triples_factory.num_relations

        ## Useful attributes for subgraph generation
        self.core_subgraph_idx = None
        edge_index, _ = self.get_edge_index_and_type()
        self.adjacency = scipy.sparse.csr_matrix(
            (torch.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
            shape=(self.num_entities, self.num_entities),
            dtype="float32",
        )
        self.adjacency.data = np.ones(self.adjacency.data.shape[0], dtype="float32")
        self.degrees = self.adjacency @ np.ones(self.num_entities)

    def get_mapped_triples(self):
        return self.triples_factory._add_inverse_triples_if_necessary(
            self.triples_factory.mapped_triples
        )

    def get_edge_index_and_type(self):
        mapped_triples = self.get_mapped_triples()
        edge_index = mapped_triples[:, 0::2].t()
        edge_type = mapped_triples[:, 1]
        return edge_index, edge_type

    def get_subgraph_edge_index_and_type(self, subgraph_idx):
        # Get mapped triples
        mapped_triples = self.get_mapped_triples()
        # Mask subgraph nodes
        mapped_triples = mapped_triples[
            np.isin(mapped_triples[:, 0::2], subgraph_idx).all(axis=1)
        ]
        # Reindex entities
        d = {subgraph_idx[i]: i for i in range(len(subgraph_idx))}
        mapped_triples[:, [0, 2]] = torch.tensor(
            np.vectorize(d.__getitem__)(mapped_triples[:, [0, 2]])
        )
        # Create edge_index and edge_type
        edge_index = mapped_triples[:, 0::2].t()
        edge_type = mapped_triples[:, 1]
        return edge_index, edge_type

    def get_adjacency_with_self_loops(self):
        return self.adjacency + scipy.sparse.diags(
            np.ones(self.num_entities), dtype="float32"
        )
