import torch
from pykeen.training import SLCWATrainingLoop
from pykeen.models import DistMult, TransE, RotatE, HolE

optimizers = {
    "SparseAdam": torch.optim.SparseAdam,
    "Adam": torch.optim.Adam,
}

models = {
    "hole": HolE,
    "distmult": DistMult,
    "transe": TransE,
    "rotate": RotatE,
}


def embed(ctrl, core_graph, model_name):
    model_kwargs = {
        "triples_factory": core_graph.triples_factory,
        "random_seed": ctrl.seed,
        "embedding_dim": ctrl.embed_dim,
    }

    model = models[model_name](**model_kwargs).to(ctrl.device)

    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=core_graph.triples_factory,
        optimizer=optimizers[ctrl.embed_setting.optimizer](
            params=model.get_grad_params(), lr=ctrl.embed_setting.lr
        ),
        negative_sampler_kwargs={
            "num_negs_per_pos": ctrl.num_negs_per_pos,
        },
    )

    losses = training_loop.train(
        triples_factory=core_graph.triples_factory,
        num_epochs=ctrl.num_epochs,
        batch_size=ctrl.batch_size,
    )

    ctrl.embed_setting.training_losses = [losses]

    return (
        model.entity_representations[0]().detach(),
        model.relation_representations[0]().detach(),
    )
