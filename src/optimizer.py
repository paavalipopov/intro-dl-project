"""Optimizer factory"""
from torch import optim


def optimizer_factory(conf, model, model_config):
    """Optimizer factory"""
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(model_config["lr"]),
    )

    return optimizer
