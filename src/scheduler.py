# pylint: disable=too-few-public-methods, missing-function-docstring
"""Scheduler factory"""

from torch import optim


class DummyScheduler:
    """Dummy scheduler that does nothing"""

    def __init__(self):
        pass

    def step(self, metric):
        pass


def scheduler_factory(conf, optimizer, model_config):
    """Scheduler factory"""
    if conf.model in ["lstm", "mean_lstm", "transformer", "mean_transformer"]:
        scheduler = DummyScheduler()

    elif conf.model == "dice":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=model_config["scheduler"]["patience"],
            factor=model_config["scheduler"]["factor"],
            cooldown=0,
        )

    return scheduler
