from torch import nn


def criterion_factory(conf):
    return nn.CrossEntropyLoss()
