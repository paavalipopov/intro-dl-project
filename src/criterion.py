"""Criterion/objective function/loss function factory"""
from torch import nn


def criterion_factory(conf):
    """Criterion factory"""
    return nn.CrossEntropyLoss()
