# pylint: disable=too-few-public-methods, no-member, unused-argument
"""Criterion/objective function/loss function factory"""

import torch
from torch import nn


def criterion_factory(conf, model_config):
    """Criterion factory"""
    if conf.model in ["lstm", "mean_lstm", "transformer", "mean_transformer"]:
        criterion = CEloss(model_config)
    elif conf.model == "dice":
        criterion = DICEregCEloss(model_config)
    else:
        raise ValueError(f"{conf.model} is not recognized")

    return criterion


class CEloss:
    """Basic Cross-entropy loss"""

    def __init__(self, model_config):
        self.ce_loss = nn.CrossEntropyLoss()

    def __call__(self, logits, target, model, device):
        ce_loss = self.ce_loss(logits, target)

        return ce_loss


class DICEregCEloss:
    """Cross-entropy loss with model regularization"""

    def __init__(self, model_config):
        self.ce_loss = nn.CrossEntropyLoss()

        self.reg_param = model_config["reg_param"]

    def __call__(self, logits, target, model, device):
        ce_loss = self.ce_loss(logits, target)

        reg_loss = torch.zeros(1).to(device)

        for name, param in model.gta_embed.named_parameters():
            if "bias" not in name:
                reg_loss += self.reg_param * torch.norm(param, p=1)

        for name, param in model.gta_attend.named_parameters():
            if "bias" not in name:
                reg_loss += self.reg_param * torch.norm(param, p=1)

        loss = ce_loss + reg_loss
        return loss
