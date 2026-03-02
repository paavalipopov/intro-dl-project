# pylint: disable=
"""Models for experiments and functions for setting them up"""

import torch
from src.models.lstm import LSTM, MeanLSTM
from src.models.transformer import Transformer, MeanTransformer
from src.models.dice import DICE


def model_factory(conf, model_config):
    """Models factory"""
    if conf.model == "lstm":
        model = LSTM(model_config)
    elif conf.model == "mean_lstm":
        model = MeanLSTM(model_config)
    elif conf.model == "transformer":
        model = Transformer(model_config)
    elif conf.model == "mean_transformer":
        model = MeanTransformer(model_config)
    elif conf.model == "dice":
        model = DICE(model_config)
    else:
        raise ValueError(f"{conf.model} is not recognized")

    if conf.mode == "introspection":
        # load trained weights
        checkpoint = torch.load(
            model_config["weights_path"], map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint)
        model.eval()

    return model
