"""Constants of the project"""
from datetime import datetime
import os

import path

PROJECT_ROOT = path.Path(os.path.dirname(__file__)).joinpath("..").abspath()
ASSETS_ROOT = PROJECT_ROOT.joinpath("assets")

DATA_ROOT = ASSETS_ROOT.joinpath("data")
LOGS_ROOT = ASSETS_ROOT.joinpath("logs")
WEIGHTS_ROOT = ASSETS_ROOT.joinpath("trained_models")
HYPERPARAMS_ROOT = ASSETS_ROOT.joinpath("model_hyperparams")
INTROSPECTION_ROOT = ASSETS_ROOT.joinpath("introspection")

UTCNOW = datetime.utcnow().strftime("%y%m%d.%H%M%S")

MODELS = [
    "lstm",
    "mean_lstm",
    "transformer",
    "mean_transformer",
    "dice",
]
DATASETS = [
    "abide",
    "cobre",
    "synth1",
    "synth2",
]
