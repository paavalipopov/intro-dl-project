# pylint: disable=invalid-name
""" Model config factory"""
import os
import json
import pandas as pd

from numpy.random import default_rng

from src.settings import LOGS_ROOT


def model_config_factory(conf, k=None, path=None):
    """Model config factory"""
    if conf.mode == "tune":
        model_config = get_tune_config(conf)
    elif conf.mode == "exp":
        model_config = get_best_config(conf, k, path)
    else:
        raise NotImplementedError

    return model_config


def get_tune_config(conf):
    """return random hyperparams for a model"""
    rng = default_rng()

    model_config = {}
    model_config["lr"] = 10 ** rng.uniform(-5, -3)

    # pick random hyperparameters

    if conf.model in ["lstm", "mean_lstm"]:
        model_config["hidden_size"] = rng.integers(32, 256)
        model_config["num_layers"] = rng.integers(1, 5)
        model_config["bidirectional"] = bool(rng.integers(0, 2))
        model_config["dropout"] = rng.uniform(0.1, 0.9)

        model_config["input_size"] = conf.data_info["data_shape"]["main"][2]
        model_config["output_size"] = conf.data_info["n_classes"]

    elif conf.model in ["transformer", "mean_transformer"]:
        model_config["head_hidden_size"] = rng.integers(4, 128)
        model_config["num_heads"] = rng.integers(1, 5)
        model_config["num_layers"] = rng.integers(1, 5)
        model_config["dropout"] = rng.uniform(0.1, 0.9)

        model_config["scaled"] = bool(rng.integers(0, 2))
        model_config["post"] = bool(rng.integers(0, 2))

        model_config["input_size"] = conf.data_info["data_shape"]["main"][2]
        model_config["output_size"] = conf.data_info["n_classes"]

    elif conf.model == "dice":
        # we might try to tune it, but it is not necessary
        raise NotImplementedError("DICE model does not require tuning")

    else:
        raise ValueError(f"{conf.model} model is not recognized")

    print("Tuning config:")
    print(model_config)

    return model_config


def get_best_config(conf, k, path):
    """return best hyperparams for a model, extracted authomatically or from the path"""
    assert k is not None or path is not None

    model_config = {}

    if k is not None:
        # find and load the best tuned model
        exp_dirs = []

        searched_dir = conf.project_name.split("-")
        searched_dir = "-".join(searched_dir[2:4])
        serached_dir = f"tune-{searched_dir}"
        if conf.prefix != conf.default_prefix:
            serached_dir = f"{conf.prefix}-{serached_dir}"
        print(f"Searching trained model in {LOGS_ROOT}/*{serached_dir}")
        for logdir in os.listdir(LOGS_ROOT):
            if logdir.endswith(serached_dir):
                exp_dirs.append(os.path.join(LOGS_ROOT, logdir))

        # if multiple run files found, choose the latest
        exp_dir = sorted(exp_dirs)[-1]
        print(f"Using best model from {exp_dir}")

        # get model config
        df = pd.read_csv(
            f"{exp_dir}/k_{k:02d}/runs.csv", delimiter=",", index_col=False
        )
        # pick hyperparams of a model with the highest test_score
        best_config_path = df.loc[df["score"].idxmax()].to_dict()
        best_config_path = best_config_path["path_to_config"]
        with open(best_config_path, "r", encoding="utf8") as fp:
            model_config = json.load(fp)

        print("Loaded model config:")
        print(model_config)

        return model_config

    if path is not None:
        raise NotImplementedError("loading best hyperparams is not implemented yet")

    raise ValueError(
        "Can't return best model hyperparams if neither `path` nor `k` is provided"
    )
