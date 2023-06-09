# pylint: disable=invalid-name
""" Model config factory"""
import os
import json
import pandas as pd

from numpy.random import default_rng

from src.settings import LOGS_ROOT, HYPERPARAMS_ROOT


def model_config_factory(conf, k=None):
    """Model config factory"""
    if conf.mode == "tune":
        model_config = get_tune_config(conf)
    elif conf.mode == "exp":
        model_config = get_best_config(conf, k)
    elif conf.mode == "introspection":
        model_config = get_introspection_config(conf, k)
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
        model_config["lstm"] = {}
        model_config["lstm"]["bidirectional"] = bool(rng.integers(0, 2))
        model_config["lstm"]["num_layers"] = rng.integers(1, 4)
        model_config["lstm"]["hidden_size"] = rng.integers(20, 60)

        model_config["clf"] = {}
        model_config["clf"]["hidden_size"] = rng.integers(16, 128)
        model_config["clf"]["num_layers"] = rng.integers(0, 3)

        model_config["MHAtt"] = {}
        model_config["MHAtt"]["n_heads"] = rng.integers(1, 4)
        model_config["MHAtt"]["head_hidden_size"] = rng.integers(16, 64)
        model_config["MHAtt"]["dropout"] = rng.uniform(0.1, 0.9)

        model_config["scheduler"] = {}
        model_config["scheduler"]["patience"] = rng.integers(1, conf.patience // 2)
        model_config["scheduler"]["factor"] = rng.uniform(0.1, 0.8)

        model_config["reg_param"] = 10 ** rng.uniform(-10, 0)

        model_config["input_size"] = conf.data_info["data_shape"]["main"][2]
        model_config["output_size"] = conf.data_info["n_classes"]

    else:
        raise ValueError(f"{conf.model} model is not recognized")

    print("Tuning config:")
    print(model_config)

    return model_config


def get_best_config(conf, k):
    """return best hyperparams for a model, extracted authomatically or from the path"""
    if conf.glob:
        with open(f"{HYPERPARAMS_ROOT}/{conf.model}.json", "r", encoding="utf8") as fp:
            model_config = json.load(fp)
            model_config["input_size"] = conf.data_info["data_shape"]["main"][2]
            model_config["output_size"] = conf.data_info["n_classes"]

    else:
        assert k is not None

        model_config = {}

        # find and load the best tuned model
        exp_dirs = []

        searched_dir = conf.project_name.split("-")
        searched_dir = "-".join(searched_dir[2:4])
        searched_dir = f"tune-{searched_dir}"
        if conf.prefix != conf.default_prefix:
            searched_dir = f"{conf.prefix}-{searched_dir}"
        print(f"Searching trained model in {LOGS_ROOT}/*{searched_dir}")
        for logdir in os.listdir(LOGS_ROOT):
            if logdir.endswith(searched_dir):
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


def get_introspection_config(conf, k):
    """return hyperparams for model introspection for test_fold=k"""
    assert k is not None

    model_config = {}
    # get best config of the k`th fold`
    df = pd.read_csv(f"{conf.weights_dir}/runs.csv", delimiter=",", index_col=False)
    # pick hyperparams of a model with the highest test_score
    best_trial = df["test_score"][k * conf.n_trials : (k + 1) * conf.n_trials].idxmax()
    best_trial = best_trial - k * conf.n_trials

    best_config_dir = f"{conf.weights_dir}/k_{k:02d}/trial_{best_trial:04d}/"
    with open(f"{best_config_dir}model_config.json", "r", encoding="utf8") as fp:
        model_config = json.load(fp)

    model_config["weights_path"] = f"{best_config_dir}best_model.pt"

    return model_config
