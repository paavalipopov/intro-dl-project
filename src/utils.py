# pylint: disable=invalid-name
"""Auxilary functions"""
import json
import glob
import shutil

import argparse
from argparse import Namespace
from apto.utils.misc import boolean_flag

import torch
from torch import nn
import numpy as np

from src.settings import MODELS, DATASETS, WEIGHTS_ROOT, INTROSPECTION_ROOT


def get_argparser(sys_argv):
    """Get args parser"""
    resume = "resume" in sys_argv
    tune = "tune" in sys_argv

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "tune",
            "exp",
            "resume",
        ],
        required=True,
        help="'tune' for model hyperparams tuning; \
            'exp' for experiments with tuned model; \
                'resume' for resuming interrupted experiment",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=resume,
        help="Path to the interrupted experiment \
            (e.g., /Users/user/mlp_project/assets/logs/prefix-mode-model-ds), \
                used in 'resume' mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS,
        required=not resume,
        help="Name of the model to run",
    )
    parser.add_argument(
        "--ds",
        type=str,
        choices=DATASETS,
        required=not resume,
        help="Name of the dataset to use for training",
    )
    parser.add_argument(
        "--test-ds",
        nargs="*",
        type=str,
        choices=DATASETS,
        help="Additional datasets for testing",
    )

    # some datasets have multiple classes; set to true if you want to load all classes
    boolean_flag(parser, "multiclass", default=False)

    # whehter dataset should be z-scored over time
    boolean_flag(parser, "zscore", default=False)

    # whehter ICA components should be filtered
    boolean_flag(parser, "filter-indices", default=True)

    # if you want to obtain or use single optimal set of hyperparams,
    # pass --glob
    boolean_flag(parser, "glob", default=False)

    # if you want to keep checkpoints of trained models,
    # pass --preserve-checkpoints
    boolean_flag(parser, "preserve-checkpoints", default=not tune)

    parser.add_argument(
        "--prefix",
        type=str,
        help="Prefix for the project name (body of the project name \
            is '$mode-$model-$dataset'): default: UTC time",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=50 if tune else 10,
        help="Number of trials to run on each test fold \
            (default: 50 for 'tune', 10 for 'exp')",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of splits for StratifiedKFold (affects the number of test folds)",
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Max number of epochs (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (default: 30)",
    )

    return parser


def get_introspection_argparser():
    """Get args parser for introspection"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["introspection"],
        default="introspection",
        help="'introspection' for model introspection",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["saliency", "ig", "ignt"],
        default=["saliency", "ig", "ignt"],
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS,
        required=True,
        help="Name of the model to run",
    )
    parser.add_argument(
        "--ds",
        type=str,
        choices=DATASETS,
        required=True,
        help="Name of the dataset to use for training",
    )
    parser.add_argument(
        "--test-ds",
        nargs="*",
        type=str,
        choices=DATASETS,
        help="Additional datasets for testing",
    )

    # some datasets have multiple classes; set to true if you want to load all classes
    boolean_flag(parser, "multiclass", default=False)

    # whehter dataset should be z-scored over time
    boolean_flag(parser, "zscore", default=False)

    # whehter ICA components should be filtered
    boolean_flag(parser, "filter-indices", default=True)

    # if you want to obtain or use single optimal set of hyperparams,
    # pass --glob
    boolean_flag(parser, "glob", default=False)

    parser.add_argument(
        "--prefix",
        type=str,
        help="Prefix for the project name (body of the project name \
            is '$mode-$model-$dataset'): default: UTC time",
    )

    return parser


def get_resumed_params(conf):
    """Gather config and k/trial of interrupted experiment"""
    # load experiment config
    with open(f"{conf.path}/general_config.json", "r", encoding="utf8") as fp:
        config = json.load(fp)

    if config["mode"] == "tune":
        try:
            start_k = len(glob.glob(f"{conf.path}/k_*")) - 1
            last_fold_dir = sorted(glob.glob(f"{conf.path}/k_*"))[-1]
            try:
                with open(last_fold_dir + "/runs.csv", "r", encoding="utf8") as fp:
                    start_trial = len(fp.readlines()) - 1
            except FileNotFoundError:
                start_trial = 0
            faildir = last_fold_dir + f"/trial_{start_trial:04d}"
            print("Deleting interrupted run logs in " + faildir)
            try:
                shutil.rmtree(faildir)
            except FileNotFoundError:
                print("Could not delete interrupted run logs - FileNotFoundError")
        except IndexError:
            start_k, start_trial = 0, 0

    elif config["mode"] == "exp":
        with open(conf.path + "/runs.csv", "r", encoding="utf8") as fp:
            lines = len(fp.readlines()) - 1
            start_k = lines // config["n_trials"]
            start_trial = lines - start_k * config["n_trials"]
        faildir = conf.path + f"/k_{start_k:02d}/trial_{start_trial:04d}"
        print("Deleting interrupted run logs in " + faildir)
        try:
            shutil.rmtree(faildir)
        except FileNotFoundError:
            print("Could not delete interrupted run logs - FileNotFoundError")

    config = Namespace(**config)

    return config, start_k, start_trial


def get_introspection_params(conf):
    """Gather n_splits and n_trials of trained model for introspection"""
    # load experiment config
    with open(f"{conf.weights_dir}general_config.json", "r", encoding="utf8") as fp:
        config = json.load(fp)

    return config["n_splits"], config["n_trials"]


class EarlyStopping:
    """Early stops the training if the given score does not improve after a given patience."""

    def __init__(
        self,
        path: str,
        minimize: bool,
        patience: int = 30,
    ):
        assert minimize in [True, False]

        self.path = path
        self.minimize = minimize
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, new_score, model, epoch):
        if self.best_score is None:
            self.best_score = new_score
            self.save_checkpoint(model)
        else:
            if self.minimize:
                change = self.best_score - new_score
            else:
                change = new_score - self.best_score

            if change > 0.0:
                self.counter = 0
                self.best_score = new_score
                self.save_checkpoint(model)
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, model):
        # based on callback from animus package
        """Saves model if criterion is met"""
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model = model.module

        if issubclass(model.__class__, torch.nn.Module):
            torch.save(model.state_dict(), self.path + "best_model.pt")
        else:
            torch.save(model, self.path + "best_model.pt")


class NpEncoder(json.JSONEncoder):
    """Numpy types wrapper for JSON.dump"""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
