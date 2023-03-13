# pylint: disable=W0201,W0223,C0103,C0115,C0116,R0902,E1101,R0914
"""Main training script for simple models that output logits"""
import os
import sys
import argparse
import json
import shutil

import scipy.stats as stats

import pandas as pd
from apto.utils.misc import boolean_flag
from apto.utils.report import get_classification_report
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from tqdm.auto import tqdm

from sklearn.linear_model import LogisticRegression
from scipy import stats

from animus import IExperiment
import wandb

from src.settings import LOGS_ROOT, UTCNOW
from src.ts_data import load_dataset


class Experiment(IExperiment):
    """
    Animus-based training script. For more info see animus documentation
    (it's quite simple)
    """

    def __init__(
        self,
        mode: str,
        path: str,
        model: str,
        dataset: str,
        scaled: bool,
        test_datasets: list,
        prefix: str,
        n_splits: int,
        n_trials: int,
        max_epochs: int,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.config = {}

        self.utcnow = self.config["default_prefix"] = UTCNOW
        # starting fold/trial; used in resumed experiments
        self.start_k = 0
        self.start_trial = 0

        if mode == "resume":
            # get params of the resumed experiment, reset starting fold/trial
            (
                mode,
                model,
                dataset,
                scaled,
                test_datasets,
                prefix,
                n_splits,
                n_trials,
                max_epochs,
                batch_size,
            ) = self.acquire_params(path)

        self.mode = self.config["mode"] = mode  # tune or experiment mode
        self.model = self.config["model"] = model  # model name
        # main dataset name (used for training)
        self.dataset_name = self.config["dataset"] = dataset
        # if dataset should be scaled by sklearn's StandardScaler
        self.scaled = self.config["scaled"] = scaled

        if test_datasets is None:  # additional test datasets
            self.test_datasets = []
        else:
            if self.mode == "tune":
                print("'Tune' mode overrides additional test datasets")
                self.test_datasets = []
            else:
                self.test_datasets = test_datasets

        if self.dataset_name in self.test_datasets:
            # Fraction of the main dataset is always used as a test dataset;
            # no need for it in the list of test datasets
            print(
                f"Received main dataset {self.dataset_name} among test datasets {self.test_datasets}; removed"
            )
            self.test_datasets.remove(self.dataset_name)
        self.config["test_datasets"] = self.test_datasets

        # num of splits for StratifiedKFold
        self.n_splits = self.config["n_splits"] = n_splits
        # num of trials for each fold
        self.n_trials = self.config["n_trials"] = n_trials
        self.max_epochs = self.config["max_epochs"] = max_epochs
        self.batch_size = self.config["batch_size"] = batch_size

        # set project name prefix
        if len(prefix) == 0:
            self.project_prefix = f"{self.utcnow}"
        else:
            # '-'s are reserved for name parsing
            self.project_prefix = prefix.replace("-", "_")
        self.config["prefix"] = self.project_prefix

        self.project_name = f"{self.mode}-{self.model}-{self.dataset_name}"
        if self.scaled:
            self.project_name = f"{self.mode}-{self.model}-scaled_{self.dataset_name}"
        if len(self.test_datasets) != 0:
            project_ending = "-tests-" + "_".join(self.test_datasets)
            self.project_name += project_ending
        self.config["project_name"] = self.project_name

        self.logdir = f"{LOGS_ROOT}/{self.project_prefix}-{self.project_name}/"
        self.config["logdir"] = self.logdir

        # create experiment directory
        os.makedirs(self.logdir, exist_ok=True)
        # save initial config
        logfile = f"{self.logdir}/config.json"
        with open(logfile, "w") as fp:
            json.dump(self.config, fp)

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        print(f"Used device: {dev}")
        self.device = torch.device(dev)

        self.initialize_data(self.dataset_name)

        self.test_dataloaders = {}
        for ds in self.test_datasets:
            self.test_dataloaders[ds] = self.initialize_data(ds, for_test=True)

    def acquire_params(self, path):
        """
        Used for extracting experiment set-up from the
        given path for resuming an interrupted experiment
        """
        # load experiment config
        with open(f"{path}/config.json", "r") as fp:
            config = json.load(fp)

        # find when the experiment got interrupted
        with open(path + "/runs.csv", "r") as fp:
            lines = len(fp.readlines()) - 1
            self.start_k = lines // config["n_trials"]
            self.start_trial = lines - self.start_k * config["n_trials"]

        # delete failed run
        faildir = path + f"/k_{self.start_k}/{self.start_trial:04d}"
        print("Deleting interrupted run logs in " + faildir)
        try:
            shutil.rmtree(faildir)
        except FileNotFoundError:
            print("Could not delete interrupted run logs - FileNotFoundError")

        # reset the default prefix
        self.utcnow = self.config["default_prefix"] = config["default_prefix"]

        return (
            config["mode"],
            config["model"],
            config["dataset"],
            config["scaled"],
            config["test_datasets"],
            config["prefix"],
            config["n_splits"],
            config["n_trials"],
            config["max_epochs"],
            config["batch_size"],
        )

    def initialize_data(self, dataset, for_test=False):
        # load core dataset (or additional datasets if for_test==True)
        # your dataset should have shape [n_features; n_channels; time_len]

        features, labels = load_dataset(dataset)

        if self.scaled:
            # z-score over time
            features = stats.zscore(features, axis=2)

        features = np.nan_to_num(features)
        print(f"Feature size before z-score: {features.shape}")
        for t in range(features.shape[0]):
            for r in range(features.shape[1]):
                features[t, r, :] = stats.zscore(features[t, r, :])
        features = np.nan_to_num(features)
        print(f"Feature size after z-score: {features.shape}")

        pearson = np.zeros(
            (
                features.shape[0],
                features.shape[1],
                features.shape[1],
            )
        )
        print(f"Pearson size: {pearson.shape}")

        for i in range(features.shape[0]):
            pearson[i, :, :] = np.corrcoef(features[i, :, :])

        pearson = np.nan_to_num(pearson)

        tril_inx = np.tril_indices(pearson.shape[1])

        features = np.zeros((pearson.shape[0], tril_inx[0].shape[0]))
        print(f"Feature size after extracting triangle: {features.shape}")

        for t in range(features.shape[0]):
            features[t] = pearson[t][tril_inx]

        if for_test:
            return features, labels

        self.features = features
        self.labels = labels

        self.data_shape = features.shape  # [n_features; n_channels;]
        self.n_classes = np.unique(labels).shape[0]
        print("data shape: ", self.data_shape)

    def initialize_dataset(self):
        # train-val/test split
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        skf.get_n_splits(self.features, self.labels)

        train_index, test_index = list(skf.split(self.features, self.labels))[self.k]

        X_train, X_test = self.features[train_index], self.features[test_index]
        y_train, y_test = self.labels[train_index], self.labels[test_index]

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)

        # create run's path directory
        self.config["runpath"] = f"{self.logdir}k_{self.k}/{self.trial:04d}"
        self.config["run_config_path"] = f"{self.config['runpath']}/config.json"
        os.makedirs(self.config["runpath"], exist_ok=True)

        # init wandb logger
        self.wandb_logger: wandb.run = wandb.init(
            project=f"{self.project_prefix}-{self.project_name}",
            name=f"k_{self.k}-trial_{self.trial}",
            save_code=True,
        )

        # init data
        self.initialize_dataset()

        self.num_epochs = self.max_epochs

        self.wandb_logger.config.update(self.config)

        # update saved config
        with open(f"{self.logdir}/config.json", "w") as fp:
            json.dump(self.config, fp)

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)

        clf = LogisticRegression(random_state=self.trial).fit(self.X_train, self.y_train)

        y_score = clf.predict_proba(self.X_test)
        y_pred = np.argmax(y_score, axis=-1).astype(np.int32)

        report = get_classification_report(
            y_true=self.y_test, y_pred=y_pred, y_score=y_score, beta=0.5
        )

        self.dataset_metrics = {
            "accuracy": report["precision"].loc["accuracy"],
            "score": report["auc"].loc["weighted"],
        }

        print("Test results:")
        print("Accuracy ", self.dataset_metrics["accuracy"])
        print("AUC ", self.dataset_metrics["score"])
        results = {
            "test_accuracy": self.dataset_metrics["accuracy"],
            "test_score": self.dataset_metrics["score"],
        }

        for ds in self.test_dataloaders:
            y_score = clf.predict_proba(self.test_dataloaders[ds][0])
            y_pred = np.argmax(y_score, axis=-1).astype(np.int32)

            report = get_classification_report(
                y_true=self.test_dataloaders[ds][1],
                y_pred=y_pred,
                y_score=y_score,
                beta=0.5,
            )

            results[f"{ds}_test_accuracy"] = report["precision"].loc["accuracy"]
            results[f"{ds}_test_score"] = report["auc"].loc["weighted"]

        df = pd.DataFrame(results, index=[0])
        with open(f"{self.logdir}/runs.csv", "a") as f:
            df.to_csv(f, header=f.tell() == 0, index=False)
        self.wandb_logger.log(results)
        self.wandb_logger.finish()

    def start(self):
        # get experiment folds-trial pairs
        # if mode is 'resume', then it won't have already completed folds
        # else it is just all folds-trial pairs from (0, 0) to (n_splits, n_trials)
        folds_of_interest = []

        if self.start_k < self.n_splits:
            # interrupted fold
            for trial in range(self.start_trial, self.n_trials):
                folds_of_interest += [(self.start_k, trial)]

            # the rest of folds
            for k in range(self.start_k + 1, self.n_splits):
                for trial in range(self.n_trials):
                    folds_of_interest += [(k, trial)]
        else:
            raise IndexError

        # run through folds
        for k, trial in folds_of_interest:
            self.k = k  # k'th test fold
            self.trial = trial  # trial'th trial on the k'th fold
            self.run()


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    for_continue = "resume" in sys.argv

    datasets = [
        "oasis",
        "adni",
        "abide",
        "abide_869",
        "abide_roi",
        "fbirn",
        "fbirn_100",
        "fbirn_200",
        "fbirn_400",
        "fbirn_1000",
        "cobre",
        "bsnip",
        "hcp",
        "hcp_roi",
        "ukb",
        "ukb_age_bins",
        "time_fbirn",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "experiment",
            "resume",
        ],
        default="experiment",
        help="'tune' for model hyperparams tuning; \
            'experiment' for experiments with tuned model; \
                'resume' for resuming interrupted experiment",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=for_continue,
        help="Path to the interrupted experiment \
            (e.g., /Users/user/mlp_project/assets/logs/prefix-mode-model-ds), \
                used in 'resume' mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "lr",
        ],
        default="lr",
        help="Name of the model to run",
    )
    parser.add_argument(
        "--ds",
        type=str,
        choices=datasets,
        required=not for_continue,
        help="Name of the dataset to use for training",
    )

    parser.add_argument(
        "--test-ds",
        nargs="*",
        type=str,
        choices=datasets,
        help="Additional datasets for testing",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for the project name (body of the project name \
            is '$mode-$model-$dataset'): default: UTC time",
    )

    # whehter dataset should be scaled by sklearn's StandardScaler
    boolean_flag(parser, "scaled", default=False)

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=30,
        help="Max number of epochs (min 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=10,
        help="Number of trials to run on each test fold",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=5,
        help="Number of splits for StratifiedKFold (affects the number of test folds)",
    )
    args = parser.parse_args()

    Experiment(
        mode=args.mode,
        path=args.path,
        model=args.model,
        dataset=args.ds,
        scaled=args.scaled,
        test_datasets=args.test_ds,
        prefix=args.prefix,
        n_splits=args.num_splits,
        n_trials=args.num_trials,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
    ).start()


# for dataset in fbirn fbirn_200 bsnip cobre abide_869 abide_roi oasis hcp hcp_roi time_fbirn
# do
#     PYTHONPATH=./ python src/scripts/fnc_lr_experiments.py --mode experiment --ds $dataset --max-epochs 200 --num-trials 10  --prefix lr_fnc
# done
