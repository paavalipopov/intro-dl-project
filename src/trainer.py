# pylint: disable=W0201,W0223,C0103,C0115,C0116,R0902,E1101,R0914
"""Main training script for simple models that output logits"""
import os
import glob
import sys
import argparse
import json
import shutil
import time

import pandas as pd
import numpy as np
import scipy.stats as stats

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from animus import EarlyStoppingCallback, IExperiment
from animus.torch.callbacks import TorchCheckpointerCallback

from apto.utils.misc import boolean_flag
from apto.utils.report import get_classification_report

import wandb

from src.settings import LOGS_ROOT, UTCNOW
from src.ts_data import load_dataset


def trainer_factory():
    pass


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
        test_datasets: list,
        multiclass: bool,
        scaled: bool,
        prefix: str,
        n_splits: int,
        n_trials: int,
        max_epochs: int,
        batch_size: int,
        patience: int,
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
                test_datasets,
                multiclass,
                scaled,
                prefix,
                n_splits,
                n_trials,
            ) = self.get_resumed_params(path)

        self.mode = self.config["mode"] = mode  # tune or experiment mode
        self.model = self.config["model"] = model  # model name
        # main dataset name (used for training)
        self.dataset_name = self.config["dataset"] = dataset
        if test_datasets is None:  # additional test datasets
            self.test_datasets = []
        else:
            if self.mode == "tune":
                print("'Tune' mode overrides additional test datasets")
                self.test_datasets = []
            else:
                self.test_datasets = test_datasets
        self.config["test_datasets"] = self.test_datasets

        # some datasets have multiple classes; set to true if you want to load all classes
        self.multiclass = self.config["multiclass"] = multiclass
        # if dataset should be z-scored over time
        self.scaled = self.config["scaled"] = scaled

        # num of splits for StratifiedKFold
        self.n_splits = self.config["n_splits"] = n_splits
        # num of trials for each fold
        self.n_trials = self.config["n_trials"] = n_trials

        self.max_epochs = self.config["max_epochs"] = max_epochs
        self.batch_size = self.config["batch_size"] = batch_size
        self.patience = self.config["patience"] = patience

        # set project name prefix (best model config in 'experiment' mode
        # will be extracted from the tuning directory with matching prefix, unless default)
        if prefix is None:
            self.project_prefix = f"{self.utcnow}"
        else:
            if len(prefix) == 0:
                self.project_prefix = f"{self.utcnow}"
            else:
                # '-'s are reserved for project name parsing
                self.project_prefix = prefix.replace("-", "_")
        self.config["prefix"] = self.project_prefix

        # set project name
        proj_dataset_name = self.dataset_name
        if self.multiclass:
            proj_dataset_name = f"multiclass_{proj_dataset_name}"
        if self.scaled:
            proj_dataset_name = f"scaled_{proj_dataset_name}"
        self.project_name = f"{self.mode}-{self.model}-{proj_dataset_name}"
        if len(self.test_datasets) != 0:
            project_ending = "-test_ds_" + "_".join(self.test_datasets)
            self.project_name += project_ending
        self.config["project_name"] = self.project_name

        self.logdir = f"{LOGS_ROOT}/{self.project_prefix}-{self.project_name}"
        self.config["logdir"] = self.logdir

        # create experiment directory
        os.makedirs(self.logdir, exist_ok=True)

        # set device
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        print(f"Used device: {dev}")
        self.config["device"] = dev
        self.device = torch.device(dev)

        # load raw data
        self.get_data(self.dataset_name)

        # save initial config
        logfile = f"{self.logdir}/general_config.json"
        with open(logfile, "w") as fp:
            json.dump(self.config, fp, indent=2)

    def get_resumed_params(self, path):
        """
        Used for extracting experiment set-up from the
        given path for resuming an interrupted experiment
        """
        # load experiment config
        with open(f"{path}/general_config.json", "r") as fp:
            config = json.load(fp)

        if config["mode"] == "tune":
            with open(path + "/runs.csv", "r") as fp:
                self.start_trial = len(fp.readlines()) - 1
            faildir = path + f"/trial_{self.start_trial:04d}"
            print("Deleting interrupted run logs in " + faildir)
            try:
                shutil.rmtree(faildir)
            except FileNotFoundError:
                print("Could not delete interrupted run logs - FileNotFoundError")
        elif config["mode"] == "experiment":
            with open(path + "/runs.csv", "r") as fp:
                lines = len(fp.readlines()) - 1
                self.start_k = lines // config["n_trials"]
                self.start_trial = lines - self.start_k * config["n_trials"]
            faildir = path + f"/k_{self.start_k:02d}/trial_{self.start_trial:04d}"
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
            config["test_datasets"],
            config["multiclass"],
            config["scaled"],
            config["prefix"],
            config["n_splits"],
            config["n_trials"],
        )

    def get_data(self, dataset, for_test=False):
        # features shape should be [n_features; n_channels; time_len]
        features, labels = load_dataset(dataset, self.multiclass)

        # get rid of invalid data (NaNs, infs)
        features[features != features] = 0

        if self.scaled:
            # z-score over time
            features = stats.zscore(features, axis=2)

        features = np.swapaxes(features, 1, 2)  # [n_features; time_len; n_channels]

        if for_test:
            # if dataset is loaded for tests, it should not be
            # split into train/val/test.
            # it is called in `on_experiment_end` when testing time comes

            return TensorDataset(
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.int64),
            )

        self.data_shape = features.shape  # [n_features; time_len; n_channels;]
        self.n_classes = np.unique(labels).shape[0]
        self.config["data_shape"] = self.data_shape
        self.config["n_classes"] = self.n_classes

        print("data shape: ", self.data_shape)

        # generate CV folds
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        self.CV_folds = list(skf.split(features, labels))

        self.features = features
        self.labels = labels

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)

        # create run's path directory
        if self.mode == "tune":
            wandb_run_name = f"trial_{self.trial:04d}-k_{self.k:02d}"
            self.runpath = f"{self.logdir}/trial_{self.trial:04d}/k_{self.k:02d}"

            self.cv_results_path = f"{self.logdir}/trial_{self.trial:04d}/runs.csv"
        elif self.mode == "experiment":
            wandb_run_name = f"k_{self.k:02d}-trial_{self.trial:04d}"
            self.runpath = f"{self.logdir}/k_{self.k:02d}/{self.trial:04d}"

        self.run_config_path = f"{self.runpath}/model_config.json"
        os.makedirs(self.runpath, exist_ok=True)

        # init wandb logger
        self.wandb_logger: wandb.run = wandb.init(
            project=f"{self.project_prefix}-{self.project_name}",
            name=wandb_run_name,
            save_code=True,
        )
        if self.mode == "tune":
            self.model_config["link"] = exp.wandb_logger.get_url()

        # init dataset
        self.initialize_dataset()

        # init model
        self.initialize_model()

        # setup data loaders
        self.datasets = {
            "train": DataLoader(
                self._train_ds,
                batch_size=self.batch_size,
                num_workers=0,
                shuffle=True,
            ),
            "valid": DataLoader(
                self._valid_ds,
                batch_size=self.batch_size,
                num_workers=0,
                shuffle=False,
            ),
        }

        # setup callbacks
        self.callbacks = {
            "early-stop": EarlyStoppingCallback(
                minimize=True,
                patience=self.patience,
                dataset_key="valid",
                metric_key="loss",
                min_delta=0.001,
            ),
            "checkpointer": TorchCheckpointerCallback(
                exp_attr="_model",
                logdir=self.runpath,
                dataset_key="valid",
                metric_key="loss",
                minimize=True,
            ),
        }

        self.num_epochs = self.max_epochs

        self.wandb_logger.config.update({"general": self.config})
        self.wandb_logger.config.update({"model": self.model_config})

        # save model params in the run's directory
        with open(self.run_config_path, "w") as fp:
            json.dump(self.model_config, fp, indent=2)

    def initialize_dataset(self):
        if self.model == "stdim":
            # self.data_shape: [n_features; time_len; n_channels;]
            # reshape data's time dimension into windows:
            subjects = self.data_shape[0]  # subjects
            tc = self.data_shape[1]  # original time length
            # window x dim, or channels (equals to encoder input_channels)
            sample_x = self.data_shape[2]
            # window y dim, or window time
            sample_y = self.model_config["datashape"]["window_size"]
            # windows shift - how much windows overlap
            window_shift = self.model_config["datashape"]["window_shift"]
            # number of windows, or new time
            samples_per_subject = tc // window_shift - (sample_y // window_shift + 1)

            features = np.zeros((subjects, samples_per_subject, sample_x, sample_y))
            for i in range(subjects):
                for j in range(samples_per_subject):
                    features[i, j, :, :] = self.features[
                        i, (j * window_shift) : (j * window_shift) + sample_y, :
                    ].swapaxes(0, 1)
        else:
            features = self.features

        train_index, test_index = self.CV_folds[self.k]

        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = self.labels[train_index], self.labels[test_index]

        # train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.data_shape[0] // self.n_splits,
            random_state=42 + self.trial if self.mode == "experiment" else 42,
            stratify=y_train,
        )

        if self.model == "stdim":
            # stdim encoder requires data that's prepared differently
            self.encoder_dataset = {"train": X_train.copy(), "val": X_val.copy()}

            # reshape data for stdim probes (last time)
            y_train = np.kron(y_train, np.ones((1, X_train.shape[1]))).squeeze()
            X_train = X_train.reshape(-1, X_train.shape[2], X_train.shape[3])

            y_val = np.kron(y_val, np.ones((1, X_val.shape[1]))).squeeze()
            X_val = X_val.reshape(-1, X_val.shape[2], X_val.shape[3])

            y_test = np.kron(y_test, np.ones((1, X_test.shape[1]))).squeeze()
            X_test = X_test.reshape(-1, X_test.shape[2], X_test.shape[3])

        self._train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.int64),
        )
        self._valid_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.int64),
        )
        self._test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.int64),
        )

    def initialize_model(self):
        # deferred import to counter cyclic import error
        from src.ts_model import (
            get_model,
            get_criterion,
            get_optimizer,
        )

        self._model = get_model(self, self.model, self.model_config)
        self._model = self._model.to(self.device)

        self.criterion = get_criterion(self.model)

        self.optimizer = get_optimizer(self, self.model_config)

    def run_dataset(self, thr_tune: bool = False):
        all_scores, all_targets = [], []
        total_loss = 0.0
        self._model.train(self.is_train_dataset)

        with torch.set_grad_enabled(self.is_train_dataset):
            for self.dataset_batch_step, (data, target) in enumerate(
                tqdm(self.dataset)
            ):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                logits = self._model(data)
                loss = self.criterion(logits, target)
                score = torch.softmax(logits, dim=-1)

                all_scores.append(score.cpu().detach().numpy())
                all_targets.append(target.cpu().detach().numpy())
                total_loss += loss.sum().item()
                if self.is_train_dataset:
                    loss.backward()
                    self.optimizer.step()

        total_loss /= self.dataset_batch_step + 1

        y_test = np.hstack(all_targets)
        y_score = np.vstack(all_scores)
        if thr_tune is True:
            return y_test, y_score

        if self.dataset_key == "test":
            y_score_tuned = y_score - self.best_threshold
            y_pred = np.argmax(y_score_tuned, axis=-1).astype(np.int32)
        else:
            y_pred = np.argmax(y_score, axis=-1).astype(np.int32)

        report = get_classification_report(
            y_true=y_test, y_pred=y_pred, y_score=y_score, beta=0.5
        )

        self.dataset_metrics = {
            "accuracy": report["precision"].loc["accuracy"],
            "score": report["auc"].loc["weighted"],
            "loss": total_loss,
        }

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(self)
        self.wandb_logger.log(
            {
                "train_accuracy": self.epoch_metrics["train"]["accuracy"],
                "train_score": self.epoch_metrics["train"]["score"],
                "train_loss": self.epoch_metrics["train"]["loss"],
                "valid_accuracy": self.epoch_metrics["valid"]["accuracy"],
                "valid_score": self.epoch_metrics["valid"]["score"],
                "valid_loss": self.epoch_metrics["valid"]["loss"],
            },
        )

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)

        print("Run threshold tuning")
        # load best weights
        model_logpath = f"{self.runpath}/_model.best.pth"
        checkpoint = torch.load(
            model_logpath, map_location=lambda storage, loc: storage
        )
        self._model.load_state_dict(checkpoint)
        self._model = self._model.to(self.device)

        self.dataset_key = "valid"
        self.dataset = self.datasets["valid"]

        val_y_test, val_y_score = self.run_dataset(thr_tune=True)

        best_acc = 0.0
        self.best_threshold = []
        for thr in self.thr_gen(self.n_classes, 1.0, []):
            thr = np.array(thr)
            new_y_score = val_y_score - thr

            new_y_pred = np.argmax(new_y_score, axis=-1).astype(np.int32)
            acc = accuracy_score(val_y_test, new_y_pred)
            if acc > best_acc:
                best_acc = acc
                self.best_threshold = [thr]
            elif acc == best_acc:
                self.best_threshold += [thr]

        self.best_threshold = np.mean(np.stack(self.best_threshold), axis=0)

        print("Run test dataset")
        # load best weights
        model_logpath = f"{self.runpath}/_model.best.pth"
        checkpoint = torch.load(
            model_logpath, map_location=lambda storage, loc: storage
        )
        self._model.load_state_dict(checkpoint)
        self._model = self._model.to(self.device)

        self.dataset_key = "test"
        self.dataset = DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )
        self.run_dataset()

        print("Test results:")
        print("Accuracy ", self.dataset_metrics["accuracy"])
        print("AUC ", self.dataset_metrics["score"])
        print("Loss ", self.dataset_metrics["loss"])
        results = {
            "test_accuracy": self.dataset_metrics["accuracy"],
            "test_score": self.dataset_metrics["score"],
            "test_loss": self.dataset_metrics["loss"],
        }

        if len(self.test_datasets) != 0:
            print("Run additional test datasets")
        for dataset_name in self.test_datasets:
            test_ds = self.get_data(dataset_name, for_test=True)
            self.dataset = DataLoader(
                test_ds,
                batch_size=self.batch_size,
                num_workers=0,
                shuffle=False,
            )
            self.run_dataset()

            print(f"On {dataset_name}:")
            print("Accuracy ", self.dataset_metrics["accuracy"])
            print("AUC ", self.dataset_metrics["score"])
            print("Loss ", self.dataset_metrics["loss"])

            results[f"{dataset_name}_test_accuracy"] = self.dataset_metrics["accuracy"]
            results[f"{dataset_name}_test_score"] = self.dataset_metrics["score"]
            results[f"{dataset_name}_test_loss"] = self.dataset_metrics["loss"]

        # save run results in csv file and log to WandB
        df = pd.DataFrame(results, index=[0])
        if self.mode == "tune":
            with open(self.cv_results_path, "a") as f:
                df.to_csv(f, header=f.tell() == 0, index=False)
        elif self.mode == "experiment":
            with open(f"{self.logdir}/runs.csv", "a") as f:
                df.to_csv(f, header=f.tell() == 0, index=False)
        self.wandb_logger.log(results)
        self.wandb_logger.finish()

        # delete checkpointers:
        for file in glob.glob(f"{self.runpath}/_model.*.pth"):
            os.remove(file)

    def thr_gen(self, depth, summ, constr_threshold):
        if depth == 1:
            yield constr_threshold + [summ]
            return
        for thr in np.arange(0.0, summ + 0.0001, 0.001):
            rest_sum = summ - thr
            if rest_sum >= 0.0:
                new_constr_threshold = constr_threshold + [thr]
                yield from self.thr_gen(depth - 1, rest_sum, new_constr_threshold)

    def start(self):
        # deferred import to counter cyclic import error
        from src.ts_config import get_tune_config, get_best_config

        # self.start_trial and self.start_k are set to 0 in the initialization,
        # and are reseted accordingly if the initial mode was 'resume'

        if self.mode == "tune":
            # in 'tune' mode for each trial we are running cross-validated experiments with
            # the same tuning config
            for trial in range(self.start_trial, self.n_trials):
                self.trial = trial
                self.model_config = get_tune_config(self, random_seed=int(time.time()))
                for k in range(self.n_splits):
                    self.k = k
                    self.run()

                # read CV results, save the average
                df = pd.read_csv(self.cv_results_path)
                auc = df["test_score"].to_numpy()
                auc = np.mean(auc)
                df = pd.DataFrame(
                    {
                        "trial": trial,
                        "score": auc,
                        "path_to_config": self.run_config_path,
                    },
                    index=[0],
                )
                with open(f"{self.logdir}/runs.csv", "a") as f:
                    df.to_csv(f, header=f.tell() == 0, index=False)

        elif self.mode == "experiment":
            # in 'experiment' mode we run cross-validated expriments; for each test fold
            # we are splitting train dataset into train/val datasets with different random seeds
            # (determined by 'trial')
            self.model_config = get_best_config(self)

            self.k = self.start_k
            for trial in range(self.start_trial, self.n_trials):
                self.trial = trial
                self.run()
            for k in range(self.start_k + 1, self.n_splits):
                self.k = k
                for trial in range(self.n_trials):
                    self.trial = trial
                    self.run()

        else:
            raise NotImplementedError()


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    models = [
        "mlp",
        "wide_mlp",
        "deep_mlp",
        "attention_mlp",
        "new_attention_mlp",
        "meta_mlp",
        "pe_mlp",
        "lstm",
        "noah_lstm",
        "transformer",
        "mean_transformer",
        "pe_transformer",
        "stdim",
    ]
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

    for_resume = "resume" in sys.argv
    tune = "tune" in sys.argv

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "tune",
            "experiment",
            "resume",
        ],
        required=True,
        help="'tune' for model hyperparams tuning; \
            'experiment' for experiments with tuned model; \
                'resume' for resuming interrupted experiment",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=for_resume,
        help="Path to the interrupted experiment \
            (e.g., /Users/user/mlp_project/assets/logs/prefix-mode-model-ds), \
                used in 'resume' mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=models,
        required=not for_resume,
        help="Name of the model to run",
    )
    parser.add_argument(
        "--ds",
        type=str,
        choices=datasets,
        required=not for_resume,
        help="Name of the dataset to use for training",
    )

    parser.add_argument(
        "--test-ds",
        nargs="*",
        type=str,
        choices=datasets,
        help="Additional datasets for testing",
    )

    # some datasets have multiple classes; set to true if you want to load all classes
    boolean_flag(parser, "multiclass", default=False)

    # whehter dataset should be z-scored over time
    boolean_flag(parser, "scaled", default=False)

    parser.add_argument(
        "--prefix",
        type=str,
        help="Prefix for the project name (body of the project name \
            is '$mode-$model-$dataset'): default: UTC time",
    )

    parser.add_argument(
        "--num-trials",
        type=int,
        default=50 if tune else 10,
        help="Number of trials to run on each test fold",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=5,
        help="Number of splits for StratifiedKFold (affects the number of test folds)",
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Max number of epochs (min 30)",
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

    args = parser.parse_args()

    Experiment(
        mode=args.mode,
        path=args.path,
        model=args.model,
        dataset=args.ds,
        test_datasets=args.test_ds,
        multiclass=args.multiclass,
        scaled=args.scaled,
        prefix=args.prefix,
        n_splits=args.num_splits,
        n_trials=args.num_trials,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    ).start()
