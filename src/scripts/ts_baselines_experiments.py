# pylint: disable=W0201,W0223,C0103,C0115,C0116,R0902,E1101,R0914
"""Script for tuning different models on different datasets"""
import argparse
import os

from apto.utils.misc import boolean_flag
from apto.utils.report import get_classification_report
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

from animus import IExperiment
import wandb

from src.settings import LOGS_ROOT, ASSETS_ROOT, UTCNOW
from src.ts_data import (
    load_ABIDE1,
    load_COBRE,
    load_FBIRN,
    load_OASIS,
    load_ABIDE1_869,
    load_UKB,
    TSQuantileTransformer,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier


class Experiment(IExperiment):
    def __init__(
        self,
        model: str,
        dataset: str,
        quantile: bool,
        n_splits: int,
        logdir: str,
    ) -> None:
        super().__init__()
        assert not quantile, "Not implemented yet"
        self._model = model
        self._dataset = dataset
        self._quantile: bool = quantile
        self.n_splits = n_splits
        self._trial: optuna.Trial = None
        self.logdir = logdir

        os.makedirs(self.logdir)

    def initialize_dataset(self) -> None:
        if self._dataset == "oasis":
            features, labels = load_OASIS()
        elif self._dataset == "abide":
            features, labels = load_ABIDE1()
        elif self._dataset == "fbirn":
            features, labels = load_FBIRN()
        elif self._dataset == "cobre":
            features, labels = load_COBRE()
        elif self._dataset == "abide_869":
            features, labels = load_ABIDE1_869()
        elif self._dataset == "ukb":
            features, labels = load_UKB()

        self.data_shape = features.shape
        print("data shape: ", self.data_shape)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        skf.get_n_splits(features, labels)

        train_index, test_index = list(skf.split(features, labels))[self.k]

        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        rng = np.random.default_rng(42 + self._trial.number)
        shuffle_array = rng.permutation(X_train.shape[0])
        X_train = X_train[shuffle_array, :, :]
        y_train = y_train[shuffle_array]

        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], -1))

        self.datasets = {
            self._dataset: (X_train, X_test, y_train, y_test),
        }

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)

        # init data
        self.initialize_dataset()

        # init wandb logger
        self.wandb_logger: wandb.run = wandb.init(
            project=f"tune-{self._model}-{self._dataset}",
            name=f"{UTCNOW}-k_{self.k}-trial_{self._trial.number}",
        )
        # config dict for wandb
        wandb_config = {}

        # setup model

        if self._model == "lr":
            solver = self._trial.suggest_categorical(
                "classifier.logistic.solver", ["liblinear", "lbfgs"]
            )
            decay = self._trial.suggest_loguniform(
                "classifier.logistic.C", low=1e-3, high=1e3
            )
            if solver == "liblinear":
                penalty = self._trial.suggest_categorical(
                    "classifier.logistic.penalty", ["l1", "l2"]
                )
            else:
                penalty = "l2"

            self.model = LogisticRegression(
                solver=solver,
                C=decay,
                penalty=penalty,
                max_iter=1000,
            )

            wandb_config = {
                "solver": solver,
                "decay": decay,
                "penalty": penalty,
            }

        elif self._model == "svm":
            penalty = self._trial.suggest_categorical(
                "classifier.sgd.penalty", ["l1", "l2", "elasticnet"]
            )
            alpha = self._trial.suggest_loguniform(
                "classifier.sgd.alpha", low=1e-4, high=1e-2
            )
            self.model = SGDClassifier(
                loss="modified_huber",
                penalty=penalty,
                alpha=alpha,
                max_iter=1000,
                tol=1e-3,
            )

            wandb_config = {
                "penalty": penalty,
                "alpha": alpha,
            }

        else:
            raise NotImplementedError()

        self.wandb_logger.config.update(wandb_config)

    def run_dataset(self) -> None:
        X_train, X_test, y_train, y_test = self.dataset

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_score = self.model.predict_proba(X_test)

        report = get_classification_report(
            y_true=y_test, y_pred=y_pred, y_score=y_score, beta=0.5
        )
        for stats_type in [0, 1, "macro", "weighted"]:
            stats = report.loc[stats_type]
            for key, value in stats.items():
                if "support" not in key:
                    self._trial.set_user_attr(f"{key}_{stats_type}", float(value))

        self.dataset_metrics = {
            "accuracy": report["precision"].loc["accuracy"],
            "score": report["auc"].loc["weighted"],
        }

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)
        self._score = self.experiment_metrics[1][self._dataset]["score"]

        self.wandb_logger.log(
            {
                "test_accuracy": self.experiment_metrics[1][self._dataset]["accuracy"],
                "test_score": self.experiment_metrics[1][self._dataset]["score"],
            },
        )

        self.wandb_logger.finish()

    def _objective(self, trial) -> float:
        self._trial = trial
        self.run()
        return self._score

    def tune(self, n_trials: int):
        for k in range(self.n_splits):
            self.k = k
            self.study = optuna.create_study(direction="maximize")
            self.study.optimize(self._objective, n_trials=n_trials, n_jobs=1)
            logfile = f"{self.logdir}/optuna.csv"
            df = self.study.trials_dataframe()
            df.to_csv(logfile, index=False)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "lr",
            "svm",
        ],
        required=True,
    )
    parser.add_argument(
        "--ds",
        type=str,
        choices=["oasis", "abide", "fbirn", "cobre", "abide_869", "ukb"],
        required=True,
    )
    boolean_flag(parser, "quantile", default=False)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--num-splits", type=int, default=5)
    args = parser.parse_args()

    Experiment(
        model=args.model,
        dataset=args.ds,
        quantile=args.quantile,
        n_splits=args.num_splits,
        logdir=f"{LOGS_ROOT}/{UTCNOW}-tune-{args.model}-{args.ds}/",
    ).tune(n_trials=args.num_trials)
