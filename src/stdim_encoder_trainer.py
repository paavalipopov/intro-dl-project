# pylint: disable=W0201,W0223,C0103,C0115,C0116,R0902,E1101,R0914
"""Main training script for STDIM model"""
from apto.utils.report import get_classification_report
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from animus import EarlyStoppingCallback, IExperiment
from animus.torch.callbacks import TorchCheckpointerCallback
import wandb

from src.settings import LOGS_ROOT, UTCNOW


class EncoderTrainer(IExperiment):
    """
    Animus-based encoder training script.
    """

    def __init__(
        self,
        encoder_config: dict,
        dataset: dict,
        logpath: str,
        wandb_logger: wandb.run,
        device,
        batch_size: int = 32,
        max_epochs: int = 200,
    ) -> None:
        super().__init__()
        # deferred import - handling import cycle
        from src.ts_model import NatureOneCNN

        self.encoder_config = encoder_config
        self.encoder = NatureOneCNN(encoder_config)

        self.original_dataset = dataset

        self.logpath = logpath
        self.wandb_logger = wandb_logger

        self.batch_size = batch_size
        self.num_epochs = max_epochs

        self.device = device

    def initialize_data(self):
        # prepare dataset for the encoder training
        # original dataset has shape [subjects; samples_per_subject; sample_x; sample_y;]

        dataset = {
            "train": {
                "t": [],
                "t-1": [],
            },
            "val": {
                "t": [],
                "t-1": [],
            },
        }
        for name in ["train", "val"]:
            orig = self.original_dataset[name]

            for i in range(orig.shape[0]):
                for j in range(1, orig.shape[1]):
                    dataset[name]["t"].append(orig[i, j, :, :])
                    dataset[name]["t-1"].append(orig[i, j - 1, :, :])

            dataset[name]["t"] = np.stack(dataset[name]["t"])
            dataset[name]["t-1"] = np.stack(dataset[name]["t-1"])

        # new data shape; [all_time_slices; sample_x; sample_y;]
        self._train_ds = TensorDataset(
            torch.tensor(dataset["train"]["t"], dtype=torch.float32),
            torch.tensor(dataset["train"]["t-1"], dtype=torch.float32),
        )
        self._valid_ds = TensorDataset(
            torch.tensor(dataset["val"]["t"], dtype=torch.float32),
            torch.tensor(dataset["val"]["t-1"], dtype=torch.float32),
        )

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)

        print("Training encoder")

        # init data
        self.initialize_data()

        # init classifiers and optimizers
        self.classifier1 = nn.Linear(
            self.encoder.feature_size, self.encoder.local_layer_depth
        )  # global-local
        self.classifier2 = nn.Linear(
            self.encoder.local_layer_depth, self.encoder.local_layer_depth
        )  # local-local
        self.encoder = self.encoder.to(self.device)
        self.classifier1 = self.classifier1.to(self.device)
        self.classifier2 = self.classifier2.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            list(self.classifier1.parameters())
            + list(self.encoder.parameters())
            + list(self.classifier2.parameters()),
            lr=self.encoder_config["lr"],
            eps=1e-5,
        )

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
                patience=15,
                dataset_key="valid",
                metric_key="loss",
                min_delta=0.001,
            ),
            "checkpointer": TorchCheckpointerCallback(
                exp_attr="encoder",
                logdir=self.logpath,
                dataset_key="valid",
                metric_key="loss",
                minimize=True,
            ),
        }

    def run_dataset(self) -> None:
        total_loss = 0.0

        self.encoder.train(self.is_train_dataset)
        self.classifier1.train(self.is_train_dataset)
        self.classifier2.train(self.is_train_dataset)

        with torch.set_grad_enabled(self.is_train_dataset):
            for self.dataset_batch_step, (x_t, x_tprev) in enumerate(
                tqdm(self.dataset)
            ):
                x_t, x_tprev = x_t.to(self.device), x_tprev.to(self.device)

                self.optimizer.zero_grad()
                f_t_maps, f_t_prev_maps = (
                    self.encoder(x_t, fmaps=True),
                    self.encoder(x_tprev, fmaps=True),
                )
                # print("Original shape: ", x_t.shape, x_tprev.shape)
                # for key in f_t_maps.keys():
                #     print(
                #         f"f_maps[{key}] shape: ",
                #         f_t_maps[key].shape,
                #         f_t_prev_maps[key].shape,
                #     )
                # # print()

                # Loss 1: Global at time t, f5 patches at time t-1
                f_t, f_t_prev = f_t_maps["out"], f_t_prev_maps["f5"]

                sx = f_t_prev.size(1)
                # print("sx shape: ", sx)

                N = f_t.size(0)
                loss1 = 0.0
                for x in range(sx):
                    predictions = self.classifier1(f_t)
                    # print("predictions1 shape: ", predictions.size())
                    positive = f_t_prev[:, x, :]
                    # print("positive1 shape: ", positive.size())
                    logits = torch.matmul(predictions, positive.t())
                    # print("logits1 shape: ", logits.size())
                    # print("torch.arange1 shape: ", torch.arange(N).size())
                    step_loss = self.criterion(logits, torch.arange(N).to(self.device))
                    loss1 += step_loss
                loss1 = loss1 / sx

                # Loss 2: f5 patches at time t, with f5 patches at time t-1
                f_t = f_t_maps["f5"]
                loss2 = 0.0
                for x in range(sx):
                    predictions = self.classifier2(f_t[:, x, :])
                    # print("predictions2 shape: ", predictions.size())
                    positive = f_t_prev[:, x, :]
                    # print("positive2 shape: ", positive.size())
                    logits = torch.matmul(predictions, positive.t())
                    # print("logits2 shape: ", logits.size())
                    # print("torch.arange1 shape: ", torch.arange(N).size())
                    step_loss = self.criterion(logits, torch.arange(N).to(self.device))
                    loss2 += step_loss
                loss2 = loss2 / sx

                loss = loss1 + loss2
                total_loss += loss.sum().item()
                if self.is_train_dataset:
                    loss.backward()
                    self.optimizer.step()

        total_loss /= self.dataset_batch_step + 1
        self.dataset_metrics = {
            "loss": total_loss,
        }

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(self)
        self.wandb_logger.log(
            {
                "encoder_train_loss": self.epoch_metrics["train"]["loss"],
                "encoder_valid_loss": self.epoch_metrics["valid"]["loss"],
            },
        )

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)

        print("Done training encoder")

        print("Best loss ", self.callbacks["early-stop"].best_score)

    def train_encoder(self):
        self.run()

        # load best encoder
        logpath = f"{self.logpath}/encoder.best.pth"
        checkpoint = torch.load(logpath, map_location=lambda storage, loc: storage)
        self.encoder.load_state_dict(checkpoint)

        return self.encoder
