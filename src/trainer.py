# pylint: disable=no-member, too-many-locals, too-many-arguments, too-many-instance-attributes, invalid-name, attribute-defined-outside-init
"""Training scripts"""
import os
import json
import time

import numpy as np
import torch

from tqdm import tqdm
from apto.utils.report import get_classification_report

from src.utils import EarlyStopping, NpEncoder


def trainer_factory(
    conf, model_config, dataloaders, model, criterion, optimizer, scheduler, logger
):
    """Trainer factory"""
    if conf.model in ["lstm", "mean_lstm", "transformer", "mean_transformer", "dice"]:
        trainer = Trainer(
            vars(conf),
            model_config,
            dataloaders,
            model,
            criterion,
            optimizer,
            scheduler,
            logger,
        )
    else:
        raise ValueError(f"{conf.model} is not recognized")

    return trainer


class Trainer:
    """Basic training script"""

    def __init__(
        self,
        conf,
        model_conf,
        dataloaders,
        model,
        criterion,
        optimizer,
        scheduler,
        logger,
    ) -> None:

        self.config = conf
        self.model_config = model_conf
        self.dataloaders = dataloaders
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger

        params = self.count_params(self.model)
        self.logger.summary["params"] = params

        self.epochs = self.config["max_epochs"]
        self.save_path = self.config["run_dir"]

        self.early_stopping = EarlyStopping(
            path=self.save_path,
            minimize=True,
            patience=self.config["patience"],
        )

        # set device
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        print(f"Used device: {dev}")
        self.config["device"] = dev
        self.device = torch.device(dev)

        self.model = model.to(self.device)

        # log configs
        self.logger.config.update({"general": self.config})
        self.logger.config.update({"model": self.model_config})

        # save model configs in the run's directory
        with open(self.save_path + "model_config.json", "w", encoding="utf8") as fp:
            json.dump(self.model_config, fp, indent=2, cls=NpEncoder)

    def count_params(self, model, only_requires_grad: bool = False):
        "count number trainable parameters in a pytorch model"
        if only_requires_grad:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            total_params = sum(p.numel() for p in model.parameters())
        return total_params

    def run_epoch(self, ds_name):
        """Run single epoch on `ds_name` dataloder"""
        is_train_dataset = ds_name == "train"

        all_scores, all_targets = [], []
        total_loss, total_size = 0.0, 0

        self.model.train(is_train_dataset)
        start_time = time.time()

        with torch.set_grad_enabled(is_train_dataset):
            for data, target in self.dataloaders[ds_name]:
                data, target = data.to(self.device), target.to(self.device)
                total_size += data.shape[0]

                logits = self.model(data)
                loss = self.criterion(logits, target, self.model, self.device)
                score = torch.softmax(logits, dim=-1)

                all_scores.append(score.cpu().detach().numpy())
                all_targets.append(target.cpu().detach().numpy())
                total_loss += loss.sum().item()

                if is_train_dataset:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        average_time = (time.time() - start_time) / total_size
        average_loss = total_loss / total_size

        y_test = np.hstack(all_targets)
        y_score = np.vstack(all_scores)
        y_pred = np.argmax(y_score, axis=-1).astype(np.int32)

        report = get_classification_report(
            y_true=y_test, y_pred=y_pred, y_score=y_score, beta=0.5
        )

        metrics = {
            ds_name + "_accuracy": report["precision"].loc["accuracy"],
            ds_name + "_score": report["auc"].loc["weighted"],
            ds_name + "_average_loss": average_loss,
            ds_name + "_average_time": average_time,
        }

        return metrics

    def train(self):
        """Start training"""
        start_time = time.time()

        for epoch in tqdm(range(self.epochs)):
            results = self.run_epoch("train")
            results.update(self.run_epoch("valid"))

            self.logger.log(results)

            self.scheduler.step(results["valid_average_loss"])

            self.early_stopping(results["valid_average_loss"], self.model, epoch)
            if self.early_stopping.early_stop:
                break

        if self.early_stopping.early_stop:
            print("EarlyStopping triggered")

        elapsed_time = time.time() - start_time
        self.logger.summary["training_time"] = elapsed_time

    def test(self):
        """Start testing"""
        for key in self.dataloaders:
            if key not in ["train", "valid"]:
                results = self.run_epoch(key)

                self.test_results.update(results)

        self.logger.log(self.test_results)

    def run(self):
        """Run training script"""
        print("Training model")
        self.train()

        print("Loading best model")
        model_logpath = f"{self.save_path}best_model.pt"
        checkpoint = torch.load(
            model_logpath, map_location=lambda storage, loc: storage
        )
        self.model.load_state_dict(checkpoint)

        print("Testing trained model")
        self.test_results = {}
        self.test()
        print(f"Test results: {self.test_results}")
        print("Done!")

        if not self.config["preserve_checkpoints"]:
            os.remove(f"{self.save_path}best_model.pt")

        return self.test_results
