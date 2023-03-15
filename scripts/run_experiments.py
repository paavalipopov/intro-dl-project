# pylint: disable=too-many-statements, too-many-locals, invalid-name, unbalanced-tuple-unpacking
"""Script for running experiments: tuning and testing hypertuned models"""
import sys
import os
import json

import pandas as pd
import numpy as np

from src.utils import get_argparser, get_resumed_params, NpEncoder
from src.data import data_factory, data_postfactory
from src.naming import project_name, run_name
from src.model_config import model_config_factory
from src.logger import logger_factory
from src.dataloader import dataloader_factory
from src.model import model_factory
from src.optimizer import optimizer_factory
from src.criterion import criterion_factory
from src.trainer import trainer_factory


os.environ["WANDB_SILENT"] = "true"


def start(conf):
    """Main script for starting the experiments"""
    conf.project_name, conf.project_dir = project_name(conf)
    if conf.mode == "resume":
        assert conf.path is not None
        conf, start_k, start_trial = get_resumed_params(conf)
    else:
        start_k, start_trial = 0, 0

    # save config
    os.makedirs(conf.project_dir, exist_ok=True)
    with open(conf.project_dir + "general_config.json", "w", encoding="utf8") as fp:
        json.dump(vars(conf), fp, indent=2, cls=NpEncoder)

    original_data, conf.data_info = data_factory(conf)

    if conf.mode == "tune":
        # outer CV: for each test set, we are looking for a unique set of optimal hyperparams
        for outer_k in range(start_k, conf.n_splits):
            # num trials: number of hyperparams sets to test
            if outer_k != start_k:
                start_trial = 0
            for trial in range(start_trial, conf.n_trials):
                model_config = model_config_factory(conf)
                # some models require data postprocessing (based on their config)
                data, conf.data_info = data_postfactory(
                    conf,
                    model_config,
                    original_data,
                )

                # inner CV: CV of the chosen hyperparams
                for inner_k in range(conf.n_splits):
                    print(
                        f"Running tune: k: {outer_k:02d}, Trial: {trial:03d}, \
                            Inner k: {inner_k:02d},"
                    )

                    (
                        conf.wandb_trial_name,
                        conf.outer_k_dir,
                        conf.trial_dir,
                        conf.run_dir,
                    ) = run_name(conf, outer_k, trial, inner_k)
                    os.makedirs(conf.run_dir, exist_ok=True)

                    dataloaders = dataloader_factory(
                        conf, data, outer_k, trial, inner_k
                    )
                    model = model_factory(conf, model_config)
                    optimizer = optimizer_factory(conf, model, model_config)
                    criterion = criterion_factory(conf)

                    logger, model_config["link"] = logger_factory(conf, model_config)

                    trainer = trainer_factory(
                        conf,
                        model_config,
                        dataloaders,
                        model,
                        optimizer,
                        criterion,
                        logger,
                    )
                    results = trainer.run()
                    # save results of nested CV
                    df = pd.DataFrame(results, index=[0])
                    with open(conf.trial_dir + "runs.csv", "a", encoding="utf8") as f:
                        df.to_csv(f, header=f.tell() == 0, index=False)

                    logger.finish()

                # save model config
                with open(
                    conf.trial_dir + "model_config.json", "w", encoding="utf8"
                ) as fp:
                    json.dump(model_config, fp, indent=2, cls=NpEncoder)
                # read inner CV results, save the average in the fold dir
                df = pd.read_csv(conf.trial_dir + "runs.csv")
                score = np.mean(df["test_score"].to_numpy())
                loss = np.mean(df["test_average_loss"].to_numpy())
                df = pd.DataFrame(
                    {
                        "trial": trial,
                        "score": score,
                        "loss": loss,
                        "path_to_config": conf.trial_dir + "model_config.json",
                    },
                    index=[0],
                )
                with open(conf.outer_k_dir + "runs.csv", "a", encoding="utf8") as f:
                    df.to_csv(f, header=f.tell() == 0, index=False)

    elif conf.mode == "exp":
        # outer CV: for each test set, we are loading a unique set of optimal hyperparams
        for outer_k in range(start_k, conf.n_splits):
            # loading best config requires project_name
            model_config = model_config_factory(conf, k=outer_k)

            # some models require data postprocessing (based on their config)
            data, conf.data_info = data_postfactory(
                conf,
                model_config,
                original_data,
            )

            # num trials: for each test set, we are splitting training set into train/val randomly
            if outer_k != start_k:
                start_trial = 0
            for trial in range(start_trial, conf.n_trials):
                print(f"Running exp: k: {outer_k:02d}, Trial: {trial:03d}")

                (conf.wandb_trial_name, conf.outer_k_dir, conf.run_dir) = run_name(
                    conf, outer_k, trial
                )
                os.makedirs(conf.run_dir, exist_ok=True)
                with open(
                    conf.outer_k_dir + "model_config.json", "w", encoding="utf8"
                ) as fp:
                    json.dump(model_config, fp, indent=2, cls=NpEncoder)

                dataloaders = dataloader_factory(conf, data, outer_k, trial)
                model = model_factory(conf, model_config)
                optimizer = optimizer_factory(conf, model, model_config)
                criterion = criterion_factory(conf)

                logger, model_config["link"] = logger_factory(conf, model_config)

                trainer = trainer_factory(
                    conf,
                    model_config,
                    dataloaders,
                    model,
                    optimizer,
                    criterion,
                    logger,
                )
                results = trainer.run()
                # save results of the trial
                df = pd.DataFrame(results, index=[0])
                with open(conf.project_dir + "runs.csv", "a", encoding="utf8") as f:
                    df.to_csv(f, header=f.tell() == 0, index=False)

                logger.finish()

    else:
        raise ValueError(f"{conf.model} is not recognized")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    parser = get_argparser(sys.argv)
    args = parser.parse_args()

    start(args)
