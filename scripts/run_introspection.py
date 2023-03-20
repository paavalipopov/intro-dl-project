# pylint: disable=too-many-statements, too-many-locals, invalid-name, unbalanced-tuple-unpacking
"""Script for running experiments: tuning and testing hypertuned models"""
import os
import json

from src.utils import get_introspection_argparser, get_introspection_params, NpEncoder
from src.data import data_factory, data_postfactory
from src.naming import introspection_project_name, introspection_run_name
from src.model_config import model_config_factory
from src.dataloader import dataloader_factory
from src.model import model_factory
from src.introspection import introspector_factory


os.environ["WANDB_SILENT"] = "true"


def start(conf):
    """Main script for trained model introspection"""
    (
        conf.project_name,
        conf.project_dir,
        conf.weights_dir,
    ) = introspection_project_name(conf)

    conf.n_splits, conf.n_trials = get_introspection_params(conf)

    # save config
    os.makedirs(conf.project_dir, exist_ok=True)
    with open(conf.project_dir + "general_config.json", "w", encoding="utf8") as fp:
        json.dump(vars(conf), fp, indent=2, cls=NpEncoder)

    original_data, conf.data_info = data_factory(conf)

    for outer_k in range(conf.n_splits):
        print(f"Introspecting model for k: {outer_k:02d}")

        conf.wandb_trial_name, conf.run_dir = introspection_run_name(conf, outer_k)
        os.makedirs(conf.run_dir, exist_ok=True)

        model_config = model_config_factory(conf, k=outer_k)
        data, conf.data_info = data_postfactory(
            conf,
            model_config,
            original_data,
        )
        dataloaders = dataloader_factory(conf, data, outer_k)
        model = model_factory(conf, model_config)

        introspector = introspector_factory(
            conf,
            model_config,
            dataloaders,
            model,
        )
        introspector.run(cutoff=10)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    parser = get_introspection_argparser()
    args = parser.parse_args()

    start(args)
