"""Functions for setting names and paths"""
from src.settings import LOGS_ROOT, UTCNOW


def project_name(conf):
    """return project name and poject dir based on config"""
    # set project name
    conf.default_prefix = f"{UTCNOW}"
    if conf.prefix is None:
        conf.prefix = conf.default_prefix
    else:
        if len(conf.prefix) == 0:
            conf.prefix = conf.default_prefix
        else:
            # '-'s are reserved for project name parsing
            conf.prefix = conf.prefix.replace("-", "_")

    proj_dataset_name = conf.ds
    if conf.multiclass:
        proj_dataset_name = f"multiclass_{proj_dataset_name}"
    if conf.zscore:
        proj_dataset_name = f"zscore_{proj_dataset_name}"

    proj_name = f"{conf.prefix}-{conf.mode}-{conf.model}-{proj_dataset_name}"
    #### not applicable for this project
    if conf.test_ds is not None:
        if len(conf.test_ds) != 0:
            project_ending = "-test-" + "_".join(conf.test_ds)
            proj_name += project_ending
    ####

    project_dir = f"{LOGS_ROOT}/{proj_name}/"

    return proj_name, project_dir


def run_name(conf, outer_k, trial=None, inner_k=None):
    """return wandb run name, and experiment directories"""
    if conf.mode == "exp":
        wandb_trial_name = f"k_{outer_k:02d}-trial_{trial:04d}"
        outer_k_dir = f"{conf.project_dir}k_{outer_k:02d}/"
        run_dir = f"{outer_k_dir}trial_{trial:04d}/"

        return wandb_trial_name, outer_k_dir, run_dir
    if conf.mode == "tune":
        wandb_trial_name = f"k_{outer_k:02d}-trial_{trial:04d}-inK_{inner_k:02d}"
        outer_k_dir = f"{conf.project_dir}k_{outer_k:02d}/"
        trial_dir = f"{outer_k_dir}trial_{trial:04d}/"
        run_dir = f"{trial_dir}inK_{inner_k:02d}/"

        return wandb_trial_name, outer_k_dir, trial_dir, run_dir

    raise ValueError(f"{conf.mode} mode is not recognized")
