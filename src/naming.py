from src.settings import LOGS_ROOT, UTCNOW


def project_name(conf):
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

    project_name = f"{conf.prefix}-{conf.mode}-{conf.model}-{proj_dataset_name}"
    #### not applicable for this project
    if conf.test_ds is not None:
        if len(conf.test_ds) != 0:
            project_ending = "-test-" + "_".join(conf.test_ds)
            project_name += project_ending
    ####

    project_dir = f"{LOGS_ROOT}/{project_name}/"

    return project_name, project_dir


def run_name(conf, outer_k, trial=None, inner_k=None):
    if conf.mode == "exp":
        wandb_trial_name = f"k_{outer_k:02d}-trial_{trial:04d}"
        outer_k_dir = f"{conf.project_dir}k_{outer_k:02d}/"
        run_dir = f"{outer_k_dir}trial_{trial:04d}/"

        return wandb_trial_name, outer_k_dir, run_dir
    elif conf.mode == "tune":
        wandb_trial_name = f"k_{outer_k:02d}-trial_{trial:04d}-inK_{inner_k:02d}"
        outer_k_dir = f"{conf.project_dir}k_{outer_k:02d}/"
        trial_dir = f"{outer_k_dir}trial_{trial:04d}/"
        run_dir = f"{trial_dir}inK_{inner_k:02d}/"

        return wandb_trial_name, outer_k_dir, trial_dir, run_dir
    else:
        raise NotImplementedError()
