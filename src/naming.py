from src.settings import LOGS_ROOT, UTCNOW


def name_factory(conf, mode, outer_k, trial, inner_k=None):
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
    if conf.scaled:
        proj_dataset_name = f"scaled_{proj_dataset_name}"

    project_name = f"{conf.prefix}-{mode}-{conf.model}-{proj_dataset_name}"
    if conf.test_ds is not None:
        if len(conf.test_ds) != 0:
            project_ending = "-test_ds_" + "_".join(conf.test_ds)
            project_name += project_ending

    # set trial name and directory
    if conf.mode != "tune":
        trial_name = f"k_{outer_k:02d}-trial_{trial:04d}"
        if inner_k is not None:
            trial_name += f"-inK_{inner_k:02d}"

        trial_dir = f"{LOGS_ROOT}/{project_name}/k_{outer_k:02d}/trial_{trial:04d}"
        if inner_k is not None:
            trial_dir += f"/inK_{inner_k:02d}"
    else:
        trial_name = f"trial_{trial:04d}-k_{outer_k:02d}"
        trial_dir = f"{LOGS_ROOT}/{project_name}/trial_{trial:04d}/k_{outer_k:02d}/"

    return project_name, trial_name, trial_dir
