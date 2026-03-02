"""Logger factory"""
import wandb


def logger_factory(conf, model_config):
    """Basic logger factory"""
    logger = wandb.init(
        project=conf.project_name,
        name=conf.wandb_trial_name,
        save_code=True,
    )

    # save tuning process wandb link
    if conf.mode == "tune":
        link = logger.get_url()
    else:
        if "link" in model_config:
            link = model_config["link"]
        else:
            link = None

    return logger, link
