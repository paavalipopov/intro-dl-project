"""Logger factory"""
import wandb


def logger_factory(conf, model_config):
    """Basic logger factory"""
    logger = wandb.init(
        project=conf.project_name,
        name=conf.wandb_trial_name,
        save_code=True,
    )
    if conf.mode == "tune":
        link = logger.get_url()
    else:
        link = model_config["link"]

    return logger, link
