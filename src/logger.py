import wandb


def logger_factory(conf, model_config, mode):
    logger = wandb.init(
        project=conf.project_name,
        name=conf.trial_name,
        save_code=True,
    )
    if mode == "tune":
        link = logger.get_url()
    else:
        link = model_config["link"]

    return logger, link
